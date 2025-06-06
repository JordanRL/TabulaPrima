import math
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
import torch.nn.functional as F
import os
import time

import wandb
from hydra.utils import to_absolute_path

from config_schema import TrainingConfig
from console import TPConsole, Colors

@dataclass
class EvaluationResult:
    eval_name: str
    loss: float
    accuracy: float
    perplexity: float

class TrainingPhases(Enum):
    WARMUP = "warmup"
    CORE_LEARNING = "core learning"
    GENERALIZATION = "generalization"
    ABSTRACTION = "abstraction"
    REFINEMENT = "refinement"

    @classmethod
    def next_phase(cls, phase: "TrainingPhases|None") -> "TrainingPhases|None":
        if phase == cls.WARMUP:
            return cls.CORE_LEARNING
        elif phase == cls.CORE_LEARNING:
            return cls.GENERALIZATION
        elif phase == cls.GENERALIZATION:
            return cls.ABSTRACTION
        elif phase == cls.ABSTRACTION:
            return cls.REFINEMENT
        elif phase == cls.REFINEMENT:
            return None

        return None

    @classmethod
    def is_final_phase(cls, phase: "TrainingPhases|None") -> bool:
        return phase == cls.REFINEMENT

    @classmethod
    def is_first_phase(cls, phase: "TrainingPhases|None") -> bool:
        return phase == cls.WARMUP

    @classmethod
    def next_phase_threshold(cls, phase: "TrainingPhases|None") -> float:
        if phase == cls.WARMUP:
            return 0.0
        elif phase == cls.CORE_LEARNING:
            return 0.25
        elif phase == cls.GENERALIZATION:
            return 0.55
        elif phase == cls.ABSTRACTION:
            return 0.80
        else:
            return 1.0

@dataclass
class TrainingState:
    # Training metrics
    loss_history: list[float]
    token_history: list[int]
    token_times: list[float]
    grad_norm_history: list[float]
    eval_history: list[EvaluationResult]
    current_loss: float
    current_perplexity: float
    tokens_per_sec: float
    tokens_per_batch: float

    # Training progress
    batches_seen: int
    tokens_seen: int
    epochs: int
    optimizer_steps: int
    inference_steps: int
    time_of_last_instrument: float
    steps_since_instrument: int
    steps_since_eval: int
    total_instrument_events: int
    total_eval_events: int
    time_per_instrument: float
    steps_per_instrument: int
    steps_per_eval: int
    steps_per_gradient_update: int

    # Mode & phase
    run_phase: TrainingPhases|None
    run_status: str
    checkpoint_interval: int
    use_amp: bool
    allow_amp_switchover: bool
    stability_steps: int
    stability_reached: bool
    use_time_based_instrument: bool

    # Optional
    scheduler_mode: Literal["tokens", "epochs"]|None = None
    target_tokens: int = 0
    target_epochs: int = 0

    def should_instrument(self):
        if self.use_time_based_instrument:
            return (
                    time.time() - self.time_of_last_instrument >= self.time_per_instrument
                    and self.steps_since_instrument > 0
                    and self.optimizer_steps >= self.steps_per_gradient_update
            ) or self.should_eval()
        else:
            return self.steps_since_instrument >= self.steps_per_instrument or self.should_eval()

    def should_eval(self):
        return self.steps_since_eval >= self.steps_per_eval and self.run_phase != "warmup"

    def finish_batch(self, micro_batch_size):
        self.batches_seen += micro_batch_size
        self.step_inference()

    def finish_epoch(self):
        self.epochs += 1

    def get_instrument_dict(self, current_lr):
        grad_norm = sum(self.grad_norm_history[-self.steps_per_instrument:]) / self.steps_per_instrument
        return {
            "training/loss": self.current_loss,
            "training/perplexity": self.current_perplexity,
            "training/learning_rate": current_lr,
            "training/tokens_per_second": self.tokens_per_sec,
            "training/grad_norm": grad_norm,
            "metric/progress": self.get_progress()
        }

    def get_progress(self):
        return self.tokens_seen / self.target_tokens

    def get_current_precision_mode(self):
        if self.use_amp or (self.use_amp == False and self.allow_amp_switchover == True and self.stability_reached == True):
            return self.get_mp_support()
        else:
            return "FP32"

    def get_mp_support(self):
        if torch.cuda.is_bf16_supported():
            if torch.cuda.is_bf16_supported(False):
                return "BF16"
            else:
                return "BF16Emu"
        else:
            return "FP16"

    def update_precision_mode(self, start_time, current_grad_norm, warmup_ratio):
        if self.stability_reached:
            pass
        if current_grad_norm < 1.0:
            self.update_stability(start_time)
        elif current_grad_norm < 2.0 and self.tokens_seen / self.target_tokens > warmup_ratio * 0.5:
            self.update_stability(start_time, True)
        else:
            self.stability_steps = 0

    def update_stability(self, start_time, force_stability=False):
        self.stability_steps += 1
        self.stability_reached = self.stability_steps >= self.steps_per_gradient_update or force_stability

        if self.stability_reached:
            # Calculate training time estimate
            elapsed_time = time.time() - start_time

            # Format time estimate nicely
            if elapsed_time > 60:
                str_time = f"{elapsed_time % 60:.2f}s"
                elapsed_time = elapsed_time / 60
                if elapsed_time > 1:
                    str_time = f"{int(elapsed_time % 60)}m " + str_time
                    elapsed_time = elapsed_time / 60
                    if elapsed_time > 1:
                        str_time = f"{int(elapsed_time % 24)}h " + str_time
                        elapsed_time = elapsed_time / 24
                        if elapsed_time > 1:
                            str_time = f"{int(elapsed_time)}d " + str_time
            else:
                str_time = f"{elapsed_time:.2f}s"
            if not force_stability:
                TPConsole().print(Colors.success(
                    f"🎉 Stability Reached in {self.tokens_seen:,} tokens ({str_time})! Training continuing in {self.get_current_precision_mode()} mode. 🎉"
                ))
            else:
                TPConsole().print_warning(
                    f"Stability Forced in {self.tokens_seen:,} tokens ({str_time})... Training continuing in {self.get_current_precision_mode()} mode."
                )

    def update_run_status(self, run_status):
        self.run_status = run_status

    def update_run_phase(self, run_phase):
        self.run_phase = run_phase

    def step_optimizer(self):
        self.optimizer_steps += 1

    def step_inference(self):
        self.inference_steps += 1

    def increment_steps_since_instrument(self):
        self.steps_since_instrument += 1

    def reset_steps_since_instrument(self):
        self.steps_since_instrument = 0
        self.time_of_last_instrument = time.time()

    def increment_steps_since_eval(self):
        self.steps_since_eval += 1

    def reset_steps_since_eval(self):
        self.steps_since_eval = 0

    def update_grad_norm(self, grad_norm):
        self.grad_norm_history.append(grad_norm)

    def update_metrics(self, actual_tokens_in_batch, loss, micro_batch_size):
        # Update moving window metrics
        self.tokens_seen += actual_tokens_in_batch
        self.token_history.append(actual_tokens_in_batch)
        self.token_times.append(time.time())
        self.loss_history.append(loss)
        if len(self.token_history) > 10:
            self.token_history = self.token_history[-10:]
        if len(self.token_times) > 10:
            self.token_times = self.token_times[-10:]
        if len(self.loss_history) > 15:
            self.loss_history = self.loss_history[-15:]

        # Calculate tokens per second
        self.tokens_per_sec = sum(self.token_history) / (self.token_times[-1] - self.token_times[0]) if len(self.token_times) > 1 else 1

        # Calculate tokens per batch
        self.tokens_per_batch = sum(self.token_history) / (len(self.token_history)*micro_batch_size) if len(self.token_history) > 1 else 1

        # Calculate current perplexity (exp of loss)
        self.current_loss = sum(self.loss_history) / len(self.loss_history)
        self.current_perplexity = math.exp(min(self.current_loss, 20)) if len(self.loss_history) > 1 else float('inf')

        TPConsole().update_progress_task(
            "training",
            advance=actual_tokens_in_batch
        )

    def transition_phase(self):
        if TrainingPhases.is_first_phase(self.run_phase):
            if self.stability_reached:
                self.run_phase = TrainingPhases.next_phase(self.run_phase)
                return True
            else:
                return False
        elif TrainingPhases.is_final_phase(self.run_phase):
            return False
        elif self.tokens_seen > self.target_tokens * TrainingPhases.next_phase_threshold(self.run_phase):
            self.run_phase = TrainingPhases.next_phase(self.run_phase)
            return True

        return False

    def state_dict(self):
        return {
            "loss_history": self.loss_history,
            "token_history": self.token_history,
            "token_times": self.token_times,
            "grad_norm_history": self.grad_norm_history,
            "eval_history": self.eval_history,
            "batches_seen": self.batches_seen,
            "tokens_seen": self.tokens_seen,
            "optimizer_steps": self.optimizer_steps,
            "inference_steps": self.inference_steps,
            "steps_since_instrument": self.steps_since_instrument,
            "steps_since_eval": self.steps_since_eval,
            "run_phase": self.run_phase,
            "run_status": self.run_status,
            "stability_steps": self.stability_steps,
            "stability_reached": self.stability_reached,
            "checkpoint_interval": self.checkpoint_interval,
            "use_amp": self.use_amp,
            "current_loss": self.current_loss,
            "current_perplexity": self.current_perplexity,
            "tokens_per_sec": self.tokens_per_sec,
            "epochs": self.epochs,
            "steps_per_instrument": self.steps_per_instrument,
            "steps_per_eval": self.steps_per_eval,
            "allow_amp_switchover": self.allow_amp_switchover,
            "steps_per_gradient_update": self.steps_per_gradient_update,
            "tokens_per_batch": self.tokens_per_batch,
            "use_time_based_instrument": self.use_time_based_instrument,
            "time_per_instrument": self.time_per_instrument,
            "time_of_last_instrument": self.time_of_last_instrument,
            "total_instrument_events": self.total_instrument_events,
            "total_eval_events": self.total_eval_events,
            "target_tokens": self.target_tokens,
            "target_epochs": self.target_epochs,
            "scheduler_mode": self.scheduler_mode,
            "_state_dict_save_time": time.time(),
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        time_diff = time.time() - state_dict["_state_dict_save_time"]
        token_times = [token_time+time_diff for token_time in state_dict["token_times"]]
        return cls(
            loss_history=state_dict["loss_history"],
            token_history=state_dict["token_history"],
            token_times=token_times,
            grad_norm_history=state_dict["grad_norm_history"],
            eval_history=state_dict["eval_history"],
            batches_seen=state_dict["batches_seen"],
            tokens_seen=state_dict["tokens_seen"],
            optimizer_steps=state_dict["optimizer_steps"],
            inference_steps=state_dict["inference_steps"],
            steps_since_instrument=state_dict["steps_since_instrument"],
            steps_since_eval=state_dict["steps_since_eval"],
            run_phase=state_dict["run_phase"],
            run_status=state_dict["run_status"],
            stability_steps=state_dict["stability_steps"],
            stability_reached=state_dict["stability_reached"],
            checkpoint_interval=state_dict["checkpoint_interval"],
            use_amp=state_dict["use_amp"],
            current_loss=state_dict["current_loss"],
            current_perplexity=state_dict["current_perplexity"],
            tokens_per_sec=state_dict["tokens_per_sec"],
            epochs=state_dict["epochs"],
            steps_per_instrument=state_dict["steps_per_instrument"],
            steps_per_eval=state_dict["steps_per_eval"],
            allow_amp_switchover=state_dict["allow_amp_switchover"],
            steps_per_gradient_update=state_dict["steps_per_gradient_update"],
            tokens_per_batch=state_dict["tokens_per_batch"],
            use_time_based_instrument=state_dict["use_time_based_instrument"],
            time_per_instrument=state_dict["time_per_instrument"],
            time_of_last_instrument=state_dict["time_of_last_instrument"]+time_diff,
            total_instrument_events=state_dict["total_instrument_events"],
            total_eval_events=state_dict["total_eval_events"],
            target_tokens=state_dict["target_tokens"],
            target_epochs=state_dict["target_epochs"],
            scheduler_mode=state_dict["scheduler_mode"],
        )

class Trainer:
    cfg: TrainingConfig

    def __init__(
            self,
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            device,
            cfg,
            wandb_instance=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.training_state = TrainingState(
            loss_history=[],
            token_history=[],
            token_times=[],
            grad_norm_history=[],
            eval_history=[],
            batches_seen=0,
            tokens_seen=0,
            optimizer_steps=0,
            inference_steps=0,
            steps_since_instrument=0,
            steps_since_eval=0,
            run_phase=TrainingPhases.WARMUP,
            run_status="training",
            stability_steps=0,
            stability_reached=False,
            checkpoint_interval=60 * 60 * 4,
            use_amp=self.cfg.use_amp,
            current_loss=float('inf'),
            current_perplexity=float('inf'),
            tokens_per_sec=0,
            epochs=0,
            steps_per_instrument=self.cfg.log_interval,
            steps_per_eval=self.cfg.eval_interval,
            allow_amp_switchover=self.cfg.allow_amp_switchover,
            steps_per_gradient_update=self.cfg.grad_steps,
            tokens_per_batch=0,
            use_time_based_instrument=self.cfg.wandb.use_time_based_instrument,
            time_per_instrument=1/self.cfg.wandb.instruments_per_second if self.cfg.wandb.instruments_per_second != 0 else 1,
            time_of_last_instrument=0,
            total_instrument_events=0,
            total_eval_events=0,
        )
        self.wandb = wandb_instance
        self.scaler = torch.amp.GradScaler() if self.training_state.use_amp and not torch.cuda.is_bf16_supported() else None
        self.input_ids = None
        self.attention_mask = None
        self.labels = None
        self.last_checkpoint_time = None
        self.start_time = None
        self.console = TPConsole()

    def state_dict(self):
        return {
            "last_checkpoint_time": self.last_checkpoint_time,
            "start_time": self.start_time,
            "_state_dict_save_time": time.time(),
        }

    @classmethod
    def load_state_dict(
            cls,
            state_dict,
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            device,
            training_state,
            cfg,
            wandb=None
    ):
        trainer = cls(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            wandb_instance=wandb,
        )
        trainer.training_state = training_state
        trainer.last_checkpoint_time = time.time()
        trainer.start_time = state_dict["start_time"]
        trainer.scaler = torch.amp.GradScaler() if trainer.training_state.use_amp and not torch.cuda.is_bf16_supported() else None
        trainer.console = TPConsole()
        return trainer

    def run_tokens(self, target_tokens):
        self.console.rule(f"Training for {Colors.highlight(f'{target_tokens:,}')} tokens")

        self.model.train()
        self.training_state.scheduler_mode = "tokens"
        self.training_state.target_tokens = target_tokens

        self.console.create_progress_task(
            "training",
            f"Phase: {self.training_state.run_phase.value.title()} ({self.training_state.get_current_precision_mode()})",
            total=self.training_state.target_tokens
        )

        # Tracking training dynamics
        self.start_time = time.time()
        self.training_state.time_of_last_instrument = self.start_time
        self.last_checkpoint_time = self.start_time

        last_cli_update_time = time.time()
        last_cli_update_tokens = 0

        self.console.subrule("Training Started")

        while self.training_state.tokens_seen < self.training_state.target_tokens:
            for batch in self.train_dataloader:
                batch_result = self._process_batch(
                    batch=batch,
                )

                if not batch_result:
                    return self.training_state.tokens_seen, "failed"

                cli_interval = time.time() - last_cli_update_time
                if cli_interval > self.cfg.update_interval:
                    last_cli_update_time = time.time()
                    self.console.rule("Periodic Training Update Summary")
                    tokens_seen = self.training_state.tokens_seen - last_cli_update_tokens
                    last_cli_update_tokens = self.training_state.tokens_seen
                    summary = [
                        {"title": "Tokens Seen This Interval", "content": f"{tokens_seen:,}"},
                        {"title": "Current Loss", "content": f"{self.training_state.current_loss:.2f}"},
                        {"title": "Current Perplexity", "content": f"{self.training_state.current_perplexity:,.2f}"},
                        {"title": "Avg Tokens/Sec", "content": f"{tokens_seen/cli_interval:,.2f}"},
                        {"title": "Run Phase", "content": self.training_state.run_phase.value.title()},
                        {"title": "Precision Mode", "content": self.training_state.get_current_precision_mode()},
                    ]
                    self.console.print_list(summary)

                if self.training_state.tokens_seen >= self.training_state.target_tokens:
                    break

            self.training_state.finish_epoch()

        self.console.remove_progress_task("training")
        return self.training_state.tokens_seen, "success"

    def run_epochs(self, target_epochs):
        self.model.train()
        self.training_state.scheduler_mode = "epochs"
        self.training_state.target_epochs = target_epochs

        self.console.create_progress_task(
            "training",
            f"Phase: {self.training_state.run_phase.value.title()} ({self.training_state.get_current_precision_mode()})",
            total=target_epochs
        )

        # Tracking training dynamics
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time

        while self.training_state.epochs < self.training_state.target_epochs:
            for batch in self.train_dataloader:
                batch_result = self._process_batch(
                    batch=batch,
                )

                if not batch_result:
                    return self.training_state.tokens_seen, "failed"
            self.training_state.finish_epoch()

        self.console.remove_progress_task("training")
        return self.training_state.tokens_seen, "success"

    def save_checkpoint(self):
        self.console.rule(Colors.highlight(f"Saving Checkpoint"), style=Colors.HEADER)

        self.console.print_notification("Creating resumable checkpoint PyTorch model file")
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state_state_dict": self.training_state.state_dict(),
            "trainer_state_dict": self.state_dict(),
            "wandb_run_id": self.wandb.run.id if self.wandb is not None else None,
        }, checkpoint_path)
        self.console.print_complete(f"Checkpoint saved to file {Colors.header(checkpoint_path)}")
        if self.wandb is not None and self.cfg.wandb.save_checkpoints:
            self.console.print_notification("Sending checkpoint to W&B")
            checkpoint_artifact = wandb.Artifact(name="checkpoints", type="model")
            checkpoint_artifact.add_file(checkpoint_path)
            self.wandb.run.log_artifact(checkpoint_artifact)
            self.console.print_complete("Checkpoint sent to W&B")

    def evaluate(self) -> EvaluationResult:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        batch_counter = 0

        if self.cfg.eval_split_size is not None:
            self.console.create_progress_task("eval", "Evaluating", total=self.cfg.eval_split_size)
        else:
            self.console.create_progress_task("eval", "Evaluating", total=len(self.test_dataloader))

        with torch.no_grad():
            for batch in self.test_dataloader:
                batch_counter += 1
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, loss = self._forward(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )

                # Calculate accuracy
                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels.view(-1)

                # Create a mask tensor from the comparison
                comparison_result = flat_labels != -100
                valid_mask = torch.zeros_like(flat_labels, dtype=torch.bool)
                valid_mask = valid_mask | comparison_result  # Explicit conversion to boolean tensor

                # Count valid tokens
                valid_count = int(torch.sum(valid_mask).item())  # Convert to Python int for clarity

                if valid_count > 0:
                    # Get logits and use argmax to find predictions
                    masked_logits = torch.zeros(
                        (valid_count, flat_logits.size(-1)),
                        device=flat_logits.device,
                        dtype=flat_logits.dtype
                    )

                    # Fill in values for valid positions
                    valid_counter = 0
                    for i in range(len(flat_labels)):
                        if valid_mask[i]:
                            masked_logits[valid_counter] = flat_logits[i]
                            valid_counter += 1

                    # Now get predictions
                    predictions = masked_logits.argmax(dim=-1)

                    # Extract valid labels into a new tensor
                    valid_labels = torch.zeros(valid_count, device=flat_labels.device, dtype=flat_labels.dtype)
                    valid_counter = 0
                    for i in range(len(flat_labels)):
                        if valid_mask[i]:
                            valid_labels[valid_counter] = flat_labels[i]
                            valid_counter += 1

                    # Check which predictions are correct
                    correct = (predictions == valid_labels)

                    total_correct += int(torch.sum(correct).item())
                    total_tokens += valid_count

                total_loss += loss.item()

                self.console.update_progress_task("eval", advance=1)

        self.console.remove_progress_task("eval")

        # Calculate final metrics
        avg_loss = (total_loss / batch_counter) if batch_counter > 0 else 0
        accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss)

        # Return to training mode
        self.model.train()

        return EvaluationResult(eval_name="dataset", loss=avg_loss, accuracy=accuracy, perplexity=perplexity)

    def _loss_calc(self, logits, labels):
        # Flatten the tensors
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)

        # Calculate loss using ignore_index on shifted tensors
        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction='mean'
        )

        return loss

    def _backprop(self, grad_clip_value):
        if self.training_state.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)
            self.optimizer.step()

        if self.training_state.scheduler_mode == "tokens":
            self.scheduler.step(self.training_state.tokens_seen)
        else:
            self.scheduler.step()
        self.optimizer.zero_grad()

    def _bidirectional(self):
        logits, loss = self._forward(
            input_ids=self.input_ids,
            labels=self.labels,
            attention_mask=self.attention_mask
        )

        loss = loss / self.cfg.grad_steps

        if self.training_state.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    def _forward(self, input_ids, labels, attention_mask):
        f16_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if self.training_state.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=f16_dtype, enabled=True):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self._loss_calc(logits, labels)
        else:
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self._loss_calc(logits, labels)

        return logits, loss

    def _process_batch(self, batch):
        # Move batch to device
        self.input_ids = batch['input_ids'].to(self.device)
        self.attention_mask = batch['attention_mask'].to(self.device)
        self.labels = batch['labels'].to(self.device)

        actual_tokens_in_batch = self.attention_mask.sum().item()

        # Initialize logging dicts
        training_dict = None
        eval_dict = None

        # Forward pass with mixed precision
        loss = self._bidirectional()

        if loss.item() > 100:
            self.console.print(Colors.warning(f"Extremely high loss detected: {loss.item():.2f}. Check model initialization."))
            self.console.remove_progress_task("training")
            return False

        # Early in training, clip loss for stability
        if self.training_state.optimizer_steps < 10:
            loss = torch.clamp(loss, max=20.0)

        # 3. More aggressive gradient clipping for initial steps
        grad_clip_value = 0.5 if self.training_state.optimizer_steps < 10 else self.cfg.max_grad_norm

        # Calculate gradient norm
        pre_clip_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                pre_clip_grad_norm += param.grad.data.norm(2).item() ** 2
        pre_clip_grad_norm = pre_clip_grad_norm ** 0.5

        self.training_state.update_precision_mode(self.start_time, pre_clip_grad_norm, self.scheduler.get_warmup_ratio())

        self.training_state.finish_batch(self.cfg.batch_size)

        # Update weights after accumulating gradients
        if self.training_state.inference_steps % self.cfg.grad_steps == 0:
            self.training_state.step_optimizer()
            self.training_state.increment_steps_since_instrument()
            self.training_state.increment_steps_since_eval()

            self.training_state.update_grad_norm(pre_clip_grad_norm)

            self._backprop(grad_clip_value=grad_clip_value)
            if self.training_state.transition_phase():
                precision_mode = self.training_state.get_current_precision_mode()

                self.console.rule(f"Phase Transition: {Colors.highlight(self.training_state.run_phase.value.title())} ({Colors.highlight(precision_mode)})")

                self.console.update_progress_task(
                    "training",
                    description=f"Phase: {self.training_state.run_phase.value.title()} ({precision_mode})"
                )

                if self.cfg.allow_amp_switchover and self.training_state.use_amp == False:
                    self.training_state.use_amp = True
                    if precision_mode == "FP16":
                        self.scaler = torch.amp.GradScaler()
                    else:
                        self.scaler = None

        self.training_state.update_metrics(actual_tokens_in_batch, loss.item() * self.cfg.grad_steps, self.cfg.batch_size)

        self.console.update_app_stats({
            "Tokens/s": f"{self.training_state.tokens_per_sec:08,.2f}",
            "Loss": f"{self.training_state.current_loss:.4f}",
            "Perplexity": f"{self.training_state.current_perplexity:,.2f}",
            "Grad Norm": f"{sum(self.training_state.grad_norm_history[-self.cfg.grad_steps:]) / self.cfg.grad_steps:.4f}",
            "Learning Rate": f"{self.scheduler.get_last_lr()[0]:.4e}",
            "W&B Logs": f"{self.training_state.total_instrument_events:,}",
            "Evals": f"{self.training_state.total_eval_events:,}",
            "Tokens/Batch": f"{self.training_state.tokens_per_batch:.2f}",
        })

        if self.training_state.should_instrument():
            self.training_state.reset_steps_since_instrument()
            self.training_state.total_instrument_events += 1
            training_dict = self.training_state.get_instrument_dict(self.scheduler.get_last_lr()[0])

            if self.training_state.should_eval():
                # Run evaluation
                self.training_state.total_eval_events += 1
                eval_results = self.evaluate()
                self.training_state.reset_steps_since_eval()
                test_loss = eval_results.loss
                test_accuracy = eval_results.accuracy
                test_perplexity = eval_results.perplexity

                self.training_state.eval_history.append(eval_results)

                eval_dict = {
                    "eval/test_loss": test_loss,
                    "eval/test_perplexity": test_perplexity,
                    "eval/test_accuracy": test_accuracy,
                }

                self.console.update_app_stats({
                    "Last Eval Loss": f"{test_loss:.4f}",
                    "Last Eval Perplexity": f"{test_perplexity:,.2f}",
                    "Last Eval Accuracy": f"{test_accuracy*100:.2f}%",
                })

                self.model.train()

        log_dict = {}
        if training_dict is not None:
            log_dict.update(training_dict)
        if eval_dict is not None:
            log_dict.update(eval_dict)

        if len(log_dict) > 0:
            self.wandb.log(log_dict)

        if (time.time() - self.last_checkpoint_time) >= self.training_state.checkpoint_interval:
            self.save_checkpoint()
            self.last_checkpoint_time = time.time()

        return True