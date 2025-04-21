import math
import datetime
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import os
import time

from hydra.utils import to_absolute_path

from config_schema import TrainingConfig
from console import TPConsole


# Console colors for better readability
class Colors:
    HEADER = '#B87FD9'
    BLUE = '#61AFEF'
    CYAN = '#56B6C2'
    GREEN = '#98C379'
    YELLOW = '#E5C07B'
    RED = '#E06C75'
    BOLD = 'bold'
    UNDERLINE = 'underline'

    @staticmethod
    def header(text):
        return f"[{Colors.HEADER} {Colors.BOLD}]{text}[/{Colors.HEADER} {Colors.BOLD}]"

    @staticmethod
    def info(text):
        return f"[{Colors.BLUE}]{text}[/{Colors.BLUE}]"

    @staticmethod
    def success(text):
        return f"[{Colors.GREEN}]{text}[/{Colors.GREEN}]"

    @staticmethod
    def warning(text):
        return f"[{Colors.YELLOW}]{text}[/{Colors.YELLOW}]"

    @staticmethod
    def error(text):
        return f"[{Colors.RED}]{text}[/{Colors.RED}]"

    @staticmethod
    def highlight(text):
        return f"[{Colors.CYAN} {Colors.BOLD}]{text}[/{Colors.CYAN} {Colors.BOLD}]"

@dataclass
class EvaluationResult:
    eval_name: str
    loss: float
    accuracy: float
    perplexity: float

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
    run_phase: str
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
            return "BF16" if torch.cuda.is_bf16_supported() else "FP16"
        else:
            return "FP32"

    def update_precision_mode(self, start_time, current_grad_norm):
        if self.stability_reached:
            return None
        if current_grad_norm < 1.0:
            self.update_stability(start_time)
        else:
            self.stability_steps = 0

    def update_stability(self, start_time):
        self.stability_steps += 1
        self.stability_reached = self.stability_steps >= self.steps_per_gradient_update

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

            TPConsole().print(Colors.success(
                f"🎉 Stability Reached in {self.tokens_seen:,} tokens ({str_time})! Continuing in {self.get_current_precision_mode()} mode. 🎉"
            ))

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
            wandb=None,
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
            run_phase="warmup",
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
        self.wandb = wandb
        self.scaler = torch.amp.GradScaler() if self.training_state.use_amp and not torch.cuda.is_bf16_supported() else None
        self.input_ids = None
        self.attention_mask = None
        self.labels = None
        self.last_checkpoint_time = None
        self.start_time = None
        self.console = TPConsole()

    def run_tokens(self, target_tokens):
        self.model.train()
        self.training_state.scheduler_mode = "tokens"
        self.training_state.target_tokens = target_tokens

        # Create checkpoint directory
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        self.console.create_progress_task(
            "training",
            f"Phase: {self.training_state.run_phase.title()} ({self.training_state.get_current_precision_mode()})",
            total=self.training_state.target_tokens
        )

        # Tracking training dynamics
        self.start_time = time.time()
        self.training_state.time_of_last_instrument = self.start_time
        self.last_checkpoint_time = self.start_time

        while self.training_state.tokens_seen < self.training_state.target_tokens:
            for batch in self.train_dataloader:
                batch_result = self._process_batch(
                    batch=batch,
                )

                if not batch_result:
                    return self.training_state.tokens_seen, "failed"

                if self.training_state.tokens_seen >= self.training_state.target_tokens:
                    break

                if self.training_state.run_phase == "core learning" and self.training_state.tokens_seen >= self.training_state.target_tokens * 0.8:
                    self.training_state.run_phase = "refinement"
                    self.console.update_progress_task(
                        "training",
                        description=f"Phase: {self.training_state.run_phase.title()} ({self.training_state.get_current_precision_mode()})"
                    )

            self.training_state.finish_epoch()

            if self.training_state.tokens_seen >= self.training_state.target_tokens:
                break

        self.console.remove_progress_task("training")
        return self.training_state.tokens_seen, "success"

    def run_epochs(self, target_epochs):
        self.model.train()
        self.training_state.scheduler_mode = "epochs"
        self.training_state.target_epochs = target_epochs

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        self.console.create_progress_task(
            "training",
            f"Phase: {self.training_state.run_phase.title()} ({self.training_state.get_current_precision_mode()})",
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

        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, checkpoint_path)
        self.console.print(Colors.success(f"✓ Saved checkpoint: {checkpoint_path}"))

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

        self.training_state.update_precision_mode(self.start_time, pre_clip_grad_norm)

        self.training_state.finish_batch(self.cfg.batch_size)

        # Update weights after accumulating gradients
        if self.training_state.inference_steps % self.cfg.grad_steps == 0:
            self.training_state.step_optimizer()
            self.training_state.increment_steps_since_instrument()
            self.training_state.increment_steps_since_eval()

            self.training_state.update_grad_norm(pre_clip_grad_norm)

            self._backprop(grad_clip_value=grad_clip_value)
            if self.training_state.run_phase == "warmup" and self.training_state.stability_reached == True:
                self.training_state.update_run_phase("core learning")
                precision_mode = self.training_state.get_current_precision_mode()
                self.console.update_progress_task(
                    "training",
                    description=f"Phase: {self.training_state.run_phase.title()} ({precision_mode})"
                )

                if self.cfg.allow_amp_switchover and self.training_state.use_amp == False:
                    self.training_state.use_amp = True
                    if precision_mode == "FP16":
                        self.scaler = torch.amp.GradScaler()
                    else:
                        self.scaler = None

        self.training_state.update_metrics(actual_tokens_in_batch, loss.item() * self.cfg.grad_steps, self.cfg.batch_size)

        self.console.update_app_stats({
            "Tokens/s": f"{self.training_state.tokens_per_sec:.2f}",
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
                    "Last Eval Perplexity": f"{test_perplexity:,.4f}",
                    "Last Eval Accuracy": f"{test_accuracy:.4f}",
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