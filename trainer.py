import math
import datetime
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import os
import time

from rich.table import Column
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, \
    TimeRemainingColumn


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

    # Training progress
    batches_seen: int
    tokens_seen: int
    optimizer_steps: int
    inference_steps: int
    steps_since_instrument: int
    steps_since_eval: int

    # Mode & phase
    run_phase: str
    run_status: str
    stability_steps: int
    stability_reached: bool
    checkpoint_interval: int
    use_amp: bool

    def update_metrics(self, actual_tokens_in_batch, loss):
        # Update moving window metrics
        self.token_history.append(actual_tokens_in_batch)
        self.token_times.append(time.time())
        self.loss_history.append(loss)
        if len(self.token_history) > 10:
            self.token_history = self.token_history[-10:]
        if len(self.token_times) > 10:
            self.token_times = self.token_times[-10:]
        if len(self.loss_history) > 15:
            self.loss_history = self.loss_history[-15:]

class Trainer:
    def __init__(
            self,
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            device,
            total_parameters,
            gradient_accumulation_steps=8,
            max_grad_norm=1.0,
            checkpoint_dir="checkpoints",
            model_dir="models",
            use_amp=False,
            log_interval=10,
            eval_interval=100,
            global_steps=None,
            wandb=None,
            allow_amp_switch=False,
            console: Console|None = None,
            dataset_size=1,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
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
            use_amp=use_amp,
        )
        self.total_parameters = total_parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.global_steps = global_steps or len(train_dataloader)
        self.target_tokens = total_parameters * 10
        self.wandb = wandb
        self.scaler = torch.amp.GradScaler() if self.training_state.use_amp else None
        self.input_ids = None
        self.attention_mask = None
        self.labels = None
        self.total_tokens = 0
        self.pre_clip_grad_norm = 0
        self.current_loss = float('inf')
        self.current_perplexity = float('inf')
        self.tokens_per_sec = 0
        self.use_cache = False
        self.allow_amp_switch = allow_amp_switch
        self.console = console or Console()
        self.dataset_size = dataset_size

    def run(self):
        self.model.train()

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Tracking training dynamics
        start_time = time.time()
        last_checkpoint_time = start_time
        spinner_col = SpinnerColumn(table_column=Column(max_width=3))
        desc_col = TextColumn(text_format="{task.description}", style=Colors.CYAN, table_column=Column(max_width=30))
        bar_col = BarColumn(bar_width=None, complete_style="#4B6BFF")
        pct_col = TaskProgressColumn(table_column=Column(max_width=10))
        time_elapsed_col = TimeElapsedColumn(table_column=Column(max_width=15))
        time_remaining_col = TimeRemainingColumn(table_column=Column(max_width=15))
        metadata_col = TextColumn(text_format="{task.fields[field1_name]}: {task.fields[field1]} {task.fields[field2_name]}: {task.fields[field2]} {task.fields[field3_name]}: {task.fields[field3]} {task.fields[field4_name]}: {task.fields[field4]}", style=Colors.BLUE, table_column=Column(min_width=40))
        progress_bar = Progress(
            spinner_col, desc_col, bar_col, pct_col, time_elapsed_col, time_remaining_col, metadata_col,
            console=self.console, transient=True, expand=True
        )
        progress_bar.start()
        training_task = progress_bar.add_task(
            description=f"Phase: {self.training_state.run_phase.title()} ({self._precision_mode()})",
            total=self.target_tokens,
            field1_name="Loss",
            field1=float('inf'),
            field2_name="PPL",
            field2=float('inf'),
            field3_name="Grad",
            field3=float('inf'),
            field4_name="TPS",
            field4=0.0,
        )
        eval_interval_steps = max(1, self.eval_interval)
        epoch = 0

        try:
            while self.total_tokens < self.target_tokens:
                epoch_task = progress_bar.add_task(
                    description=f"Epoch {epoch + 1}",
                    total=len(self.train_dataloader),
                    field1_name="Batches",
                    field1=f"{len(self.train_dataloader):,}",
                    field2_name="Tok/Batch",
                    field2=f"{self.dataset_size/len(self.train_dataloader):.2f}",
                    field3_name="Avg",
                    field3=0,
                    field4_name=":",
                    field4="",
                )
                tokens_this_epoch = 0
                batches_this_epoch = 0
                for batch in self.train_dataloader:
                    # Move batch to device
                    self.input_ids = batch['input_ids'].to(self.device)
                    self.attention_mask = batch['attention_mask'].to(self.device)
                    self.labels = batch['labels'].to(self.device)

                    actual_tokens_in_batch = self.attention_mask.sum().item()

                    # Initialize logging dicts
                    training_dict = {}
                    eval_dict = {}

                    # Increment
                    batches_this_epoch += 1
                    self.training_state.inference_steps += 1

                    # Forward pass with mixed precision
                    loss = self._bidirectional()

                    if loss.item() > 100:
                        self.console.print(Colors.warning(f"Extremely high loss detected: {loss.item():.2f}. Check model initialization."))
                        progress_bar.stop()
                        return self.total_tokens, "failed"

                    # Early in training, clip loss for stability
                    if self.training_state.optimizer_steps < 10:
                        loss = torch.clamp(loss, max=20.0)

                    # 3. More aggressive gradient clipping for initial steps
                    grad_clip_value = 0.5 if self.training_state.optimizer_steps < 10 else self.max_grad_norm

                    # Calculate gradient norm
                    self.pre_clip_grad_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            self.pre_clip_grad_norm += param.grad.data.norm(2).item() ** 2
                    self.pre_clip_grad_norm = self.pre_clip_grad_norm ** 0.5

                    self._precision_phase(start_time)

                    # Update weights after accumulating gradients
                    if self.training_state.inference_steps % self.gradient_accumulation_steps == 0:
                        self.training_state.optimizer_steps += 1
                        self.training_state.steps_since_instrument += 1
                        self.training_state.steps_since_eval += 1

                        self.training_state.grad_norm_history.append(self.pre_clip_grad_norm)

                        self._backprop(grad_clip_value=grad_clip_value)
                        if self.training_state.run_phase == "warmup" and self.training_state.stability_reached == True:
                            self.training_state.run_phase = "core learning"
                            progress_bar.update(training_task, description=f"Phase: {self.training_state.run_phase.title()} ({self._precision_mode()})")

                            if self.allow_amp_switch and self.training_state.use_amp == False:
                                self.training_state.use_amp = True
                                if self._precision_mode() == "FP16":
                                    self.scaler = torch.amp.GradScaler()
                                else:
                                    self.scaler = None

                    # Update tracking metrics
                    self.total_tokens += actual_tokens_in_batch
                    tokens_this_epoch += actual_tokens_in_batch

                    self._update_metrics(loss, actual_tokens_in_batch)

                    progress_bar.update(
                        epoch_task,
                        advance=1,
                        field3=f"{tokens_this_epoch/batches_this_epoch:.2f}",
                    )
                    progress_bar.update(
                        training_task,
                        advance=actual_tokens_in_batch,
                        field1=f"{self.current_loss:.4f}",
                        field2=f"{self.current_perplexity:.2f}",
                        field3=f"{self.pre_clip_grad_norm:.4f}",
                        field4=f"{self.tokens_per_sec:.1f}",
                    )

                    if (self.training_state.steps_since_instrument == self.log_interval or self.training_state.optimizer_steps % eval_interval_steps == 0) and self.training_state.steps_since_instrument > 0 and self.training_state.optimizer_steps > 0:
                        self.training_state.steps_since_instrument = 0
                        grad_norm = sum(self.training_state.grad_norm_history[-self.log_interval:])/self.log_interval
                        training_dict = {
                            "training/loss": self.current_loss,
                            "training/perplexity": self.current_perplexity,
                            "training/learning_rate": self.scheduler.get_last_lr()[0],
                            "training/tokens_per_second": self.tokens_per_sec,
                            "training/grad_norm": grad_norm,
                            "metric/progress": self.training_state.optimizer_steps / self.global_steps
                        }

                        if self.training_state.steps_since_eval > 0 and self.training_state.optimizer_steps % eval_interval_steps == 0 and self.training_state.optimizer_steps > 0 and self.training_state.run_phase != "warmup":
                            # Run evaluation
                            eval_results = self.evaluate(progress_bar=progress_bar)
                            self.training_state.steps_since_eval = 0
                            test_loss = eval_results.loss
                            test_accuracy = eval_results.accuracy
                            test_perplexity = eval_results.perplexity

                            self.training_state.eval_history.append(eval_results)

                            eval_dict = {
                                "eval/test_loss": test_loss,
                                "eval/test_perplexity": test_perplexity,
                                "eval/test_accuracy": test_accuracy,
                            }
                            self.model.train()

                    log_dict = {}
                    if training_dict is not None:
                        log_dict.update(training_dict)
                    if eval_dict is not None:
                        log_dict.update(eval_dict)

                    if len(log_dict) > 0:
                        self.wandb.log(log_dict)

                    if self.total_tokens >= self.target_tokens:
                        break

                    if self.training_state.run_phase == "core learning" and self.total_tokens >= self.target_tokens*0.8:
                        self.training_state.run_phase = "refinement"
                        progress_bar.update(training_task, description=f"Phase: {self.training_state.run_phase.title()} ({self._precision_mode()})")

                    if (time.time() - last_checkpoint_time) >= self.training_state.checkpoint_interval:
                        self.save_checkpoint()
                        last_checkpoint_time = time.time()

                epoch += 1
                self.console.print(Colors.success(f"Epoch {epoch} completed."))
                self.console.print(Colors.info(f"  • Observed tokens: {Colors.highlight(f'{tokens_this_epoch:,}')}"))
                self.console.print(Colors.info(f"  • Observed average tokens per batch: {Colors.highlight(f'{tokens_this_epoch/batches_this_epoch:.2f}')}"))
                progress_bar.stop_task(epoch_task)
                progress_bar.remove_task(epoch_task)
                if self.total_tokens >= self.target_tokens:
                    break

            progress_bar.stop()
            return self.total_tokens, "success"
        except KeyboardInterrupt:
            progress_bar.stop_task(training_task)
            progress_bar.remove_task(training_task)
            progress_bar.stop()
            raise KeyboardInterrupt
        except Exception as e:
            progress_bar.stop_task(training_task)
            progress_bar.remove_task(training_task)
            progress_bar.stop()
            raise e

    def save_checkpoint(self):
        self.console.rule(Colors.highlight(f"Saving Checkpoint"), style=Colors.HEADER)

        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, checkpoint_path)
        self.console.print(Colors.success(f"✓ Saved checkpoint: {checkpoint_path}"))

    def evaluate(self, progress_bar: Progress|None=None) -> EvaluationResult:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        if progress_bar is None:
            progress_bar = Progress(console=self.console, transient=True, expand=True)
            progress_bar.start()
        eval_task = progress_bar.add_task(
            description=f"Eval [Dataset]", total=len(self.test_dataloader),
            field1_name="", field1="", field2_name="", field2="", field3_name="", field3="", field4_name="", field4="",
        )

        with torch.no_grad():
            for batch in self.test_dataloader:
                progress_bar.advance(eval_task)
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

        progress_bar.remove_task(eval_task)

        # Calculate final metrics
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss)
        # TODO: Implement normalized accuracy

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
            ignore_index=-100,  # Use built-in ignore index
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

        self.scheduler.step()
        self.optimizer.zero_grad()

    def _bidirectional(self):
        logits, loss = self._forward(
            input_ids=self.input_ids,
            labels=self.labels,
            attention_mask=self.attention_mask
        )

        loss = loss / self.gradient_accumulation_steps

        if self.training_state.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    def _forward(self, input_ids, labels, attention_mask):
        f16_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if self.training_state.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=f16_dtype, enabled=True):
                logits = self.model(input_ids, attention_mask=attention_mask, use_cache=self.use_cache)
                loss = self._loss_calc(logits, labels)
        else:
            logits = self.model(input_ids, attention_mask=attention_mask, use_cache=self.use_cache)
            loss = self._loss_calc(logits, labels)

        return logits, loss

    def _precision_phase(self, start_time):
        if self.training_state.stability_reached:
            return None
        if self.pre_clip_grad_norm < 1.0:
            self.training_state.stability_steps += 1
        else:
            self.training_state.stability_steps = 0

        if self.training_state.stability_steps >= self.gradient_accumulation_steps and self.training_state.stability_reached == False:
            self.training_state.stability_reached = True
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

            self.console.print(Colors.success(f"🎉 Stability Reached in {self.total_tokens:,} tokens ({str_time})! Continuing in {self._precision_mode()} mode. 🎉"))

    def _update_metrics(self, loss, actual_tokens_in_batch):
        self.training_state.update_metrics(
            actual_tokens_in_batch=actual_tokens_in_batch,
            loss=loss.item() * self.gradient_accumulation_steps
        )

        # Calculate tokens per second
        self.tokens_per_sec = sum(self.training_state.token_history) / (self.training_state.token_times[-1] - self.training_state.token_times[0]) if len(self.training_state.token_times) > 1 else 1

        # Calculate current perplexity (exp of loss)
        self.current_loss = sum(self.training_state.loss_history) / len(self.training_state.loss_history)
        self.current_perplexity = math.exp(min(self.current_loss, 20)) if len(self.training_state.loss_history) > 1 else float('inf')

    def _precision_mode(self):
        if self.training_state.use_amp or (self.training_state.use_amp == False and self.allow_amp_switch == True and self.training_state.stability_reached == True):
            return "BF16" if torch.cuda.is_bf16_supported() else "FP16"
        else:
            return "FP32"