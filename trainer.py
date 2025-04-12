import math
import datetime

import torch
import torch.nn.functional as F
import os
import time
from tqdm import tqdm


# Console colors for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def header(text):
        return f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}"

    @staticmethod
    def info(text):
        return f"{Colors.BLUE}{text}{Colors.ENDC}"

    @staticmethod
    def success(text):
        return f"{Colors.GREEN}{text}{Colors.ENDC}"

    @staticmethod
    def warning(text):
        return f"{Colors.YELLOW}{text}{Colors.ENDC}"

    @staticmethod
    def error(text):
        return f"{Colors.RED}{text}{Colors.ENDC}"

    @staticmethod
    def highlight(text):
        return f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}"

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
            eval_interval=1000,
            global_steps=None,
            wandb=None,
            allow_mp_switch=False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.total_parameters = total_parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.global_steps = global_steps or len(train_dataloader)
        self.target_tokens = total_parameters * 10
        self.wandb = wandb
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        self.input_ids = None
        self.attention_mask = None
        self.labels = None
        self.optimizer_steps = 0
        self.inference_steps = 0
        self.steps_since_instrument = 0
        self.checkpoint_interval = 60 * 60 * 4
        self.total_tokens = 0
        self.target_tokens = self.total_parameters * 10
        self.token_window = []
        self.token_times = []
        self.loss_window = []
        self.pre_clip_grad_norm = 0
        self.current_loss = float('inf')
        self.current_perplexity = float('inf')
        self.tokens_per_sec = 0
        self.run_status = "training"
        self.run_phase = "warmup"
        self.use_cache = False
        self.stability_steps = 0
        self.stability_reached = False
        self.allow_mp_switch = allow_mp_switch

    def run(self):
        self.model.train()

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Tracking training dynamics
        start_time = time.time()
        last_checkpoint_time = start_time
        progress_bar = tqdm(total=self.target_tokens, desc=f"Pretrain [{self.run_phase.title()}] ({self._precision_mode()})", unit="tokens", colour="#4B6BFF")
        eval_interval_steps = max(1, self.eval_interval)

        while self.total_tokens < self.target_tokens:

            for batch in self.train_dataloader:
                # Move batch to device
                self.input_ids = batch['input_ids'].to(self.device)
                self.attention_mask = batch['attention_mask'].to(self.device)
                self.labels = batch['labels'].to(self.device)

                # Initialize logging dicts
                training_dict = {}
                eval_dict = {}

                # Increment
                self.inference_steps += 1

                # Forward pass with mixed precision
                loss = self._bidirectional()

                if loss.item() > 100:
                    print(Colors.warning(
                        f"\n Extremely high loss detected: {loss.item():.2f}. Check model initialization."))
                    progress_bar.close()
                    return self.total_tokens, "failed"

                # Early in training, clip loss for stability
                if self.optimizer_steps < 10:
                    loss = torch.clamp(loss, max=20.0)

                # 3. More aggressive gradient clipping for initial steps
                grad_clip_value = 0.5 if self.optimizer_steps < 10 else self.max_grad_norm

                # Calculate gradient norm
                self.pre_clip_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        self.pre_clip_grad_norm += param.grad.data.norm(2).item() ** 2
                self.pre_clip_grad_norm = self.pre_clip_grad_norm ** 0.5

                self._precision_phase()

                # Update weights after accumulating gradients
                if self.inference_steps % self.gradient_accumulation_steps == 0:
                    self.optimizer_steps += 1
                    self.steps_since_instrument += 1

                    self._backprop(grad_clip_value=grad_clip_value)
                    if self.run_phase == "warmup" and self.stability_reached == True:
                        self.run_phase = "core learning"
                        mode = self._precision_mode() if self.allow_mp_switch else "FP32"
                        progress_bar.desc = f"Pretrain [{self.run_phase.title()}] ({mode})"

                        if self.use_amp == False and self.allow_mp_switch:
                            self.use_amp = True
                            if self._precision_mode() == "FP16":
                                self.scaler = torch.amp.GradScaler()
                            else:
                                self.scaler = None
                    # Clear CUDA cache periodically to prevent fragmentation
                    # if inference_steps // gradient_accumulation_steps % empty_cache_interval == 0:
                    #    torch.cuda.empty_cache()

                # Update tracking metrics
                self.total_tokens += self.input_ids.numel()

                progress_bar.update(self.input_ids.numel())

                self._update_metrics(loss)

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{self.current_loss:.4f}",
                    "ppl": f"{self.current_perplexity:.2f}",
                    "grad_norm": f"{self.pre_clip_grad_norm:.4f}",
                })

                if (self.steps_since_instrument == self.log_interval or self.optimizer_steps % eval_interval_steps == 0) and self.steps_since_instrument > 0 and self.optimizer_steps > 0:
                    self.steps_since_instrument = 0
                    training_dict = {
                        "training/loss": self.current_loss,
                        "training/perplexity": self.current_perplexity,
                        "training/learning_rate": self.scheduler.get_last_lr()[0],
                        "training/tokens_per_second": self.tokens_per_sec,
                        "training/grad_norm": self.pre_clip_grad_norm,
                        "metric/progress": self.optimizer_steps / self.global_steps
                    }

                if self.optimizer_steps % eval_interval_steps == 0 and self.optimizer_steps > 0:
                    print(Colors.header(f"\n{'-' * 40}"))
                    print(Colors.header(f" Evaluating model performance on test dataset"))
                    print(Colors.header(f"{'-' * 40}"))

                    # Run evaluation
                    eval_results = self.evaluate()
                    test_loss = eval_results['loss']
                    test_accuracy = eval_results['accuracy']
                    test_perplexity = eval_results['perplexity']

                    # Print intermediate results
                    print(Colors.info(f"  • Eval loss: {Colors.highlight(f'{test_loss:.4f}')}"))
                    print(Colors.info(f"  • Eval perplexity: {Colors.highlight(f'{test_perplexity:.2f}')}"))
                    print(Colors.info(f"  • Eval accuracy: {Colors.highlight(f'{test_accuracy:.2%}')}"))

                    eval_dict = {
                        "eval/test_loss": test_loss,
                        "eval/test_perplexity": test_perplexity,
                        "eval/test_accuracy": test_accuracy,
                        "metric/progress": self.optimizer_steps / self.global_steps,
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

                if self.run_phase == "core learning" and self.total_tokens >= self.target_tokens*0.8:
                    self.run_phase = "refinement"
                    progress_bar.desc = f"Pretrain [{self.run_phase.title()}] ({self._precision_mode()})"

                if (time.time() - last_checkpoint_time) >= self.checkpoint_interval:
                    self.save_checkpoint()
                    last_checkpoint_time = time.time()

            if self.total_tokens >= self.target_tokens:
                progress_bar.close()
                break

        return self.total_tokens, "success"

    def save_checkpoint(self):
        print(Colors.header(f"\n{'-' * 40}"))
        print(Colors.header(f" Saving Checkpoint"))
        print(Colors.header(f"{'-' * 40}"))

        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, checkpoint_path)
        print(Colors.success(f"✓ Saved checkpoint: {checkpoint_path}"))

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.test_dataloader:
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

        # Calculate final metrics
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss)

        # Return to training mode
        self.model.train()

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity
        }

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

        loss = loss / self.gradient_accumulation_steps
        return loss

    def _backprop(self, grad_clip_value):
        if self.use_amp and self.scaler is not None:
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

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    def _forward(self, input_ids, labels, attention_mask):
        f16_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if self.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=f16_dtype, enabled=True):
                logits = self.model(input_ids, attention_mask=attention_mask, use_cache=self.use_cache)
                loss = self._loss_calc(logits, labels)
        else:
            logits = self.model(input_ids, attention_mask=attention_mask, use_cache=self.use_cache)
            loss = self._loss_calc(logits, labels)

        return logits, loss

    def _precision_phase(self):
        if self.stability_reached:
            return None
        if self.pre_clip_grad_norm < 1.0:
            self.stability_steps += 1
        else:
            self.stability_steps = 0

        if self.stability_steps >= self.gradient_accumulation_steps and self.stability_reached == False:
            f16_dtype = "BF16" if torch.cuda.is_bf16_supported() else "FP16"
            self.stability_reached = True
            print(Colors.success(f"\n  🎉 Stability Reached! Switching to {f16_dtype}. 🎉"))

    def _update_metrics(self, loss):
        # Update moving window metrics
        self.token_window.append(self.input_ids.numel())
        self.token_times.append(time.time())
        self.loss_window.append(loss.item() * self.gradient_accumulation_steps)
        if len(self.token_window) > 10:
            self.token_window = self.token_window[-10:]
        if len(self.token_times) > 10:
            self.token_times = self.token_times[-10:]
        if len(self.loss_window) > 15:
            self.loss_window = self.loss_window[-15:]

        # Calculate tokens per second
        self.tokens_per_sec = sum(self.token_window) / (self.token_times[-1] - self.token_times[0]) if len(
            self.token_times) > 1 else 1

        # Calculate current perplexity (exp of loss)
        self.current_loss = sum(self.loss_window) / len(self.loss_window)
        self.current_perplexity = math.exp(min(self.current_loss, 20)) if len(self.loss_window) > 1 else float('inf')

    def _precision_mode(self):
        if self.use_amp:
            return "BF16" if torch.cuda.is_bf16_supported() else "FP16"
        else:
            return "FP32"