import math
import torch
import time

from dataclasses import dataclass
from enum import Enum
from typing import Literal
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

    def train_info_panel_content(self):
        gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_mem_util = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        gpu_util = torch.cuda.utilization()
        if gpu_mem_util >= 1 or gpu_mem_util <= 0.5:
            gpu_mem_text = Colors.error(f"{gpu_mem_used:.2f}GB/{gpu_mem_total:.2f}GB")
            gpu_mem_util_text = Colors.error(f"{gpu_mem_util*100:.2f}%")
        elif gpu_mem_util >= 0.95 or gpu_mem_util <= 0.75:
            gpu_mem_text = Colors.warning(f"{gpu_mem_used:.2f}GB/{gpu_mem_total:.2f}GB")
            gpu_mem_util_text = Colors.warning(f"{gpu_mem_util*100:.2f}%")
        else:
            gpu_mem_text = Colors.success(f"{gpu_mem_used:.2f}GB/{gpu_mem_total:.2f}GB")
            gpu_mem_util_text = Colors.success(f"{gpu_mem_util*100:.2f}%")
        if gpu_util <= 50:
            gpu_util_text = Colors.error(f"{gpu_util}%")
        elif gpu_util <= 90:
            gpu_util_text = Colors.warning(f"{gpu_util}%")
        else:
            gpu_util_text = Colors.success(f"{gpu_util}%")
        return [
            {"title": "World Size", "content": f"{torch.cuda.device_count()} GPUs"},
            {"title": "GPU Usage", "content": f"{gpu_util_text}"},
            {"title": "GPU Memory", "content": f"{gpu_mem_text}"},
            {"title": "GPT Memory Usage", "content": f"{gpu_mem_util_text}"},
            {"title": "Run Phase", "content": self.run_phase.value.title() if self.run_phase else "N/A"},
            {"title": "Current Precision Mode", "content": self.get_current_precision_mode()},
            {"title": "Tokens Seen", "content": f"{self.tokens_seen:,}"},
            {"title": "Target Tokens", "content": f"{self.target_tokens:,}"},
            {"title": "Optimizer Steps", "content": f"{self.optimizer_steps}"},
            {"title": "Inference Steps", "content": f"{self.inference_steps}"},
            {"title": "Stability Reached", "content": f"{Colors.success('Yes') if self.stability_reached else Colors.error('No')}"},
        ]

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
