"""Learning-rate schedulers: Warmup + Reduce-Once.

Provides `WarmupReduceOnceLR` which integrates with PyTorch optimizers.
"""

from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupReduceOnceLR(_LRScheduler):
    """
    LR scheduler with linear warmup for `warmup_steps` steps followed by a
    single (or permanent) multiplicative reduction of the learning rate.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps to linearly warm up (0 = no warmup).
        reduce_factor (float): Multiplicative factor to reduce LR (0 < factor <= 1).
        reduce_at_step (Optional[int]): Step index at which to apply reduction.
            Defaults to `warmup_steps`.
        permanent (bool): If True, reduction is permanent from `reduce_at_step` onward.
            If False, reduction applies only for a single step.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 0,
        reduce_factor: float = 0.1,
        reduce_at_step: Optional[int] = None,
        permanent: bool = True,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if reduce_factor <= 0.0 or reduce_factor > 1.0:
            raise ValueError("reduce_factor should be in (0, 1]")

        self.warmup_steps = int(warmup_steps)
        self.reduce_factor = float(reduce_factor)
        self.reduce_at_step = (
            int(reduce_at_step) if reduce_at_step is not None else self.warmup_steps
        )
        self.permanent = bool(permanent)

        # Internal state to track whether reduction has been applied
        self._applied_perm = False
        self._applied_once_step = None

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # Linear warmup multiplier: (step+1)/warmup_steps for step in [0, warmup_steps-1]
        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_scale = float(step + 1) / float(max(1, self.warmup_steps))
        else:
            warmup_scale = 1.0

        # Decide whether to apply the reduction this step
        apply_reduce = False

        if self._applied_perm:
            apply_reduce = True
        elif self._applied_once_step is not None:
            if step == self._applied_once_step:
                apply_reduce = True
            else:
                # once-step has passed -> clear
                self._applied_once_step = None
                apply_reduce = False
        elif step >= self.reduce_at_step:
            if self.permanent:
                self._applied_perm = True
                apply_reduce = True
            else:
                self._applied_once_step = step
                apply_reduce = True

        factor = self.reduce_factor if apply_reduce else 1.0

        return [base_lr * warmup_scale * factor for base_lr in self.base_lrs]

    def state_dict(self):
        base = super().state_dict()
        base["_warmup_reduce_once_state"] = {
            "applied_perm": self._applied_perm,
            "applied_once_step": self._applied_once_step,
        }
        return base

    def load_state_dict(self, state_dict):
        extra = state_dict.pop("_warmup_reduce_once_state", None)
        super().load_state_dict(state_dict)
        if extra is not None:
            self._applied_perm = extra.get("applied_perm", False)
            self._applied_once_step = extra.get("applied_once_step", None)


__all__ = ["WarmupReduceOnceLR"]
