from .common_imports import *


class PosteriorCollapseStopper(Callback):
    def __init__(
        self,
        monitor: str = "kl_loss/val",
        min_threshold: float = 0.1,
        patience: int = 3,
    ):
        """
        Stops training if the monitored metric falls below min_threshold for `patience` consecutive checks.
        Useful for detecting Posterior Collapse in VAEs (KL ~ 0).
        """
        super().__init__()
        self.monitor = monitor
        self.min_threshold = min_threshold
        self.patience = patience
        self.wait_count = 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if self.monitor in trainer.callback_metrics:
            current_value = trainer.callback_metrics[self.monitor]
            if current_value < self.min_threshold:
                self.wait_count += 1
                if trainer.is_global_zero:
                    print(
                        cl.Fore.yellow
                        + f"\n[PosteriorCollapseStopper] Warning: {self.monitor} ({current_value:.6f}) is below threshold ({self.min_threshold}). "
                        + f"Patience: {self.wait_count}/{self.patience}"
                        + cl.Style.reset
                    )

                if self.wait_count >= self.patience:
                    if trainer.is_global_zero:
                        print(
                            cl.Fore.red
                            + f"\n[PosteriorCollapseStopper] Stopping training! {self.monitor} stayed below threshold for {self.patience} epochs. Posterior collapse detected."
                            + cl.Style.reset
                        )
                    trainer.should_stop = True
            else:
                if self.wait_count > 0 and trainer.is_global_zero:
                    print(
                        cl.Fore.green
                        + f"\n[PosteriorCollapseStopper] {self.monitor} recovered ({current_value:.6f}). Resetting patience."
                        + cl.Style.reset
                    )
                self.wait_count = 0
