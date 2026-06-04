from .common_imports import *


class RobustCheckpointCallback(ModelCheckpoint):
    def on_fit_end(self, trainer, pl_module):
        """
        Called when the train ends.
        Checks if enough checkpoints exist and forces a save if necessary.
        """

        # Ensure the directory exists to avoid FileNotFoundError
        # We prefer self.dirpath if available, otherwise fallback to logger dir
        save_dir = self.dirpath or os.path.join(trainer.logger.log_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        # Count existing checkpoints
        try:
            current_checkpoints = len(
                [name for name in os.listdir(save_dir) if name.endswith(".ckpt")]
            )
        except FileNotFoundError:
            current_checkpoints = 0

        # Scenario 1: No best model was saved (e.g., metric never improved or short run)
        if self.best_model_path == "":
            print(
                cl.Fore.yellow
                + "No checkpoint saved. Forcing validation and saving last state..."
                + cl.Style.reset
            )

            # Optionally force a validation run to update metrics before saving
            # trainer.validate(pl_module)
            # call validation epoch end manually if needed
            trainer.validate(pl_module, dataloaders=trainer.val_dataloaders)

        # Scenario 2: Fewer than 3 checkpoints exist
        elif current_checkpoints < self.save_top_k:
            print(
                cl.Fore.yellow
                + f"Only {current_checkpoints} checkpoints found (target {self.save_top_k}). Forcing saving last checkpoint..."
                + cl.Style.reset
            )

            # Force save the current state to ensure we have the latest
            save_path = os.path.join(save_dir, "forced_last.ckpt")
            trainer.save_checkpoint(save_path)

        # Allow the parent class to finish its standard cleanup
        super().on_fit_end(trainer, pl_module)
