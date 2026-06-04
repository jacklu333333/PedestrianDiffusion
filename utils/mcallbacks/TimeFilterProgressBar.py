from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n

from .common_imports import *


class TimeFilterProgressBar(TQDMProgressBar):
    def __init__(
        self,
        keep_keywords: list = None,
        remove_keywords: list = None,
        bold_keywords: list = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.keep_keywords = keep_keywords
        self.remove_keywords = remove_keywords
        self.bold_keywords = bold_keywords

    def filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self.keep_keywords:
            metrics = {
                k: v
                for k, v in metrics.items()
                if any(keyword in k for keyword in self.keep_keywords)
            }
        if self.remove_keywords:
            metrics = {
                k: v
                for k, v in metrics.items()
                if not any(keyword in k for keyword in self.remove_keywords)
            }
        return metrics

    def add_time(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["time"] = current_time
        return metrics

    def add_eta(
        self, metrics: Dict[str, Any], progress_bar, current_n: Optional[int] = None
    ) -> Dict[str, Any]:
        # prefer the caller-provided current position because progress_bar.n
        # may not have been updated yet (we call _update_n after computing metrics)
        n = current_n if current_n is not None else getattr(progress_bar, "n", 0)
        total = getattr(progress_bar, "total", None)

        if n <= 0 or total is None:
            metrics["ETA"] = "N/A"
            return metrics

        # use format_dict.get to avoid KeyError; fallback to 0.0
        elapsed = progress_bar.format_dict.get("elapsed", 0.0)
        if not elapsed:
            # if elapsed is zero-ish, avoid division by zero and return N/A
            metrics["ETA"] = "N/A"
            return metrics

        remaining_steps = total - n
        try:
            eta_seconds = (elapsed / n) * remaining_steps
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            metrics["ETA"] = eta_time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            metrics["ETA"] = "N/A"
        return metrics

    def add_learning_rate(
        self, metrics: Dict[str, Any], trainer: "pl.Trainer"
    ) -> Dict[str, Any]:
        for idx, optimizer in enumerate(trainer.optimizers):
            learning_rate = optimizer.param_groups[0]["lr"]
            metrics[f"lr_{idx}"] = learning_rate
        return metrics

    def sort_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # sort it in the order of train val time
        sorted_metrics = {}
        for keyword in ["v_num", "train", "val", "lr", "time", "ETA"]:
            for k, v in metrics.items():
                if keyword in k:
                    sorted_metrics[k] = v
        return sorted_metrics

    def apply_bold(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if not self.bold_keywords:
            return metrics

        new_metrics = {}
        for k, v in metrics.items():
            if any(keyword in k for keyword in self.bold_keywords):
                # Bold the entire "key=value" string by manipulating key and value
                # ANSI escape code for bold is \033[1m, reset is \033[0m
                new_key = f"\033[1m{k}"
                if isinstance(v, float):
                    v_str = f"{v:.3f}"
                else:
                    v_str = str(v)
                new_value = f"{v_str}\033[0m"
                new_metrics[new_key] = new_value
            else:
                new_metrics[k] = v
        return new_metrics

    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     n = batch_idx + 1
    #     if self._should_update(n, self.train_progress_bar.total):
    #         metrics = self.get_metrics(trainer, pl_module)
    #         metrics = self.filter_metrics(metrics)

    #         metrics = self.add_time(metrics)
    #         metrics = self.add_learning_rate(metrics, trainer)
    #         metrics = self.add_eta(metrics, self.train_progress_bar)

    #         metrics = self.sort_metrics(metrics)

    #         _update_n(self.train_progress_bar, n)
    #         self.train_progress_bar.set_postfix(metrics)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            metrics = self.add_learning_rate(metrics, trainer)
            metrics = self.add_eta(metrics, self.train_progress_bar, current_n=n)

            metrics = self.sort_metrics(metrics)
            metrics = self.apply_bold(metrics)

            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(metrics)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # do the same thing as on_train_batch_end

        if not self.train_progress_bar.disable:
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            metrics = self.add_learning_rate(metrics, trainer)
            metrics = self.add_eta(metrics, self.train_progress_bar)

            metrics = self.sort_metrics(metrics)
            metrics = self.apply_bold(metrics)

            self.train_progress_bar.set_postfix(metrics)

    # def on_validation_batch_start(
    #     self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    # ):
    #     n = batch_idx + 1
    #     if self._should_update(n, self.val_progress_bar.total):
    #         metrics = self.get_metrics(trainer, pl_module)
    #         metrics = self.filter_metrics(metrics)

    #         metrics = self.add_time(metrics)
    #         metrics = self.add_learning_rate(metrics, trainer)
    #         metrics = self.add_eta(metrics, self.val_progress_bar)

    #         metrics = self.sort_metrics(metrics)

    #         _update_n(self.val_progress_bar, n)
    #         self.val_progress_bar.set_postfix(metrics)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            metrics = self.add_learning_rate(metrics, trainer)
            metrics = self.add_eta(metrics, self.val_progress_bar, current_n=n)

            metrics = self.sort_metrics(metrics)
            metrics = self.apply_bold(metrics)

            _update_n(self.val_progress_bar, n)
            self.val_progress_bar.set_postfix(metrics)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._train_progress_bar is not None and trainer.state.fn == "fit":
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            # metrics = self.add_learning_rate(metrics, trainer)
            metrics = self.add_eta(metrics, self.val_progress_bar)

            metrics = self.sort_metrics(metrics)
            metrics = self.apply_bold(metrics)

            self.val_progress_bar.set_postfix(metrics)
        self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    # def on_test_batch_start(
    #     self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    # ):
    #     # update the progress bar on the starting of the training as well
    #     n = batch_idx + 1
    #     if self._should_update(n, self.test_progress_bar.total):
    #         metrics = self.get_metrics(trainer, pl_module)
    #         metrics = self.filter_metrics(metrics)

    #         metrics = self.add_time(metrics)
    #         # metrics = self.add_learning_rate(metrics, trainer)
    #         metrics = self.add_eta(metrics, self.test_progress_bar)

    #         metrics = self.sort_metrics(metrics)

    #         _update_n(self.test_progress_bar, n)
    #         self.test_progress_bar.set_postfix(metrics)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.test_progress_bar.total):
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            # metrics = self.add_learning_rate(metrics, trainer)
            metrics = self.add_eta(metrics, self.test_progress_bar, current_n=n)

            metrics = self.sort_metrics(metrics)
            metrics = self.apply_bold(metrics)

            _update_n(self.test_progress_bar, n)
            self.test_progress_bar.set_postfix(metrics)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.test_progress_bar is not None:
            metrics = self.get_metrics(trainer, pl_module)
            metrics = self.filter_metrics(metrics)

            metrics = self.add_time(metrics)
            # metrics = self.add_learning_rate(metrics, trainer)
            metrics = self.add_eta(metrics, self.test_progress_bar)

            metrics = self.sort_metrics(metrics)
            metrics = self.apply_bold(metrics)

            self.test_progress_bar.set_postfix(metrics)
            self.test_progress_bar.close()
