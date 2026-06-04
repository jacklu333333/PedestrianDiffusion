from .common_imports import *


class TrajectoryTestMetricsSummaryCallback(Callback):
    """
    A callback to print a summary of test metrics in a 'mean ± std' format
    and a separate table for other metrics. It also suppresses the default
    Pytorch Lightning test summary.
    """

    @rank_zero_only
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Called when the test epoch ends."""

        metrics = trainer.callback_metrics
        mean_std_data = []
        other_data = []
        processed_keys = set()

        # Sort keys for a consistent output order
        sorted_keys = sorted(metrics.keys())

        # First pass: process mean/std pairs
        for key in sorted_keys:
            if key in processed_keys or not ("_mean" in key or "_std" in key):
                continue

            if key.endswith("_mean"):
                base_key = key.removesuffix("_mean")
                mean_key = key
                std_key = f"{base_key}_std"
            else:  # key ends with _std
                base_key = key.removesuffix("_std")
                mean_key = f"{base_key}_mean"
                std_key = key

            if mean_key in metrics and std_key in metrics:
                mean_val = metrics[mean_key].item()
                std_val = metrics[std_key].item()

                value_str = f"{mean_val:.4f} ± {std_val:.4f}"
                clean_name = base_key.replace("/test", "")

                clean_name, value_str = self._format_metric_string(
                    clean_name, mean_val, value_str
                )

                mean_std_data.append([clean_name, value_str])

                processed_keys.add(mean_key)
                processed_keys.add(std_key)

        # Second pass: process other metrics
        for key in sorted_keys:
            if key in processed_keys:
                continue

            # Ignore internal lightning metrics
            if "v_num" in key:
                continue

            value = metrics[key].item()
            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            clean_name = key.replace("/test", "")
            clean_name = clean_name.replace("_mean", "")

            clean_name, value_str = self._format_metric_string(
                clean_name, value, value_str
            )

            other_data.append([clean_name, value_str])
            processed_keys.add(key)

        # Print the tables
        self._print_table(
            "TEST Paired Metrics Summary (mean ± std)",
            ["Metric", "Mean ± Std"],
            mean_std_data,
        )
        self._print_table("TEST Other Test Metrics", ["Metric", "Value"], other_data)

        # # Clear metrics to prevent default Pytorch Lightning summary
        # metrics.clear()

    def _format_metric_string(self, clean_name, value, value_str):
        if "SIM" in clean_name or "simVector" in clean_name:
            is_loss = "loss" in clean_name.lower()
            # Bad if: (Loss AND High) OR (Not Loss AND Low)
            # This is equivalent to: is_loss == is_high
            is_bad = is_loss == (value > 0.5)

            color = "red" if is_bad else "green"
            value_str = f"[bold {color}]{value_str}[/bold {color}]"
            clean_name = f"[bold]{clean_name}[/bold]"
        return clean_name, value_str

    def _print_table(self, title, headers, data):
        if not data:
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=box.MINIMAL_HEAVY_HEAD,
            show_edge=True,
            # expand=True,
        )
        table.add_column(headers[0], justify="left", style="cyan", no_wrap=True)
        table.add_column(headers[1], justify="center", style="magenta")

        for row in data:
            table.add_row(*row)

        print("\n")
        console = Console(force_terminal=True)
        console.print(Panel(table, title=title, border_style="white", expand=True))
