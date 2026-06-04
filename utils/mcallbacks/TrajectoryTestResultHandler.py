import os
import tempfile
import time
import uuid

import pandas as pd
import torch.nn.functional as F
from lightning_fabric.utilities.types import _PATH
from tqdm.auto import tqdm

from ..mloss.mSSIM import metricsSSIM
from ..utility import compute_error_scipy
from .common_imports import *

CS_EXCLUDED_PL_MODULES = [
    "RoNINModule",
    "TLIOModule",
    "IoNetModule",
    "RoNINModuleNext",
    "TLIOModuleNext",
    "IoNetModuleNext",
]


def compute_absolute_trajectory_error(est, gt):
    """
    The Absolute Trajectory Error (ATE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.

    Return:
        Absolution trajectory error, which is the Root Mean Squared Error between
        two trajectories.
    """
    return np.sqrt(np.mean((est - gt) ** 2))


def compute_relative_trajectory_error(est, gt, delta, max_delta=-1):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = (
        np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    )
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i] :] + gt[: -deltas[i]] - est[: -deltas[i]] - gt[deltas[i] :]
        rtes[i] = np.sqrt(np.mean(err**2))

    # The average of RTE of all window sized is returned.
    return np.mean(rtes)


def compute_trajectory_cosinesimilarty(est, gt, window_size):
    """
    Compute cosine similarity statistics for one trajectory inference.

    Args:
        est: estimated trajectory with shape (channels, N)
        gt: ground truth trajectory with shape (channels, N)

    Returns:
        A tensor of per-channel cosine similarities matching CosSimMetric
        semantics for a single trajectory sample.
    """
    est_t = (
        torch.as_tensor(est, dtype=torch.float32)
        .reshape(3, window_size, -1)
        .transpose(0, -1)
    )
    gt_t = (
        torch.as_tensor(gt, dtype=torch.float32)
        .reshape(3, window_size, -1)
        .transpose(0, -1)
    )

    return torch.vmap(F.cosine_similarity)(est_t, gt_t)


def compute_preintegration_cosinesimilarty_by_metric(est, gt, window_size):
    """
    Compute cosine similarity stats for trajectory channels before integration.

    Args:
        est: estimated trajectory with shape (channels, N)
        gt: ground truth trajectory with shape (channels, N)

    Returns:
        dict keyed by metric name ("pos", optionally "ori") with
        per-timestep cosine similarity tensors.
    """

    channels = est.shape[0]
    if channels > 3:
        metrics = {"pos": (0, 3), "ori": (3, 6)}
    elif channels == 3:
        metrics = {"pos": (0, 3)}
    elif channels == 2:
        metrics = {"pos": (0, 2)}
    else:
        raise ValueError(f"Unsupported channels for cosine similarity: {channels}")

    output = {}
    for metric_name, (start, end) in metrics.items():
        output[metric_name] = compute_trajectory_cosinesimilarty(
            est=est[start:end],
            gt=gt[start:end],
            window_size=window_size,
        )
        # print(
        #     cl.Fore.yellow, f"output[metric_name] {output[metric_name]}", cl.Style.reset
        # )
    return output


def compute_preintegration_ssim_by_metric(est, gt, window_size, window_type="gaussian"):
    """
    Compute SSIM stats for trajectory channels before integration.

    Args:
        est: estimated trajectory with shape (channels, N)
        gt: ground truth trajectory with shape (channels, N)
        window_size: window length used to form non-overlapping windows
        window_type: 'gaussian' or 'uniform' window for SSIM

    Returns:
        dict keyed by metric name ("pos", optionally "ori") with
        per-window SSIM tensors shaped (num_windows, window_size).
    """
    channels = est.shape[0]
    if channels > 3:
        metrics = {"pos": (0, 3), "ori": (3, 6)}
    elif channels == 3:
        metrics = {"pos": (0, 3)}
    elif channels == 2:
        metrics = {"pos": (0, 2)}
    else:
        raise ValueError(f"Unsupported channels for SSIM: {channels}")

    ssim_net = metricsSSIM(window_size=window_size, window_type=window_type)
    output = {}
    for metric_name, (start, end) in metrics.items():
        est_metric = torch.as_tensor(est[start:end], dtype=torch.float32)
        gt_metric = torch.as_tensor(gt[start:end], dtype=torch.float32)

        # Reshape into non-overlapping windows: (C, window_size, num_windows) -> (num_windows, C, window_size)
        est_windows = est_metric.reshape(end - start, window_size, -1).permute(2, 0, 1)
        gt_windows = gt_metric.reshape(end - start, window_size, -1).permute(2, 0, 1)

        # ssim_map: (num_windows, 1, window_size) -> squeeze to (num_windows, window_size)
        ssim_map = ssim_net.ssim_1d(est_windows, gt_windows).squeeze(1)
        output[metric_name] = ssim_map

    return output


class TrajectoryTestResultHandler(Callback):
    def __init__(
        self,
        video_rendering=False,
        sampling_rate=100,
        shared_dir: Optional[_PATH] = None,
    ):
        super().__init__()
        self.video_rendering = video_rendering
        self.sampling_rate = sampling_rate
        # store user-provided shared dir (may be None)
        self.shared_dir_arg = shared_dir
        self.estimation_result = {}
        rank_zero_info(
            cl.Fore.red
            + f"!!! PLEASE ENSURE ALL THE NODES SHARE A COMMON FILESYSTEM FOR TEST RESULT AGGREGATION !!!"
            + cl.Style.reset
        )

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        # Decide base shared root (user-provided > env var > system temp)
        if self.shared_dir_arg:
            shared_root = Path(self.shared_dir_arg)
        else:
            env_shared = os.environ.get("SHARED_TMP_DIR") or os.environ.get(
                "PL_SHARED_TMP_DIR"
            )
            if env_shared:
                shared_root = Path(env_shared)
            else:
                shared_root = Path(tempfile.gettempdir())

        # Create a run-unique shared directory on rank 0 and broadcast it to others
        if trainer.world_size > 1:
            if trainer.global_rank == 0:
                run_id = f"{pl_module.__class__.__name__}_{uuid.uuid4().hex}_{int(time.time())}"
                shared_dir = shared_root / run_id
                if shared_dir.exists():
                    print(
                        cl.Fore.yellow
                        + f"Removing existing shared directory {shared_dir}"
                        + cl.Style.reset
                    )
                    shutil.rmtree(shared_dir)
                shared_dir.mkdir(parents=True, exist_ok=True)
            else:
                shared_dir = None

            # Broadcast the chosen path from rank 0 to all workers
            shared_str = [str(shared_dir) if trainer.global_rank == 0 else ""]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(shared_str, src=0)
            else:
                try:
                    # Some strategies expose a broadcast helper
                    shared_str = trainer.strategy.broadcast(shared_str, src=0)
                except Exception:
                    pass

            self.shared_dir = Path(shared_str[0])
        else:
            # single-process: create local unique dir under shared_root
            run_id = (
                f"{pl_module.__class__.__name__}_{uuid.uuid4().hex}_{int(time.time())}"
            )
            self.shared_dir = shared_root / run_id
            if self.shared_dir.exists():
                shutil.rmtree(self.shared_dir)
            self.shared_dir.mkdir(parents=True, exist_ok=True)

        # Temporary results folder structure
        self._tmp = self.shared_dir / "test_results_tmp"
        if trainer.global_rank == 0:
            self._tmp.mkdir(parents=True, exist_ok=True)
            for i in range(trainer.world_size):
                (self._tmp / f"rank_{i}").mkdir(parents=True, exist_ok=True)

        self._tmp_rank_dir = self._tmp / f"rank_{trainer.global_rank}"
        self._tmp_rank_dir.mkdir(parents=True, exist_ok=True)

        if trainer.world_size > 1:
            trainer.strategy.barrier()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if outputs is None:
            return

        result_dict = outputs["results"]
        cpu_results = {k: v.clone().detach().cpu() for k, v in result_dict.items()}
        cpu_results["original_dataX"] = batch["dataX"].clone().detach().cpu()
        torch.save(
            cpu_results,
            self._tmp_rank_dir / f"batch_{batch_idx}_dl_{dataloader_idx}.pt",
        )

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        if trainer.world_size > 1:
            trainer.strategy.barrier()
        if trainer.global_rank != 0:
            return

        try:
            gathered_results = self._gather_results_from_disk(trainer)
            if not gathered_results:
                print("No test results found to process.")
                return

            results = {key: torch.cat(value) for key, value in gathered_results.items()}
            log_dir = trainer.logger.log_dir
            unique_labels = torch.unique(results.get("label", torch.tensor([0])))
            cs_enabled = pl_module.__class__.__name__ not in CS_EXCLUDED_PL_MODULES

            # --- MODIFICATION: Added TLR_total dict ---
            ATE_error_total = {"pos": defaultdict(list), "ori": defaultdict(list)}
            RTE_error_total = {"pos": defaultdict(list), "ori": defaultdict(list)}
            TLR_total = {
                "pos": defaultdict(list),
                "ori": defaultdict(list),
            }  # Trajectory Length Ratio
            cosinesimilarty_total = {
                "pos": defaultdict(list),
                "ori": defaultdict(list),
            }
            ssims_total = {
                "pos": defaultdict(list),
                "ori": defaultdict(list),
            }

            for label in unique_labels:
                self._process_label_group(
                    label,
                    results,
                    trainer,
                    log_dir,
                    ATE_error_total,
                    RTE_error_total,
                    TLR_total,
                    cosinesimilarty_total,
                    ssims_total,
                    cs_enabled,
                    window_size=pl_module.config["window_size"],
                )

            # Log final metrics
            console = Console()
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim", width=12)
            table.add_column("Label", style="dim")
            table.add_column("ATE (error)", justify="center")
            table.add_column("RTE (1 min error)", justify="center")
            # --- MODIFICATION: Added TLR Column ---
            table.add_column("TLR (Ratio)", justify="center")
            if cs_enabled:
                table.add_column("CS", justify="center")
                table.add_column("SSIM", justify="center")

            rows = []
            total_rows = []
            csv_data = []
            module_name = pl_module.__class__.__name__

            sorted_metrics = sorted(ATE_error_total.keys())
            for key in sorted_metrics:
                sorted_labels = sorted(ATE_error_total[key].keys())
                for k in sorted_labels:
                    if ATE_error_total[key][k]:
                        ate_mean = np.mean(ATE_error_total[key][k])
                        ate_std = np.std(ATE_error_total[key][k])
                        rte_mean = np.mean(RTE_error_total[key][k])
                        rte_std = np.std(RTE_error_total[key][k])

                        tlr_mean = (
                            np.mean(TLR_total[key][k]) if TLR_total[key][k] else 0.0
                        )
                        tlr_std = (
                            np.std(TLR_total[key][k]) if TLR_total[key][k] else 0.0
                        )
                        if cs_enabled and cosinesimilarty_total[key][k]:
                            cs_values = torch.cat(cosinesimilarty_total[key][k])
                            cos_mean = cs_values.mean().item()
                            cos_std = cs_values.std().item()
                        else:
                            cos_mean = None
                            cos_std = None

                        if cs_enabled and ssims_total[key][k]:
                            ssim_values = torch.cat(ssims_total[key][k])
                            ssim_mean = ssim_values.mean().item()
                            ssim_std = ssim_values.std().item()
                        else:
                            ssim_mean = None
                            ssim_std = None

                        extra_metrics = {}
                        if cs_enabled:
                            if cos_mean is not None:
                                extra_metrics[f"CS_{key}_{k}/mean"] = cos_mean
                                extra_metrics[f"CS_{key}_{k}/std"] = cos_std
                            if ssim_mean is not None:
                                extra_metrics[f"SSIM_{key}_{k}/mean"] = ssim_mean
                                extra_metrics[f"SSIM_{key}_{k}/std"] = ssim_std

                        pl_module.logger.log_metrics(
                            {
                                f"ATE_{key}_{k}/mean": ate_mean,
                                f"ATE_{key}_{k}/std": ate_std,
                                f"RTE_{key}_{k}/mean": rte_mean,
                                f"RTE_{key}_{k}/std": rte_std,
                                f"TLR_{key}_{k}/mean": tlr_mean,
                                f"TLR_{key}_{k}/std": tlr_std,
                                **extra_metrics,
                            },
                            step=pl_module.current_epoch,
                        )
                        row_data = (
                            key,
                            k,
                            f"{ate_mean:.4f} ± {ate_std:.4f}",
                            f"{rte_mean:.4f} ± {rte_std:.4f}",
                            f"{tlr_mean:.4f} ± {tlr_std:.4f}",
                        )
                        if cs_enabled:
                            cs_str = (
                                f"{cos_mean:.4f} ± {cos_std:.4f}"
                                if cos_mean is not None
                                else "-"
                            )
                            ssim_str = (
                                f"{ssim_mean:.4f} ± {ssim_std:.4f}"
                                if ssim_mean is not None
                                else "-"
                            )
                            row_data = row_data + (cs_str, ssim_str)

                        extra_csv = {}
                        if cs_enabled:
                            if cos_mean is not None:
                                extra_csv["CS_mean"] = cos_mean
                                extra_csv["CS_std"] = cos_std
                            if ssim_mean is not None:
                                extra_csv["SSIM_mean"] = ssim_mean
                                extra_csv["SSIM_std"] = ssim_std

                        csv_data.append(
                            {
                                "Metric": key,
                                "Label": k,
                                "ATE_mean": ate_mean,
                                "ATE_std": ate_std,
                                "RTE_mean": rte_mean,
                                "RTE_std": rte_std,
                                "TLR_mean": tlr_mean,
                                "TLR_std": tlr_std,
                                "modules name": module_name,
                                **extra_csv,
                            }
                        )
                        if k == "total":
                            total_rows.append(row_data)
                        else:
                            rows.append(row_data)

            last_metric = None
            for row in rows:
                if last_metric is not None and row[0] != last_metric:
                    table.add_section()
                table.add_row(*row)
                last_metric = row[0]

            if rows and total_rows:
                table.add_section()

            for row in total_rows:
                table.add_row(*row, style="bold")

            console.print(table)

            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_path = Path(log_dir) / "test_metrics.csv"
                df.to_csv(csv_path, index=False)
                print(f"Test metrics saved to {csv_path}")
        finally:
            if self._tmp and self._tmp.exists():
                shutil.rmtree(self.shared_dir)

    def _gather_results_from_disk(self, trainer):
        all_results = defaultdict(list)
        for rank_dir in sorted(self._tmp.iterdir()):
            if not rank_dir.is_dir():
                continue
            files = sorted(
                rank_dir.glob("*.pt"),
                key=lambda f: tuple(
                    int(x)
                    for x in f.stem.replace("batch_", "").replace("dl_", "").split("_")
                ),
            )
            for file_path in files:
                batch_result = torch.load(
                    file_path, map_location="cpu", weights_only=False, mmap=True
                )
                for key, value in batch_result.items():
                    all_results[key].append(value)
        return all_results

    def _process_label_group(
        self,
        label,
        results,
        trainer,
        log_dir,
        ATE_total,
        RTE_total,
        TLR_total,
        cosinesimilarty_total,
        ssims_total,
        cs_enabled,
        window_size,
    ):
        label_mask = (
            results.get("label", torch.ones_like(results["name"], dtype=torch.bool))
            == label
        )
        if not torch.any(label_mask):
            return

        label_names = results["name"][label_mask]
        unique_names = torch.unique(label_names)
        label_n = INV_DATASET_DICT[label.item()]
        rank_zero_info(f"- Processing label: {label_n}")

        for name in tqdm(unique_names, desc=f"Processing {label_n}", leave=False):
            name_mask = label_mask & (results["name"] == name)
            est, targ, original_x, length, str_name = self._prepare_sequence_data(
                name_mask, results, trainer, label
            )

            save_path = Path(log_dir) / f"label_{label_n}" / f"{str_name}"
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save(est, save_path / "estimates.pt")
            torch.save(targ, save_path / "targets.pt")
            torch.save(original_x, save_path / "original_dataX.pt")

            # Compute per-inference cosine similarity on raw trajectories
            # before integration and before dataset-level aggregation.
            if cs_enabled:
                cosine_by_metric = compute_preintegration_cosinesimilarty_by_metric(
                    est, targ, window_size=window_size
                )
                for metric_name, cosine_values in cosine_by_metric.items():
                    cosinesimilarty_total[metric_name][label_n].append(cosine_values)
                    cosinesimilarty_total[metric_name]["total"].append(cosine_values)
                # Also compute SSIM per the same pre-integration windows
                ssim_by_metric = compute_preintegration_ssim_by_metric(
                    est, targ, window_size=window_size
                )
                for metric_name, ssim_values in ssim_by_metric.items():
                    ssims_total[metric_name][label_n].append(ssim_values)
                    ssims_total[metric_name]["total"].append(ssim_values)

            est_integrated, targ_integrated = self._integrate_trajectories(
                est,
                targ,
                # isIoNetModule=trainer.model.__class__.__name__ == "IoNetModule",
            )
            torch.save(est_integrated, save_path / "integral_estimates.pt")
            torch.save(targ_integrated, save_path / "integral_targets.pt")

            if self.video_rendering:
                animation_fun(
                    est=est_integrated,
                    targ=targ_integrated,
                    str_name=f"label_{label_n}_{str_name}",
                    save_path=save_path,
                    fps=self.sampling_rate,
                )

            self._calculate_and_log_errors(
                est_integrated,
                targ_integrated,
                label_n,
                ATE_total,
                RTE_total,
                TLR_total,
            )

    def _prepare_sequence_data(self, name_mask, results, trainer, label):
        indexes = results["index"][name_mask]
        sorted_indices = torch.argsort(indexes)

        est = results["dataX"][name_mask][sorted_indices]
        targ = results["dataY"][name_mask][sorted_indices]
        original_x = results["original_dataX"][name_mask][sorted_indices]
        length = results["dataL"][name_mask][sorted_indices]

        temp_est = torch.cat([est[i, :, :L] for i, L in enumerate(length)], dim=-1)
        temp_targ = torch.cat([targ[i, :, :L] for i, L in enumerate(length)], dim=-1)
        temp_original_x = torch.cat(
            [original_x[i, :, :L] for i, L in enumerate(length)], dim=-1
        )

        dm = trainer.datamodule
        name_val = results["name"][name_mask][0].item()
        if len(torch.unique(results.get("label", torch.tensor([0])))) > 1:
            for idx, i in enumerate(range(len(dm.test_dataset.datasets))):
                if label.item() == dm.test_dataset.datasets[i]._label:
                    dataset_idx = i
                    break
            str_name = dm.test_dataset.datasets[dataset_idx].files[name_val]
        else:
            str_name = dm.test_dataset.files[name_val]

        if "OxIOD" in INV_DATASET_DICT[label.item()]:
            str_name = str_name[1]

        clean_name = (
            str(str_name)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
        ).rsplit(".", 1)[0]
        return (
            temp_est.numpy(),
            temp_targ.numpy(),
            temp_original_x.numpy(),
            length,
            clean_name,
        )

    def _integrate_trajectories(self, est, targ, isIoNetModule=False):
        est_integrated = est.copy()
        targ_integrated = targ.copy()
        channels = est.shape[0]

        if channels == 2:
            pos_channels = 2
        else:
            pos_channels = 3

        # Integrate Position (works for 2D or 3D)
        est_integrated[:pos_channels] = (
            est_integrated[:pos_channels].cumsum(axis=-1) / self.sampling_rate
        )
        targ_integrated[:pos_channels] = (
            targ_integrated[:pos_channels].cumsum(axis=-1) / self.sampling_rate
        )

        # Integrate Orientation only if we have enough channels (e.g. 6)
        if channels >= 6 and not isIoNetModule:
            est_integrated[pos_channels:] = integrate_orientation(
                est_integrated[pos_channels:], 1 / self.sampling_rate
            )
            targ_integrated[pos_channels:] = integrate_orientation(
                targ_integrated[pos_channels:], 1 / self.sampling_rate
            )

        return est_integrated, targ_integrated

    def _angular_difference(self, a, b):
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def _calculate_and_log_errors(
        self,
        est_integrated,
        targ_integrated,
        label_n,
        ATE_total,
        RTE_total,
        TLR_total,
    ):
        channels = est_integrated.shape[0]

        # Determine metrics and channel count per metric
        if channels > 3:
            metric = ["pos", "ori"]
            jump = 3
        elif channels == 3:
            metric = ["pos"]
            jump = 3
        elif channels == 2:
            metric = ["pos"]
            jump = 2  # Set jump to 2 for 2D position
        else:
            raise ValueError(f"This should not happen, unsupported channels {channels}")

        window_size_60 = 60 * self.sampling_rate

        # Pad to the nearest full window using edge mode
        remainder = est_integrated.shape[1] % window_size_60
        if remainder == 0:
            pad_width = ((0, 0), (0, 0))
        else:
            pad_width = ((0, 0), (0, window_size_60 - remainder))

        est_v_padded = np.pad(
            est_integrated,
            pad_width,
            mode="edge",
        )
        targ_v_padded = np.pad(
            targ_integrated,
            pad_width,
            mode="edge",
        )

        for idxm, m in enumerate(metric):
            # Extract current metric data
            current_est = est_integrated[idxm * jump : idxm * jump + jump]
            current_targ = targ_integrated[idxm * jump : idxm * jump + jump]

            est_v_crop = est_v_padded[idxm * jump : idxm * jump + jump]
            targ_v_crop = targ_v_padded[idxm * jump : idxm * jump + jump]

            # --- ATE Calculation ---
            if m == "ori":
                ate_errors = compute_error_scipy(current_est.T, current_targ.T)
                ates = np.sqrt(np.mean(ate_errors**2))
            else:
                # Transpose from (channels, N) to (N, channels)
                est_t = current_est.T
                targ_t = current_targ.T
                ates = compute_absolute_trajectory_error(est_t, targ_t)

            ATE_total[m][label_n].append(ates)
            ATE_total[m]["total"].append(ates)

            # --- RTE Calculation ---
            # FIX: Use 'jump' instead of hardcoded 3 for the reshape
            est_v_crop, targ_v_crop = est_v_crop.reshape(
                jump, -1, window_size_60
            ), targ_v_crop.reshape(jump, -1, window_size_60)

            if m == "ori":
                est_v_crop_rel = self._angular_difference(
                    est_v_crop, est_v_crop[:, :, :1]
                )
                targ_v_crop_rel = self._angular_difference(
                    targ_v_crop, targ_v_crop[:, :, :1]
                )

                pred_delta = est_v_crop_rel[:, :, -1]
                targ_delta = targ_v_crop_rel[:, :, -1]

                rte_errors = compute_error_scipy(pred_delta.T, targ_delta.T)
                rtes = rte_errors.mean()

            else:
                # Transpose from (channels, N) to (N, channels)
                est_t = current_est.T
                targ_t = current_targ.T

                # Ensure delta is not larger than the trajectory length
                delta = min(window_size_60, est_t.shape[0] - 1)
                if delta > 0:
                    rtes = compute_relative_trajectory_error(est_t, targ_t, delta=delta)
                else:
                    rtes = 0.0  # Cannot compute RTE if trajectory is too short

            if np.ndim(rtes) == 0:
                RTE_total[m][label_n].append(rtes)
                RTE_total[m]["total"].append(rtes)
            else:
                RTE_total[m][label_n].extend(list(rtes))
                RTE_total[m]["total"].extend(list(rtes))

            # --- TLR Calculation ---
            if m == "ori":
                est_steps = compute_error_scipy(
                    current_est[:, 1:].T, current_est[:, :-1].T
                )
                est_len = np.sum(est_steps)

                targ_steps = compute_error_scipy(
                    current_targ[:, 1:].T, current_targ[:, :-1].T
                )
                targ_len = np.sum(targ_steps)
            else:
                est_diff = np.diff(current_est, axis=1)
                targ_diff = np.diff(current_targ, axis=1)

                est_len = np.sum(np.linalg.norm(est_diff, axis=0))
                targ_len = np.sum(np.linalg.norm(targ_diff, axis=0))

            tlr = est_len / targ_len if targ_len > 1e-7 else 0.0

            TLR_total[m][label_n].append(tlr)
            TLR_total[m]["total"].append(tlr)
