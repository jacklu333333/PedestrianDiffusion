import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Assuming this exists in your environment
from utils.utility import compute_error_scipy

# --- Reverted to Original Helper Functions (Component-wise RMSE) ---


def compute_absolute_trajectory_error(est, gt):
    """
    Original component-wise RMSE.
    """
    return np.sqrt(np.mean((est - gt) ** 2))


def compute_relative_trajectory_error(est, gt, delta, max_delta=-1):
    """
    Original component-wise RTE.
    """
    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = (
        np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    )
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        err = est[deltas[i] :] + gt[: -deltas[i]] - est[: -deltas[i]] - gt[deltas[i] :]
        rtes[i] = np.sqrt(np.mean(err**2))

    return np.mean(rtes)


def angular_difference(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi


# --- Updated Metrics Calculation ---


def calculate_metrics(est_integrated, targ_integrated, sampling_rate=100, is_2d=False):
    """
    Computes ATE, RTE, and TLR.
    Now aligns 2D Position data to 3D (pads Z=0) to ensure ATE/RTE scaling matches 3D mode.
    """
    channels = est_integrated.shape[0]

    # Determine metrics based on mode or channels
    metrics_to_compute = []
    if channels >= 6:
        metrics_to_compute = ["pos", "ori"]
    elif channels >= 3:
        metrics_to_compute = ["pos"]
        # If we are in 2D mode but have 3 channels (x,y,yaw), we might want ori
        if is_2d:
            metrics_to_compute = ["pos", "ori"]
    elif channels == 2:
        metrics_to_compute = ["pos"]

    window_size_60 = 60 * sampling_rate

    # Padding for RTE windowing
    remainder = est_integrated.shape[1] % window_size_60
    if remainder == 0:
        pad_width = ((0, 0), (0, 0))
    else:
        pad_width = ((0, 0), (0, window_size_60 - remainder))

    est_v_padded = np.pad(est_integrated, pad_width, mode="edge")
    targ_v_padded = np.pad(targ_integrated, pad_width, mode="edge")

    ate_results = {}
    rte_results = {}
    tlr_results = {}

    for m in metrics_to_compute:
        if m == "pos":
            if is_2d:
                # --- THE FIX: Pad 2D (x,y) to 3D (x,y,0) ---
                # This ensures the denominator in np.mean is 3*N, matching the 3D calculation.

                # Slicing x, y
                raw_est = est_integrated[0:2]
                raw_targ = targ_integrated[0:2]

                # Create Zero Z-axis
                zeros = np.zeros((1, raw_est.shape[1]))

                # Stack to make (3, N)
                current_est = np.vstack([raw_est, zeros])
                current_targ = np.vstack([raw_targ, zeros])

                # Do the same for padded versions (for RTE)
                raw_est_pad = est_v_padded[0:2]
                raw_targ_pad = targ_v_padded[0:2]
                zeros_pad = np.zeros((1, raw_est_pad.shape[1]))

                est_v_crop_full = np.vstack([raw_est_pad, zeros_pad])
                targ_v_crop_full = np.vstack([raw_targ_pad, zeros_pad])

            else:
                # Standard 3D
                slc = slice(0, 3)
                current_est = est_integrated[slc]
                current_targ = targ_integrated[slc]
                est_v_crop_full = est_v_padded[slc]
                targ_v_crop_full = targ_v_padded[slc]

        elif m == "ori":
            if is_2d:
                # 2D Orientation: Last channel (Yaw)
                current_est = est_integrated[-1:, :]
                current_targ = targ_integrated[-1:, :]
                est_v_crop_full = est_v_padded[-1:, :]
                targ_v_crop_full = targ_v_padded[-1:, :]
            else:
                # 3D Orientation
                current_est = est_integrated[3:6]
                current_targ = targ_integrated[3:6]
                est_v_crop_full = est_v_padded[3:6]
                targ_v_crop_full = targ_v_padded[3:6]

        # ==========================
        #       ATE Calculation
        # ==========================
        if m == "ori":
            if is_2d:
                diff = angular_difference(current_est, current_targ)
                ates = np.sqrt(np.mean(diff**2))
            else:
                ate_errors = compute_error_scipy(current_est.T, current_targ.T)
                ates = np.sqrt(np.mean(ate_errors**2))
        else:
            # Position Error using the Unified 3D shape
            est_t = current_est.T
            targ_t = current_targ.T
            ates = compute_absolute_trajectory_error(est_t, targ_t)

        ate_results[m] = ates

        # ==========================
        #       RTE Calculation
        # ==========================
        num_windows = est_v_crop_full.shape[1] // window_size_60
        num_channels = est_v_crop_full.shape[0]

        est_v_crop = est_v_crop_full.reshape(num_channels, num_windows, window_size_60)
        targ_v_crop = targ_v_crop_full.reshape(
            num_channels, num_windows, window_size_60
        )

        if m == "ori":
            est_v_crop_rel = angular_difference(est_v_crop, est_v_crop[:, :, :1])
            targ_v_crop_rel = angular_difference(targ_v_crop, targ_v_crop[:, :, :1])

            pred_delta = est_v_crop_rel[:, :, -1]
            targ_delta = targ_v_crop_rel[:, :, -1]

            if is_2d:
                rte_diff = angular_difference(pred_delta, targ_delta)
                rtes = np.mean(np.abs(rte_diff))
            else:
                rte_errors = compute_error_scipy(pred_delta.T, targ_delta.T)
                rtes = rte_errors.mean()
        else:
            est_t = current_est.T
            targ_t = current_targ.T
            delta = min(window_size_60, est_t.shape[0] - 1)
            if delta > 0:
                rtes = compute_relative_trajectory_error(est_t, targ_t, delta=delta)
            else:
                rtes = 0.0

        rte_results[m] = rtes

        # ==========================
        #       TLR Calculation
        # ==========================
        # TLR is a ratio of lengths, so padding Z=0 (which adds 0 length)
        # doesn't change the result, but we use the padded vars for consistency.
        if m == "ori":
            if is_2d:
                est_steps = angular_difference(current_est[:, 1:], current_est[:, :-1])
                targ_steps = angular_difference(
                    current_targ[:, 1:], current_targ[:, :-1]
                )
                est_len = np.sum(np.abs(est_steps))
                targ_len = np.sum(np.abs(targ_steps))
            else:
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
        tlr_results[m] = tlr

    return ate_results, rte_results, tlr_results


def process_directory(
    base_dir, sampling_rate=None, output_file="recomputed_metrics.csv", is_2d=False
):
    base_path = Path(base_dir)
    config = json.load(open(base_path / "config.json", "r"))

    if sampling_rate is None:
        sampling_rate = config["sampling_rate"]

    if not base_path.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    ATE_total = {"pos": defaultdict(list), "ori": defaultdict(list)}
    RTE_total = {"pos": defaultdict(list), "ori": defaultdict(list)}
    TLR_total = {"pos": defaultdict(list), "ori": defaultdict(list)}

    label_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("label_")
    ]
    if not label_dirs:
        print("No label directories found.")
        return

    mode_str = "2D" if is_2d else "3D"
    print(f"Starting processing in {mode_str} mode...")

    for label_dir in label_dirs:
        if "ADVIO" in str(label_dir):
            continue
        clean_label = label_dir.name[6:]
        print(f"Processing label: {clean_label}")

        for seq_dir in label_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            est_path = seq_dir / "integral_estimates.pt"
            targ_path = seq_dir / "integral_targets.pt"

            if not est_path.exists() or not targ_path.exists():
                continue

            try:
                est = torch.load(est_path, weights_only=False)
                targ = torch.load(targ_path, weights_only=False)
                if isinstance(est, torch.Tensor):
                    est = est.numpy()
                if isinstance(targ, torch.Tensor):
                    targ = targ.numpy()
                if not isinstance(est, np.ndarray):
                    est = np.array(est)
                if not isinstance(targ, np.ndarray):
                    targ = np.array(targ)

                ates, rtes, tlrs = calculate_metrics(
                    est, targ, sampling_rate, is_2d=is_2d
                )

                for m in ates.keys():
                    ATE_total[m][clean_label].append(ates[m])
                    ATE_total[m]["total"].append(ates[m])
                    RTE_total[m][clean_label].append(rtes[m])
                    RTE_total[m]["total"].append(rtes[m])
                    TLR_total[m][clean_label].append(tlrs[m])
                    TLR_total[m]["total"].append(tlrs[m])

            except Exception as e:
                print(f"Error processing {seq_dir}: {e}")

    # --- Rich Table Output ---
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column(f"Metric ({mode_str})", style="dim", width=12)
    table.add_column("Label", style="dim")
    table.add_column("ATE (error)", justify="center")
    table.add_column("RTE (1 min error)", justify="center")
    table.add_column("TLR (Ratio)", justify="center")

    rows = []
    total_rows = []
    csv_data = []

    sorted_metrics = sorted(ATE_total.keys())
    for key in sorted_metrics:
        sorted_labels = sorted(ATE_total[key].keys())
        for k in sorted_labels:
            if ATE_total[key][k]:
                ate_mean = np.mean(ATE_total[key][k])
                ate_std = np.std(ATE_total[key][k])
                rte_mean = np.mean(RTE_total[key][k])
                rte_std = np.std(RTE_total[key][k])
                tlr_mean = np.mean(TLR_total[key][k])
                tlr_std = np.std(TLR_total[key][k])
                print(f"length of ATE_total[{key}][{k}]: {len(ATE_total[key][k])}")
                print(f"length of RTE_total[{key}][{k}]: {len(RTE_total[key][k])}")
                print(f"length of TLR_total[{key}][{k}]: {len(TLR_total[key][k])}")

                row_data = (
                    key,
                    k,
                    f"{ate_mean:.4f} ± {ate_std:.4f}",
                    f"{rte_mean:.4f} ± {rte_std:.4f}",
                    f"{tlr_mean:.4f} ± {tlr_std:.4f}",
                )

                if k == "total":
                    total_rows.append(row_data)
                else:
                    rows.append(row_data)

                csv_data.append(
                    {
                        "Mode": mode_str,
                        "Metric": key,
                        "Label": k,
                        "ATE_mean": ate_mean,
                        "ATE_std": ate_std,
                        "RTE_mean": rte_mean,
                        "RTE_std": rte_std,
                        "TLR_mean": tlr_mean,
                        "TLR_std": tlr_std,
                    }
                )

    for row in rows:
        table.add_row(*row)
    if rows and total_rows:
        table.add_section()
    for row in total_rows:
        table.add_row(*row)

    console.print(
        Panel(
            table,
            title=f'Module: {str(base_path.parents[1]).replace("logs_", "")} [{mode_str}]',
            expand=False,
        )
    )

    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to results directory.")
    parser.add_argument("--sampling_rate", type=int, default=None)
    parser.add_argument("--output", type=str, default="recomputed_metrics.csv")
    parser.add_argument(
        "--2d", dest="is_2d", action="store_true", help="Compute metrics in 2D."
    )
    args = parser.parse_args()
    process_directory(args.directory, args.sampling_rate, args.output, is_2d=args.is_2d)
