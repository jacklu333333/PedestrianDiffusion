import glob
import math
import os
import random
import sys
from pathlib import Path
from typing import Union

import colored as cl
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.spatial.transform import Rotation as R


def compute_error_scipy(pred_euler, target_euler, return_in_degrees=False):
    """
    Computes the error in degrees between two sets of Euler angles.

    Args:
        pred_euler: (N, 3) array of predicted Euler angles
        target_euler: (N, 3) array of target Euler angles
    """
    # 1. Create Rotation objects
    # Ensure you specify the correct order (e.g., 'xyz', 'zyx')
    # and whether input is in degrees or radians.
    r_pred = R.from_euler("xyz", pred_euler)
    r_target = R.from_euler("xyz", target_euler)

    # 2. Compute the relative rotation (Difference)
    # Conceptually: R_diff = R_target * R_pred^(-1)
    # The magnitude of this difference represents the error.
    r_diff = r_target * r_pred.inv()

    # 3. Get the magnitude (angle) in radians and convert to degrees
    # .magnitude() returns the geodesic distance in radians [0, pi]
    errors_rad = r_diff.magnitude()
    if not return_in_degrees:
        return errors_rad

    errors_deg = np.rad2deg(errors_rad)
    return errors_deg


def set_seed(seed: int = 42, reproducibility: bool = True):
    # 基礎種子設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(
        seed, workers=True
    )  # 建議加上 workers=True 以確保 DataLoader 重現性

    if reproducibility:
        # 1. 精度控制 (關閉 TF32)
        torch.set_float32_matmul_precision("highest")
        if hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = "ieee"

        # 2. 演算法確定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)
    else:
        # 1. 精度控制 (開啟 TF32 加速)
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"

        # 2. 效能優先
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


# class weight_finder():
#     def __init__(self, path: Path):
#         self.model = torch.load(path)


#     def find(self, name):
#         for n, p in self.model.named_parameters():
#             if name in n:
#                 print(n)
#                 print(p)
#                 print(p.size())
#                 print()


def weight_finder(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)

    if not ("checckpoints" in path.parts):
        # check subfolder "checkpoints" exist or not
        if not path.exists():
            print(cl.Style.red, f"Path {path} does not exist", cl.Style.reset)
            sys.exit()
        else:
            path = path / "checkpoints"

    # get all the files and subdoler in the folder
    files = glob.glob(str(path) + "/*")
    files = sorted(files)
    return files[0]


def find_closest_factors(n: int) -> tuple[int, int]:
    """
    Finds two integer factors of a number that are closest to each other.

    For example, for 100, it returns (10, 10).
    For 99, it returns (9, 11).

    Args:
        n: The integer number.

    Returns:
        A tuple containing the two closest factors.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")

    # Start searching from the integer part of the square root downwards
    start = int(math.sqrt(n))

    for i in range(start, 0, -1):
        if n % i == 0:
            # The first factor found will be the largest one less than or equal to the square root,
            # which guarantees the pair is the closest.
            factor1 = i
            factor2 = n // i
            return (factor1, factor2)

    # This part is technically unreachable for n > 0, as 1 is always a factor.
    return (1, n)
