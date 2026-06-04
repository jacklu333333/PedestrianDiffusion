import datetime
from pathlib import Path
from typing import Union

import ahrs
import numpy as np
import pandas as pd
import torch

REFERENCE = dict(
    RoNIN=dict(
        lat=49.2791,
        lon=-122.9202,
        date=datetime.datetime.strptime("2019-06-01-00-00-00", "%Y-%m-%d-%H-%M-%S"),
    ),
    RIDI=dict(
        lat=47.655548,
        lon=-122.303200,
        date=datetime.datetime.strptime("2018-06-01-00-00-00", "%Y-%m-%d-%H-%M-%S"),
    ),
    Oxford=dict(
        lat=51.7598,
        lon=-1.2586,
        date=datetime.datetime.strptime("2017-06-01-00-00-00", "%Y-%m-%d-%H-%M-%S"),
    ),
    TLIO=dict(
        lat=40.4433,
        lon=-79.9436,
        date=datetime.datetime.strptime("2023-06-01-00-00-00", "%Y-%m-%d-%H-%M-%S"),
    ),
)


def GravityRemoval(
    acc: Union[np.ndarray, torch.Tensor],
    location: str,
    date: Union[str, float, int, None] = None,
    rescale: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Gravity Removal from accelerometer data
    ---------------------------------------
    Input:
        acc: accelerometer data, shape (N, 4) x, y, z, norm or (N, 3) x, y, z

        location: location of the sensor
        date: date of the sensor
    Output:
        acc: accelerometer data after gravity removal
    """
    TENSOR = True
    # check torch or numpy
    if isinstance(acc, np.ndarray):
        TENSOR = False
        acc = torch.from_numpy(acc).float()
    else:
        acc = acc.clone().float()
    # check is it (N,4) if (N,3) add norm else raise error
    dim = acc.shape[1]
    if dim != 3 and dim != 4:
        raise ValueError("acc shape must be (N,3) or (N,4)")

    wgs = ahrs.utils.WGS()
    lat, lon = REFERENCE[location]["lat"], REFERENCE[location]["lon"]
    if date is None:
        date = REFERENCE[location]["date"]

    gravity = wgs.normal_gravity(lat)

    bgGravity_Z = gravity

    acc[:, 2] -= bgGravity_Z
    if dim == 4:
        acc[:, 3] = torch.linalg.norm(acc[:, :3], dim=1)

    if rescale:
        acc = acc / gravity

    if not TENSOR:
        acc = acc.numpy()
    return acc


def MagneticRemoval(
    mag: Union[np.ndarray, torch.Tensor],
    location: str,
    date: Union[str, float, int, None] = None,
    rescale: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Magnetic Removal from accelerometer data
    ---------------------------------------
    Input:
        mag: magnetometer data (N, 4) x, y, z, norm
        location: location of the sensor
        date: date of the sensor
    Output:
        mag: magnetometer data after magnetic removal
    """
    # check torch or numpy
    if isinstance(mag, np.ndarray):
        TENSOR = False
        mag = torch.from_numpy(mag).float()
    else:
        TENSOR = True
        mag = mag.clone().float()

    # check is it (N,4) if (N,3) add norm else raise error
    if mag.shape[1] == 3:
        mag = torch.cat([mag, torch.linalg.norm(mag, dim=1).unsqueeze(1)], dim=1)
    elif mag.shape[1] == 4:
        pass
    else:
        raise ValueError("mag shape must be (N,3) or (N,4)")

    wmm = ahrs.utils.WMM()
    lat, lon = REFERENCE[location]["lat"], REFERENCE[location]["lon"]
    if date is None:
        date = REFERENCE[location]["date"]

    wmm.magnetic_field(lat, lon, date=date)
    bgMag_X = wmm.X * 1e-5
    bgMag_Y = wmm.Y * 1e-5
    bgMag_Z = wmm.Z * 1e-5

    mag[:, 0] -= bgMag_X
    mag[:, 1] -= bgMag_Y
    mag[:, 2] -= bgMag_Z
    mag[:, 3] = torch.linalg.norm(mag[:, :3], dim=1)

    strength = wmm.F * 1e-5
    if rescale:
        mag = mag / strength

    if not TENSOR:
        mag = mag.numpy()
    return mag


def magRescale(
    mag: Union[np.ndarray, torch.Tensor],
    location: str,
    date: Union[str, float, int, None] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Magnetic Rescale from magnetometer data
    ---------------------------------------
    Input:
        mag: magnetometer data (N, 4) x, y, z, norm
        location: location of the sensor
        date: date of the sensor
    Output:
        mag: magnetometer data after magnetic rescale
    """
    # check torch or numpy
    if isinstance(mag, np.ndarray):
        TENSOR = False
        mag = torch.from_numpy(mag).float()
    else:
        TENSOR = True
        mag = mag.clone().float()

    wmm = ahrs.utils.WMM()
    lat, lon = REFERENCE[location]["lat"], REFERENCE[location]["lon"]
    if date is None:
        date = REFERENCE[location]["date"]

    wmm.magnetic_field(lat, lon, date=date)
    strength = wmm.F * 1e-5
    # print(f"strength: {strength:.2f}")
    mag = mag / strength

    if not TENSOR:
        mag = mag.numpy()
    return mag
