import datetime

import ahrs
import colored as cl
import numpy as np
import pandas as pd
import torch
from kornia.geometry.liegroup import So3
from kornia.geometry.quaternion import quaternion_to_rotation_matrix
from kornia.geometry.vector import Vector3
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as R

# print("import the right file !")


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# def rotateToWorldFrame(
#     acc: np.array,
#     gyr: np.array,
#     mag: np.array,
#     sample_rate: float,
#     rotation: np.array = None,
#     cutoff: float = 50.0,
#     name: str = "",
#     return_ori: bool = False,
# ) -> (np.array, np.array, np.array):
#     """
#     This function rotates the sensor frame to world frame using the Madgwick algorithm
#     Input:
#         acc: nx3 numpy array of accelerometer data
#         gyr: nx3 numpy array of gyroscope data
#         mag: nx3 numpy array of magnetometer data
#     Output:
#         acc: nx3 numpy array of accelerometer data in world frame
#         gyr: nx3 numpy array of gyroscope data in world frame
#         mag: nx3 numpy array of magnetometer data in world frame
#     """
#     # # Filter requirements.
#     # order = 4
#     # fs = 100.0  # sample rate, Hz
#     # # cutoff = 50  # desired cutoff frequency of the filter, Hz
#     # b, a = butter_lowpass(cutoff, fs, order)
#     # acc = butter_lowpass_filter(acc, cutoff, fs, order)
#     # gyr = butter_lowpass_filter(gyr, cutoff, fs, order)
#     # mag = butter_lowpass_filter(mag, cutoff, fs, order)

#     if rotation is not None:
#         # np.array in x y z w format
#         quad = rotation
#         r = R.from_quat(quad)
#         inv = r.inv()
#     else:
#         # the orientation is not provided
#         # Initialize the filter
#         filter = ahrs.filters.Madgwick(
#             acc=acc,
#             gyr=gyr,
#             mag=mag / 10,
#             frequency=sample_rate,
#             # gain=0.1,
#         )
#         quad = filter.Q
#         quad = np.concatenate((quad[:, 1:], quad[:, :1]), axis=1)
#         r = R.from_quat(quad)
#         inv = r.inv()

#     acc = r.apply(acc)
#     gyr = r.apply(gyr)
#     mag = r.apply(mag)
#     if return_ori:
#         return acc, gyr, mag, r.as_quat()
#     return acc, gyr, mag


def rotateToWorldFrame(
    acc: np.array,
    gyr: np.array,
    mag: np.array,
    sample_rate: float,
    rotation: np.array = None,
    cutoff: float = 50.0,
    name: str = "",
    return_ori: bool = False,
) -> (np.array, np.array, np.array):
    """
    Rotate sensor-frame data to world-frame. Uses Madgwick (CPU) to estimate
    orientation if not provided, then applies batched rotation on GPU via Kornia.
    """
    n = acc.shape[0]
    if rotation is not None:
        quad = np.asarray(rotation)
        if quad.ndim == 1:
            if quad.shape[-1] != 4:
                raise ValueError("rotation must be a 4D quaternion in xyzw format.")
            quad = np.broadcast_to(quad, (n, 4))
        elif quad.shape == (n, 4):
            pass
        else:
            raise ValueError(
                f"rotation shape {quad.shape} is incompatible with data length {n}."
            )
        # Ensure xyzw for downstream (SciPy uses xyzw; we keep consistency)
    else:
        # Estimate orientation with Madgwick (CPU, numpy)
        filter = ahrs.filters.Madgwick(
            acc=acc,
            gyr=gyr,
            mag=mag / 10,
            frequency=sample_rate,
        )
        # ahrs returns quaternions as wxyz; convert to xyzw to be consistent
        quad = filter.Q
        quad = np.concatenate((quad[:, 1:], quad[:, :1]), axis=1)

    # Select device and move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    acc_t = torch.as_tensor(acc, dtype=dtype, device=device)
    gyr_t = torch.as_tensor(gyr, dtype=dtype, device=device)
    mag_t = torch.as_tensor(mag, dtype=dtype, device=device)

    q_xyzw_t = torch.as_tensor(quad, dtype=dtype, device=device)
    # Normalize quaternions
    q_xyzw_t = q_xyzw_t / torch.clamp(q_xyzw_t.norm(dim=-1, keepdim=True), min=1e-8)

    # Kornia quaternion_to_rotation_matrix expects wxyz; reorder from xyzw -> wxyz
    q_wxyz_t = torch.cat([q_xyzw_t[:, 3:4], q_xyzw_t[:, :3]], dim=-1)
    R_batched = quaternion_to_rotation_matrix(q_wxyz_t)  # [N, 3, 3]

    # Apply rotation in batch: world_vec = R * sensor_vec
    acc_rot = torch.bmm(R_batched, acc_t.unsqueeze(-1)).squeeze(-1)
    gyr_rot = torch.bmm(R_batched, gyr_t.unsqueeze(-1)).squeeze(-1)
    mag_rot = torch.bmm(R_batched, mag_t.unsqueeze(-1)).squeeze(-1)

    if return_ori:
        # Return normalized quaternions in xyzw format
        return (
            acc_rot.detach().cpu().numpy(),
            gyr_rot.detach().cpu().numpy(),
            mag_rot.detach().cpu().numpy(),
            q_xyzw_t.detach().cpu().numpy(),
        )

    return (
        acc_rot.detach().cpu().numpy(),
        gyr_rot.detach().cpu().numpy(),
        mag_rot.detach().cpu().numpy(),
    )


def coordinate_exchange(datas: list, _from: str, _to: str):
    """
    This function exchange the coordinate from one to another
    Input:
        datas: nx3 numpy array of data
        _from: the coordinate system to be converted from
        _to: the coordinate system to be converted to
    Output:
        datas: nx3 numpy array of data in the new coordinate system
    """
    result = []
    for data in datas:
        if _from == "ENU" and _to == "NED":
            data = [data[:, 1], data[:, 0], -data[:, 2]]
        elif _from == "NED" and _to == "ENU":
            if np.mean(data[2]) < 0:
                print(
                    cl.Fore.red
                    + "- The data might not be in NED coordinate system"
                    + cl.Style.reset
                )
            data = [data[:, 1], data[:, 0], -data[:, 2]]
        # elif _from == "ENU" and _to == "NWU":
        #     data = [data[1],
        # elif _from == "NWU" and _to == "ENU":
        #     pass
        # elif _from == "NED" and _to == "NWU":
        #     pass
        # elif _from == "NWU" and _to == "NED":
        #     pass
        else:
            raise ValueError(
                f"The coordinate system is not supported from: {_from} to {_to}"
            )
        result.append(np.array(data).swapaxes(0, 1))
    return result
