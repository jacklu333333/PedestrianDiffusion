import json
import math

import colored as cl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.spatial.transform import Rotation


class mModule(nn.Module):
    """
    Base class for all the transformation modules
    1. save the config as class attribute
    # 2. return the input data if the probability is less than the threshold
    """

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        for k, v in config.items():
            setattr(self, k, v)

    # def forward(self, data):
    #     # convert to tensor first
    #     p = torch.rand(1).item()
    #     if p >= self.probability:
    #         return data
    def should_apply(self):
        return torch.rand(1).item() < self.probability


class rotationNoise(mModule):
    """
    input: torch.Tensor of shape ( 9, time)
    split the input into 3 parts, each part is a 3x100 matrix
    randomly rotate all three 3x100 matrix
    ----------------------------------------------
    output: torch.Tensor of shape ( 9, time)

    """

    def __init__(self, config):
        super(rotationNoise, self).__init__(config)

    def genRotationMatrix(self, yaw, pitch, roll):
        """
        yaw:   float, unit: radians
        pitch: float, unit: radians
        roll:  float, unit: radians
        """
        # yaw
        Rz = torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), 0],
                [torch.sin(yaw), torch.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        # pitch
        Ry = torch.tensor(
            [
                [torch.cos(pitch), 0, torch.sin(pitch)],
                [0, 1, 0],
                [-torch.sin(pitch), 0, torch.cos(pitch)],
            ]
        )
        # roll
        Rx = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(roll), -torch.sin(roll)],
                [0, torch.sin(roll), torch.cos(roll)],
            ]
        )
        return Rz @ Ry @ Rx

    def forward_with_parameters(self, data, parameter):
        # p = torch.rand(1).item()
        # if p >= self.probability:
        #     return data
        # -----------------------------------------
        x, y = data

        if self.mode == "random":
            # randomly rotate all three 3x100 matrix
            yaw = parameter[0] * 2 * torch.pi
            pitch = parameter[1] * 2 * torch.pi
            roll = parameter[2] * 2 * torch.pi
        elif self.mode == "axis":
            raise NotImplementedError(
                "axis mode is not implemented for rotationNoise with parameters"
            )
        elif self.mode == "XY":
            yaw = parameter[0] * 2 * torch.pi
            # yaw = torch.randint(0, 72, (1,)) / 180.0 * torch.pi  # every 5 degree
            pitch = torch.zeros(1)
            roll = torch.zeros(1)
        elif self.mode == "wabble":
            raise NotImplementedError(
                "wabble mode is not implemented for rotationNoise with parameters"
            )
        else:
            raise ValueError("mode should be random, axis, XY, wabble")

        R = self.genRotationMatrix(yaw, pitch, roll).to(x.device)

        # split the input into 3 parts
        # acc = x[:3]
        # gyr = x[3:6]
        # mag = x[6:]
        # acc = R @ acc
        # gyr = R @ gyr
        # mag = R @ mag
        observation = []
        for i in range(0, x.shape[0], 3):
            observation.append(R @ x[i : i + 3])
        observation = torch.cat(observation, dim=0)

        if self.label_transform:
            newVel = y.clone().detach()
            for i in range(0, y.shape[0], 3):
                newVel[i : i + 3] = R @ y[i : i + 3]
            # newVel = R @ y
        else:
            newVel = y.clone()

        assert (
            observation.shape == x.shape
        ), f"observation shape {observation.shape} does not match x shape {x.shape}"
        assert (
            newVel.shape == y.shape
        ), f"newVel shape {newVel.shape} does not match y shape {y.shape}"
        # return torch.cat([acc, gyr, mag], dim=0), newVel
        return observation, newVel

    def forward(self, data):
        if not self.should_apply():
            return data
        # -----------------------------------------
        x, y = data
        assert x.dim() == 2, "Input x should be a 2D tensor"
        assert y.dim() == 2, "Input y should be a 2D tensor"
        # assert (
        #     x.shape[0] == y.shape[0]
        # ), "Input x and y should have the same number of rows"
        assert (
            x.shape[0] % 3 == 0
        ), "Input x should have a number of columns that is a multiple of 3"

        if self.mode == "random":
            # randomly rotate all three 3x100 matrix
            yaw = torch.rand(1) * 2 * torch.pi
            pitch = torch.rand(1) * 2 * torch.pi
            roll = torch.rand(1) * 2 * torch.pi
        elif self.mode == "axis":
            yaw = torch.randint(0, 4, (1,)) * torch.pi / 2
            pitch = torch.randint(0, 4, (1,)) * torch.pi / 2
            roll = torch.randint(0, 4, (1,)) * torch.pi / 2
        elif self.mode == "XY":
            yaw = torch.rand(1) * 2 * torch.pi
            # yaw = torch.randint(0, 72, (1,)) / 180.0 * torch.pi  # every 5 degree
            pitch = torch.zeros(1)
            roll = torch.zeros(1)
        elif self.mode == "wabble":
            # yaw = 5 / 180 * (torch.randn(size=(1,)) - 0.5) * torch.pi / 2
            # pitch = 5 / 180 * (torch.randn(size=(1,)) - 0.5) * torch.pi / 2
            # roll = 5 / 180 * (torch.randn(size=(1,)) - 0.5) * torch.pi / 2
            mean = torch.zeros(1)
            std = torch.ones(1) / 3
            yaw = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
            pitch = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
            roll = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
        elif self.mode == "wabble_XY":
            # yaw = 5 / 180 * (torch.randn(size=(1,)) - 0.5) * torch.pi / 2
            # pitch = 5 / 180 * (torch.randn(size=(1,)) - 0.5) * torch.pi / 2
            # roll = 5 / 180 * (torch.randn(size=(1,)) - 0.5) * torch.pi / 2
            mean = torch.zeros(1)
            std = torch.ones(1) / 3
            yaw = torch.zeros(1)
            pitch = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
            roll = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
        else:
            raise ValueError("mode should be random, axis, XY, wabble")

        R = self.genRotationMatrix(yaw, pitch, roll).to(x.device)

        # split the input into 3 parts
        # acc = x[:3]
        # gyr = x[3:6]
        # mag = x[6:]
        # acc = R @ acc
        # gyr = R @ gyr
        # mag = R @ mag
        observation = []
        for i in range(0, x.shape[0], 3):
            observation.append(R @ x[i : i + 3])
        observation = torch.cat(observation, dim=0)

        if self.label_transform:
            newVel = y.clone().detach()
            for i in range(0, y.shape[0], 3):
                newVel[i : i + 3] = R @ y[i : i + 3]
            # newVel = R @ y
        else:
            newVel = y.clone()

        assert (
            observation.shape == x.shape
        ), f"observation shape {observation.shape} does not match x shape {x.shape}"
        assert (
            newVel.shape == y.shape
        ), f"newVel shape {newVel.shape} does not match y shape {y.shape}"
        # return torch.cat([acc, gyr, mag], dim=0), newVel
        return observation, newVel


# ---------------------------------------------------------------------------------------------- not test yet
class gaussianNoise(mModule):
    """
    input: torch.Tensor of shape ( 9, time)
    add guassian noise to the input
    ----------------------------------------------
    output: torch.Tensor of shape ( 9, time)

    """

    def __init__(self, config):
        super(gaussianNoise, self).__init__(config)

    def forward(self, data):
        if not self.should_apply():
            return data

        x, y = data
        # split the input into 3 parts
        acc = x[:3]
        gyr = x[3:6]
        # mag = x[6:]

        # add guassian noise to all three 3x100 matrix
        # acc_std = torch.ones_like(acc) * self.accNoise
        # acc_std = acc_std.cumsum(dim=-1)
        acc = torch.normal(mean=acc, std=self.accNoise)
        gyr = torch.normal(mean=gyr, std=self.gyrNoise)
        # mag = torch.normal(mean=mag, std=self.magNoise)

        # return torch.cat([acc, gyr, mag], dim=0), y
        return torch.cat([acc, gyr], dim=0), y


class scaleNoise(mModule):
    """
    input: torch.Tensor of shape ( 9, time)
    add scale noise to the input
    ----------------------------------------------
    output: torch.Tensor of shape ( 9, time)

    """

    def __init__(self, config):
        super(scaleNoise, self).__init__(config)

    def forward(self, data):
        if not self.should_apply():
            return data

        # -----------------------------------------

        x, y = data
        # split the input into 3 parts
        acc = x[:3]
        gyr = x[3:6]
        # mag = x[6:]

        # add scale noise to all three 3x100 matrix
        acc *= (
            torch.normal(
                mean=torch.tensor([1.0] * 3), std=torch.tensor([self.accNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )

        gyr *= (
            torch.normal(
                mean=torch.tensor([1.0] * 3), std=torch.tensor([self.gyrNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )
        # mag *= (
        #     torch.normal(
        #         mean=torch.tensor([1.0] * 3),
        #         std=torch.tensor([self.magNoise / 3] * 3),
        #     )
        #     .reshape(3, 1)
        #     .to(x.device)
        # )

        # acc *= torch.rand(1).to(x.device) * self.accNoise + (1 - self.accNoise)
        # gyr *= torch.rand(1).to(x.device) * self.gyrNoise + (1 - self.gyrNoise)
        # preVel *= torch.rand(1).to(x.device) * self.magNoise + (1 - self.magNoise)

        # return torch.cat([acc, gyr, mag], dim=0), y
        return torch.cat([acc, gyr], dim=0), y


class shiftNoise(mModule):
    """
    input: torch.Tensor of shape ( 9, time)
    add shift noise to the input
    ----------------------------------------------
    output: torch.Tensor of shape ( 9, time)

    """

    def __init__(self, config):
        super(shiftNoise, self).__init__(config)

    def forward(self, data):
        if not self.should_apply():
            return data
        # -----------------------------------------

        x, y = data
        # split the input into 3 parts
        acc = x[:3]
        gyr = x[3:6]
        # mag = x[6:]

        # add shift noise to all three 3x100 matrix
        acc += (
            torch.normal(
                mean=torch.tensor([0.0] * 3), std=torch.tensor([self.accNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )
        gyr += (
            torch.normal(
                mean=torch.tensor([0.0] * 3), std=torch.tensor([self.gyrNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )
        # mag += (
        #     torch.normal(
        #         mean=torch.tensor([0.0] * 3),
        #         std=torch.tensor([self.magNoise / 3] * 3),
        #     )
        #     .reshape(3, 1)
        #     .to(x.device)
        # )

        # acc += torch.rand(1).to(x.device) * self.accNoise
        # gyr += torch.rand(1).to(x.device) * self.gyrNoise
        # preVel += torch.rand(1).to(x.device) * self.magNoise

        # return torch.cat([acc, gyr, mag], dim=0), y
        return torch.cat([acc, gyr], dim=0), y


class axisMasking(mModule):
    """
    input: torch.Tensor of shape ( 9, time)
    randomly mask one axis of the input
    ----------------------------------------------
    output: torch.Tensor of shape ( 9, time)

    """

    def __init__(self, config):
        super(axisMasking, self).__init__(config)

    def forward(self, data):
        if not self.should_apply():
            return data
        # -----------------------------------------

        x, y = data
        # split the input into 3 parts
        acc = x[:3].clone()
        gyr = x[3:6].clone()
        mag = x[6:].clone()

        # randomly mask one axis of all three 3x100 matrix
        mask_number = torch.randint(0, self.max_channel + 1, (1,)).item()
        mask = torch.randperm(3)[:mask_number]
        acc[mask] = torch.zeros_like(acc[mask])
        gyr[mask] = torch.zeros_like(gyr[mask])
        mag[mask] = torch.zeros_like(mag[mask])

        newVel = y.clone()
        newVel[mask] = torch.zeros_like(newVel[mask])

        return torch.cat([acc, gyr, mag], dim=0), newVel
        # return torch.cat([acc, gyr], dim=0), newVel


class speedMasking(mModule):
    """
    input: torch.Tensor of shape ( 9, time)
    mask the speed of the input
    ----------------------------------------------
    output: torch.Tensor of shape ( 9, time)
    """

    def __init__(self, config):
        super(speedMasking, self).__init__(config)

    def masker(self, y):
        # find the first number reach speed threshold 0.1 and mask everything before it
        mask = torch.where(torch.norm(y, dim=0) > self.config["threshold"])
        # if mask is empty, mask everything
        if len(mask[0]) == 0:
            mask = y.shape[1]
        else:
            mask = mask[0][0]

        maskVel = y.clone()
        maskVel[:, :mask] = torch.zeros_like(maskVel[:, :mask])

        return maskVel

    def forward(self, data):
        if not self.should_apply():
            return data
        # -----------------------------------------

        x, y = data
        newX = x.clone()
        newX[6:] = self.masker(newX[6:])
        newY = self.masker(y)

        return newX, newY


class keepSensorData(mModule):
    def __init__(self, config):
        super(keepSensorData, self).__init__(config)
        self.keepAcc = "acc" in self.config["keepSensor"]
        self.keepGyr = "gyr" in self.config["keepSensor"]
        self.keepMag = "Mag" in self.config["keepSensor"]

    def forward(self, data):
        x, y = data
        acc = x[0][:3]
        gyr = x[0][3:6]
        mag = x[0][6:]

        if not self.keepAcc:
            acc = torch.zeros_like(acc)
        if not self.keepGyr:
            gyr = torch.zeros_like(gyr)
        if not self.keepMag:
            mag = torch.zeros_like(mag)

        newX = torch.cat([acc, gyr, mag], dim=0)

        return newX, y


class trnasformBatch(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        # new_param = self.config["gaussianNoise"]
        # new_param.update({"probability": 0.01})
        self.transform = nn.Sequential(
            # gaussianNoise(new_param),
            scaleNoise(self.config["scaleNoise"]),
            shiftNoise(self.config["shiftNoise"]),
            gaussianNoise(self.config["gaussianNoise"]),
            rotationNoise(self.config["rotationNoise"]),
            axisMasking(self.config["axisMasking"]),
        )

    def forward(self, batch):
        x, y = batch
        result_x = x.clone()
        result_y = y.clone()

        for i in range(x.shape[0]):
            result_x[i], result_y[i] = self.transform((result_x[i], result_y[i]))

        return (result_x, result_y)


class IMUToYUV(nn.Module):
    """
    Convert IMU data from XYZ to YUV color space.

    Purpose:
    - To transform tri-axis IMU data (accelerometer and gyroscope) into YUV color space for further processing.

    Functions:
    - xyz_to_yuv: Converts a single tri-axis measurement from XYZ to YUV.
    - forward: Applies the xyz_to_yuv conversion to the entire batch of IMU data.

    Input:
    - x: torch.Tensor of shape (batch_size, 6, time), where the first 3 channels are accelerometer data and the next 3 channels are gyroscope data.

    Output:
    - YUV: torch.Tensor of shape (batch_size, 6, time), where the first 3 channels are YUV-transformed accelerometer data and the next 3 channels are YUV-transformed gyroscope data.

    Value Range:
    - Input XYZ values: [-1,1]
    - Output YUV values: [0, 1].
    """

    def __init__(self):
        super(IMUToYUV, self).__init__()
        self.esp = 1e-6

    def xyz_to_yuv(self, tri_axis_measurement):
        x, y, z = (
            tri_axis_measurement[:, 0],
            tri_axis_measurement[:, 1],
            tri_axis_measurement[:, 2],
        )
        # add esp to where x is zeros to avoid division by zero
        # temp = torch.ones_like(x)
        # temp[x.abs() < 0] = -1
        Y = torch.sqrt(x**2 + y**2 + z**2)
        U = torch.atan2(y, z)
        V = torch.atan2(x, torch.sqrt(z**2 + y**2))

        # Y = Y / math.sqrt(3)
        # U = ((U / math.pi) + 1) / 2
        # V = ((V / math.pi) + 1) / 2

        assert torch.isfinite(Y).all(), "Y is not finite"
        assert torch.isfinite(U).all(), "U is not finite"
        assert torch.isfinite(V).all(), "V is not finite"

        YUV = torch.stack((Y, U, V), dim=1)
        return YUV

    def forward(self, x):
        # Convert accelerometer data to YUV
        acc_YUV = self.xyz_to_yuv(x[:, 0:3, :])

        # Convert gyroscope data to YUV
        gyr_YUV = self.xyz_to_yuv(x[:, 3:6, :])

        # merge at sedondary dimension
        YUV = torch.cat((acc_YUV, gyr_YUV), dim=1)

        return YUV


class YUVToIMU(nn.Module):
    """
    Convert IMU data from YUV color space back to XYZ.

    Purpose:
    - To transform YUV color space data back into tri-axis IMU data (accelerometer and gyroscope).

    Functions:
    - yuv_to_xyz: Converts a single YUV measurement back to XYZ.
    - forward: Applies the yuv_to_xyz conversion to the entire batch of YUV data.

    Input:
    - yuv: torch.Tensor of shape (batch_size, 6, time), where the first 3 channels are YUV-transformed accelerometer data and the next 3 channels are YUV-transformed gyroscope data.

    Output:
    - imu: torch.Tensor of shape (batch_size, 6, time), where the first 3 channels are accelerometer data and the next 3 channels are gyroscope data.

    Value Range:
    - Input YUV values:  [0, 1].
    - Output XYZ values: [-1,1]
    """

    def __init__(self):
        super(YUVToIMU, self).__init__()
        self.esp = 1e-6

    def yuv_to_xyz(self, yuv_measurement):
        Y, U, V = (
            yuv_measurement[:, 0],
            yuv_measurement[:, 1],
            yuv_measurement[:, 2],
        )
        # Y = Y * math.sqrt(3)
        # U = (U * 2 - 1) * math.pi
        # V = (V * 2 - 1) * math.pi

        # temp = torch.ones_like(Y)
        # temp[Y < 0] = -1

        x = Y * torch.sin(V)
        y = Y * torch.cos(V) * torch.sin(U)
        z = Y * torch.cos(V) * torch.cos(U)

        # check nan in XYZ
        assert torch.isfinite(x).all(), "nan in x"
        assert torch.isfinite(y).all(), "nan in y"
        assert torch.isfinite(z).all(), "nan in z"

        XYZ = torch.stack((x, y, z), dim=1)

        return XYZ

    def forward(self, yuv):
        # Convert YUV to accelerometer data
        acc = self.yuv_to_xyz(yuv[:, 0:3, :])

        # Convert YUV to gyroscope data
        gyr = self.yuv_to_xyz(yuv[:, 3:6, :])

        # Stack the results back to (batch_size, 6, 512)
        imu = torch.cat((acc, gyr), dim=1)

        return imu


class SensorNormalizer(nn.Module):
    def __init__(self, mode="mean_std"):
        super(SensorNormalizer, self).__init__()
        assert mode in [
            "mean_std",
            "min_max",
            "max",
        ], "Mode must be 'mean_std', 'min_max' or 'max'"
        self.mode = mode

    def forward(self, x, mask):
        # Assuming x is of shape (batch, 6, 512)
        batch_size, _, time_series = x.shape

        # Reshape to combine the axes
        x = x.view(batch_size, 2, 3, time_series)  # Shape: (batch, 2, 3, 512)

        if self.mode == "mean_std":
            # Normalize each sensor data separately by mean and std
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True)
            x_normalized = (x - mean) / std
        elif self.mode == "min_max":
            # Normalize each sensor data separately by min and max
            min_val = x.min(dim=(2, 3), keepdim=True)[0]
            max_val = x.max(dim=(2, 3), keepdim=True)[0]
            x_normalized = (x - min_val) / (max_val - min_val)
        elif self.mode == "max":
            # Normalize each sensor data separately by min and max
            max_val = x.abs().max(dim=(2, 3), keepdim=True)[0]
            x_normalized = x / max_val
        else:
            raise ValueError("Mode must be 'mean_std', 'min_max' or 'max'")

        # Reshape back to original shape
        x_normalized = x_normalized.view(batch_size, 6, time_series)
        x_normalized = x_normalized.masked_fill(
            mask.unsqueeze(1).expand(-1, x_normalized.shape[1], -1), 0
        )

        return x_normalized


class IMUToIntensityQuaternion(nn.Module):
    """
    Convert IMU data from XYZ to intensity + quaternion (XYZW) space.

    Input:
    - x: torch.Tensor of shape (batch_size, 6, 512), where the first 3 channels are accelerometer data and the next 3 channels are gyroscope data.

    Output:
    - intensity_quaternion: torch.Tensor of shape (batch_size, 8, 512), where the first 2 channels are intensity and the next 6 channels are quaternions (XYZW).
    """

    def __init__(self):
        super(IMUToIntensityQuaternion, self).__init__()

    def xyz_to_intensity_quaternion(self, tri_axis_measurement):
        x, y, z = (
            tri_axis_measurement[:, 0],
            tri_axis_measurement[:, 1],
            tri_axis_measurement[:, 2],
        )
        intensity = tri_axis_measurement.norm(dim=1, keepdim=True)
        quaternion = tri_axis_measurement
        quaternion = F.normalize(quaternion, p=2, dim=1)
        return torch.cat((intensity, quaternion), dim=1)

    def forward(self, x):
        acc_intensity_quaternion = self.xyz_to_intensity_quaternion(x[:, 0:3, :])
        gyr_intensity_quaternion = self.xyz_to_intensity_quaternion(x[:, 3:6, :])
        intensity_quaternion = torch.cat(
            (acc_intensity_quaternion, gyr_intensity_quaternion), dim=1
        )
        return intensity_quaternion


class IntensityQuaternionToIMU(nn.Module):
    """
    Convert IMU data from intensity + quaternion (XYZW) space back to XYZ.

    Input:
    - x: torch.Tensor of shape (batch_size, 8, 512), where the first 2 channels are intensity and the next 6 channels are quaternions (XYZW).

    Output:
    - xyz: torch.Tensor of shape (batch_size, 6, 512), where the first 3 channels are accelerometer data and the next 3 channels are gyroscope data.
    """

    def __init__(self):
        super(IntensityQuaternionToIMU, self).__init__()

    def intensity_quaternion_to_xyz(self, intensity_quaternion):
        intensity = intensity_quaternion[:, 0]
        quaternion = intensity_quaternion[:, 1:4]
        xyz = quaternion * intensity.unsqueeze(1)
        return xyz

    def forward(self, x):
        acc_xyz = self.intensity_quaternion_to_xyz(x[:, 0:4, :])
        gyr_xyz = self.intensity_quaternion_to_xyz(x[:, 4:8, :])
        xyz = torch.cat((acc_xyz, gyr_xyz), dim=1)
        return xyz


class TimeNormalization(nn.Module):
    def __init__(self, path: str):
        super(TimeNormalization, self).__init__()
        self.df = pd.read_csv(path)
        self.imu_mean = (
            torch.from_numpy(self.df[["acc_mean", "gyr_mean"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )
        self.imu_std = (
            torch.from_numpy(self.df[["acc_std", "gyr_std"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )
        self.label_mean = (
            torch.from_numpy(self.df[["l_acc_mean", "l_gyr_mean"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )
        self.label_std = (
            torch.from_numpy(self.df[["l_acc_std", "l_gyr_std"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )

        # replace std 0 with 1
        self.imu_std[self.imu_std == 0] = 1
        self.label_std[self.label_std == 0] = 1

        # self._scale = 5.485
        self._scale = 1

    def forward(self, x, y, L):
        Batch, Channel, Time = x.shape

        x = (x - self.imu_mean.to(x.device)) / self.imu_std.to(x.device) / self._scale
        y = (
            (y - self.label_mean.to(y.device))
            / self.label_std.to(y.device)
            / self._scale
        )

        mask = torch.arange(Time).expand(Batch, Time).to(L.device)
        mask = mask < L.unsqueeze(1)
        mask = torch.logical_not(mask)
        x = x.masked_fill(mask.unsqueeze(1).expand(-1, Channel, -1), 0)
        y = y.masked_fill(mask.unsqueeze(1).expand(-1, Channel, -1), 0)

        return x.float().clamp(-1, 1), y.float().clamp(-1, 1)


class TimeDenormalization(nn.Module):
    def __init__(self, path: str):
        super(TimeDenormalization, self).__init__()
        self.df = pd.read_csv(path)
        self.imu_mean = (
            torch.from_numpy(self.df[["acc_mean", "gyr_mean"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )
        self.imu_std = (
            torch.from_numpy(self.df[["acc_std", "gyr_std"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )
        self.label_mean = (
            torch.from_numpy(self.df[["l_acc_mean", "l_gyr_mean"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )
        self.label_std = (
            torch.from_numpy(self.df[["l_acc_std", "l_gyr_std"]].to_numpy())
            .view(2, 1)
            .expand(-1, 3)
            .reshape(-1, 6, 1)
        )

        # replace std 0 with 1
        self.imu_std[self.imu_std == 0] = 1
        self.label_std[self.label_std == 0] = 1
        # self._scale = 5.485
        self._scale = 1

    def forward(self, x, y, L):
        Batch, Channel, Time = x.shape
        x = x.clip(-1, 1)

        x = x * self._scale * self.imu_std.to(x.device) + self.imu_mean.to(x.device)
        y = y * self._scale * self.label_std.to(y.device) + self.label_mean.to(y.device)

        mask = torch.arange(Time).expand(Batch, Time).to(L.device)
        mask = mask < L.unsqueeze(1)
        mask = torch.logical_not(mask)

        x = x.masked_fill(mask.unsqueeze(1).expand(-1, Channel, -1), 0)
        y = y.masked_fill(mask.unsqueeze(1).expand(-1, Channel, -1), 0)

        return x.float(), y.float()


class FreqNormalization(nn.Module):
    def __init__(self, path: str):
        super(FreqNormalization, self).__init__()
        self.df = pd.read_csv(path)
        self.imu_acc_mean = torch.from_numpy(self.df["imu_acc_mean"].to_numpy()).view(
            -1, 1
        )
        self.imu_acc_std = torch.from_numpy(self.df["imu_acc_std"].to_numpy()).view(
            -1, 1
        )
        self.label_acc_mean = torch.from_numpy(
            self.df["label_acc_mean"].to_numpy()
        ).view(-1, 1)
        self.label_acc_std = torch.from_numpy(self.df["label_acc_std"].to_numpy()).view(
            -1, 1
        )

        self.imu_gyr_mean = torch.from_numpy(self.df["imu_gyr_mean"].to_numpy()).view(
            -1, 1
        )
        self.imu_gyr_std = torch.from_numpy(self.df["imu_gyr_std"].to_numpy()).view(
            -1, 1
        )
        self.label_gyr_mean = torch.from_numpy(
            self.df["label_gyr_mean"].to_numpy()
        ).view(-1, 1)
        self.label_gyr_std = torch.from_numpy(self.df["label_gyr_std"].to_numpy()).view(
            -1, 1
        )

        # replace std 0 with 1
        self.imu_acc_std[self.imu_acc_std == 0] = 1
        self.label_acc_std[self.label_acc_std == 0] = 1

        self.imu_gyr_std[self.imu_gyr_std == 0] = 1
        self.label_gyr_std[self.label_gyr_std == 0] = 1

        # self._scale = 5.485
        self._scale = 1

    def forward(self, x, y):
        Batch, Channel, Freq, Time, Image = x.shape

        x_acc = rearrange(x[:, : Channel // 2], "b c f t i -> (c f i) (b t)")
        x_gyr = rearrange(x[:, Channel // 2 :], "b c f t i -> (c f i) (b t)")

        x_acc = (
            (x_acc - self.imu_acc_mean.to(x.device))
            / self.imu_acc_std.to(x.device)
            / self._scale
        )
        x_gyr = (
            (x_gyr - self.imu_gyr_mean.to(x.device))
            / self.imu_gyr_std.to(x.device)
            / self._scale
        )
        x_acc = rearrange(
            x_acc,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )
        x_gyr = rearrange(
            x_gyr,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )

        x = torch.cat([x_acc, x_gyr], dim=1).clamp(-1, 1)

        y_acc = rearrange(y[:, : Channel // 2], "b c f t i -> (c f i) (b t)")
        y_gyr = rearrange(y[:, Channel // 2 :], "b c f t i -> (c f i) (b t)")
        y_acc = (
            (y_acc - self.label_acc_mean.to(y.device))
            / self.label_acc_std.to(y.device)
            / self._scale
        )
        y_gyr = (
            (y_gyr - self.label_gyr_mean.to(y.device))
            / self.label_gyr_std.to(y.device)
            / self._scale
        )
        y_acc = rearrange(
            y_acc,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )
        y_gyr = rearrange(
            y_gyr,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )
        y = torch.cat([y_acc, y_gyr], dim=1).clamp(-1, 1)

        return x.float(), y.float()


class FreqDenormalization(nn.Module):
    def __init__(self, path: str):
        super(FreqDenormalization, self).__init__()
        self.df = pd.read_csv(path)
        self.imu_acc_mean = torch.from_numpy(self.df["imu_acc_mean"].to_numpy()).view(
            -1, 1
        )
        self.imu_acc_std = torch.from_numpy(self.df["imu_acc_std"].to_numpy()).view(
            -1, 1
        )
        self.label_acc_mean = torch.from_numpy(
            self.df["label_acc_mean"].to_numpy()
        ).view(-1, 1)
        self.label_acc_std = torch.from_numpy(self.df["label_acc_std"].to_numpy()).view(
            -1, 1
        )

        self.imu_gyr_mean = torch.from_numpy(self.df["imu_gyr_mean"].to_numpy()).view(
            -1, 1
        )
        self.imu_gyr_std = torch.from_numpy(self.df["imu_gyr_std"].to_numpy()).view(
            -1, 1
        )
        self.label_gyr_mean = torch.from_numpy(
            self.df["label_gyr_mean"].to_numpy()
        ).view(-1, 1)
        self.label_gyr_std = torch.from_numpy(self.df["label_gyr_std"].to_numpy()).view(
            -1, 1
        )

        # replace std 0 with 1
        self.imu_acc_std[self.imu_acc_std == 0] = 1
        self.label_acc_std[self.label_acc_std == 0] = 1

        self.imu_gyr_std[self.imu_gyr_std == 0] = 1
        self.label_gyr_std[self.label_gyr_std == 0] = 1

        # self._scale = 5.485
        self._scale = 1

    def forward(self, x, y):
        Batch, Channel, Freq, Time, Image = x.shape
        x = x.clip(-1, 1)

        x_acc = rearrange(x[:, : Channel // 2], "b c f t i -> (c f i) (b t)")
        x_gyr = rearrange(x[:, Channel // 2 :], "b c f t i -> (c f i) (b t)")

        x_acc = x_acc * self._scale * self.imu_acc_std.to(
            x.device
        ) + self.imu_acc_mean.to(x.device)
        x_gyr = x_gyr * self._scale * self.imu_gyr_std.to(
            x.device
        ) + self.imu_gyr_mean.to(x.device)

        x_acc = rearrange(
            x_acc,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )
        x_gyr = rearrange(
            x_gyr,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )
        x = torch.cat([x_acc, x_gyr], dim=1)

        y_acc = rearrange(y[:, : Channel // 2], "b c f t i -> (c f i) (b t)")
        y_gyr = rearrange(y[:, Channel // 2 :], "b c f t i -> (c f i) (b t)")
        y_acc = y_acc * self._scale * self.label_acc_std.to(
            y.device
        ) + self.label_acc_mean.to(y.device)
        y_gyr = y_gyr * self._scale * self.label_gyr_std.to(
            y.device
        ) + self.label_gyr_mean.to(y.device)
        y_acc = rearrange(
            y_acc,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )
        y_gyr = rearrange(
            y_gyr,
            "(c f i) (b t) -> b c f t i",
            b=Batch,
            c=Channel // 2,
            f=Freq,
            t=Time,
            i=Image,
        )
        y = torch.cat([y_acc, y_gyr], dim=1)
        return x.float(), y.float()


class TFConverterBase(nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = 14
        self.win_length = n_fft
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)


class Time2Frequency(TFConverterBase):
    def __init__(self, n_fft):
        super(Time2Frequency, self).__init__(n_fft)

    def forward(self, waveforms):
        waveformX, waveformY = waveforms

        # Compute the Short-Time Fourier Transform (STFT) for each channel
        freqDomainX = torch.stft(
            waveformX,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
            # center=False,
            # normalized=True,
        )
        freqDomainY = torch.stft(
            waveformY,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
            # center=False,
            # normalized=True,
        )

        freqDomainX = torch.view_as_real(freqDomainX)
        freqDomainY = torch.view_as_real(freqDomainY)

        return freqDomainX, freqDomainY


class batchTimeToFrequency(TFConverterBase):
    def __init__(self, n_fft):
        super(batchTimeToFrequency, self).__init__(n_fft)

    def stft_fn(self, waveform):
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
        )

    def forward(self, waveform):
        shapes = list(waveform.shape)
        timeseries = shapes[-1]

        # Compute the Short-Time Fourier Transform (STFT) for each channel
        freqDomain = torch.vmap(torch.vmap(self.stft_fn))(waveform)  # → (B, C, F, T)

        merged = torch.view_as_real(freqDomain)
        assert torch.isfinite(merged).all(), "NaN or Inf in the frequency domain"

        return merged


class batchFrequencyToTime(TFConverterBase):
    def __init__(self, output_length, n_fft):
        super(batchFrequencyToTime, self).__init__(n_fft)
        self.output_length = output_length

    def istft_fn(self, x):
        return torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=self.output_length,
        )

    def forward(self, merged):
        shapes = merged.shape
        freq, time = shapes[-3], shapes[-2]
        # Split real and imaginary parts from the extra dimension
        freqDomain = torch.view_as_complex(merged.contiguous())
        waveform = torch.vmap(torch.vmap(self.istft_fn))(freqDomain)  # → (B, C, time)

        ind = waveform == torch.inf
        if ind.any():
            print(cl.Fore.RED + "Inf exists in waveform" + cl.Style.reset)
            # waveform[ind] = torch.finfo(torch.float32).max
            # find the negative inf and replace with torch.finfo(torch.float32).min

        ind = waveform == -torch.inf
        if ind.any():
            print(cl.Fore.RED + "-Inf exists in waveform" + cl.Style.reset)
            # waveform[ind] = torch.finfo(torch.float32).min

        ind = torch.isnan(waveform)
        if ind.any():
            print(cl.Fore.RED + "NaN exists in waveform" + cl.Style.reset)
            # waveform[ind] = 0

        waveform = torch.nan_to_num(
            waveform,
            nan=0.0,
            posinf=torch.finfo(waveform.dtype).max,
            neginf=torch.finfo(waveform.dtype).min,
        )
        return waveform


class observationBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = (torch.finfo(torch.float32).tiny) ** 0.25
        self.pre_scale = 1e6
        # self.scale = self.pre_scale**2
        self.scale = torch.log2(torch.tensor(self.min)).abs()
        # self.scale = 1e9
        # self.scale = torch.log(torch.tensor(150)).abs()

    def forward(self, x):
        raise NotImplementedError


class toObservation(observationBase):
    def forward(self, x):
        x = x.to(torch.float64)
        batch_size, channels, frequency, time, image = x.shape
        x = x.reshape(batch_size * channels, frequency, time, image)
        x = torch.view_as_complex(x.contiguous())

        # convert
        magnitude = torch.abs(x)
        angle = torch.angle(x)

        # normalize
        magnitude = torch.log2(magnitude * self.pre_scale + self.min) / self.scale
        # magnitude = (magnitude * self.pre_scale) / self.scale
        angle = angle / torch.pi

        out = torch.stack([magnitude, angle], dim=-1)
        out = out.reshape(batch_size, channels, frequency, time, image)
        assert torch.isfinite(out).all(), "NaN or Inf in the observation space"
        return out.to(torch.float32)


class deObservation(observationBase):
    def forward(self, x):
        x = x.to(torch.float64)
        magnitude = x[..., 0]
        angle = x[..., 1]

        # denormalize
        magnitude = 2 ** (magnitude * self.scale + self.min) / self.pre_scale
        # magnitude = (magnitude * self.scale) / self.pre_scale
        angle = angle * torch.pi

        out = magnitude * torch.exp(1j * angle)
        out = torch.view_as_real(out)
        assert torch.isfinite(out).all(), "NaN or Inf in the observation space"
        return out.to(torch.float32)


class videoTransform(nn.Module):
    """
    A wrapper for a video-level transform that applies an image-level transform to each frame of the video.
    """

    def __init__(self, image_transform):
        super(videoTransform, self).__init__()
        self.image_transform = image_transform
        # image_transform is a nn.Sequential.
        #  check whether the rotationNoise is in the image_transform
        # if yes remove it and make another variable called rotationNoise
        if self.image_transform is not None:
            if "rotationNoise" in image_transform._modules:
                self.rotationNoise = image_transform._modules["rotationNoise"]
                del image_transform._modules["rotationNoise"]

    def forward(self, batch):
        """
        video: is in (frame, channel, time) format
        the image_transform is applied to each frame of the video.
        """
        x, y = batch
        assert x.dim() == 3, "Input x should be in (frame, channel, time) format"
        assert y.dim() == 3, "Input y should be in (frame, channel, time) format"

        if hasattr(self, "rotationNoise"):
            parameters = torch.rand(3)
            # apply the rotationNoise to each frame
            frames = [
                # processing the transfomr in pair
                self.rotationNoise(frame, parameters)
                for frame in zip(x, y)
            ]
            x = torch.stack([frame[0] for frame in frames], dim=0)
            y = torch.stack([frame[1] for frame in frames], dim=0)
        else:
            x, y = x, y

        if self.image_transform is None:
            return x, y

        # apply the image_transform to each frame
        transformed_frames = [self.image_transform(frame) for frame in zip(x, y)]

        x = torch.stack([frame[0] for frame in transformed_frames], dim=0)
        y = torch.stack([frame[1] for frame in transformed_frames], dim=0)
        return x, y


class batchNormalizeQuaternion(nn.Module):
    """
    Normalize quaternion data to unit length.

    Input:
    - x: torch.Tensor of shape (4, ....), where the last dimension represents the quaternion (x, y, z, w).

    Output:
    - normalized_x: torch.Tensor of the same shape as input, with quaternions normalized to unit length.
    """

    def __init__(self, epsilon=1e-7):
        super(batchNormalizeQuaternion, self).__init__()
        self.epsilon = epsilon

    def forward(self, y):
        # x, y = batch
        norm = torch.norm(y[:, 3:7], p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=self.epsilon)  # Prevent division by zero
        normalized_y = y[:, 3:7] / norm
        new_y = y.clone()
        new_y[:, 3:7] = normalized_y
        # return x, new_y
        # check the quad is within the range  of [-1,1]
        assert torch.all(
            (new_y[:, 3:7] >= -1) & (new_y[:, 3:7] <= 1)
        ), "Quaternion values out of range [-1, 1]"

        return new_y


class batchNormalizeSensor(nn.Module):
    """
    Normalize sensor data to unit length.

    Input:
    - x: torch.Tensor of shape (6, ....), where the last dimension represents the sensor data (x, y, z).

    Output:
    - normalized_x: torch.Tensor of the same shape as input, with sensor data normalized to unit length.

    default mean and std:
    acc_mean=0, acc_std=16*9.81 m/s^2, gyr_mean=0, gyr_std=2000/ 360* torch.pi rad/s
    """

    def __init__(
        self,
        acc_mean=0,
        acc_std=16 * 9.81,
        gyr_mean=0,
        gyr_std=2000 / 180 * torch.pi,
        epsilon=1,
    ):
        super(batchNormalizeSensor, self).__init__()
        self.epsilon = epsilon
        # self.epsilon = 1
        self.acc_mean = acc_mean
        self.acc_std = acc_std
        self.gyr_mean = gyr_mean
        self.gyr_std = gyr_std

    def forward(self, x):
        # x, y = batch
        new_x = x.clone()
        new_x[:, 0:3] = (x[:, 0:3] - self.acc_mean) / self.acc_std
        new_x[:, 3:6] = (x[:, 3:6] - self.gyr_mean) / self.gyr_std
        # sign = torch.sign(x)
        # new_x[:, 0:3] = torch.log2(x[:, 0:3].abs() + self.epsilon) / math.log2(
        #     self.acc_std
        # )
        # new_x[:, 3:6] = torch.log2(x[:, 3:6].abs() + self.epsilon) / math.log2(
        #     self.gyr_std
        # )
        # new_x = new_x * sign
        # check the sensor data is within the range of [-1,1]
        # assert torch.all(
        #     (new_x >= -1) & (new_x <= 1)
        # ), "Sensor values out of range [-1, 1]"
        return new_x
        # return new_x, y


class batchDenormalizeSensor(batchNormalizeSensor):
    """
    Denormalize sensor data from unit length back to original scale.

    Input:
    - x: torch.Tensor of shape (6, ....), where the last dimension represents the sensor data (x, y, z).

    Output:
    - denormalized_x: torch.Tensor of the same shape as input, with sensor data denormalized to original scale.
    """

    def forward(self, x):
        new_x = x.clone()
        new_x[:, 0:3] = x[:, 0:3] * self.acc_std + self.acc_mean
        new_x[:, 3:6] = x[:, 3:6] * self.gyr_std + self.gyr_mean
        # sign = torch.sign(x)
        # new_x[:, 0:3] = (
        #     2 ** (x[:, 0:3] * math.log2(self.acc_std)) - self.epsilon
        # ) * sign[:, 0:3]
        # new_x[:, 3:6] = (
        #     2 ** (x[:, 3:6] * math.log2(self.gyr_std)) - self.epsilon
        # ) * sign[:, 3:6]

        assert torch.isfinite(new_x).all(), "NaN or Inf in denormalized sensor data"
        return new_x


class bathNormalizeRelativePosNOri(nn.Module):
    """
    Normalize relative position data to unit length.

    Input:
    - x: torch.Tensor of shape (3, ....), where the last dimension represents the relative position data (x, y, z).

    Output:
    - normalized_x: torch.Tensor of the same shape as input, with relative position data normalized to unit length.
    """

    def __init__(self, mean=0, std=50.0, ori_mean=0, ori_std=1, epsilon=1):
        super(bathNormalizeRelativePosNOri, self).__init__()
        self.epsilon = epsilon
        self.mean = mean
        self.std = std
        self.ori_mean = ori_mean
        self.ori_std = ori_std

    def forward(self, y):
        # x, y = batch
        new_y = y.clone()
        new_y[:, 0:3] = (y[:, 0:3] - self.mean) / self.std
        new_y[:, 3:] = (y[:, 3:] - self.ori_mean) / self.ori_std
        # sign = torch.sign(y)
        # new_y[:, 0:3] = torch.log2(y[:, 0:3].abs() + self.epsilon) / math.log2(self.std)
        # new_y[:, 3:] = torch.log2(y[:, 3:].abs() + self.epsilon) / math.log2(
        #     self.ori_std
        # )
        # new_y = new_y * sign
        # check the relative position data is within the range of [-1,1]
        # assert torch.all(
        #     (new_y[:, 0:3] > -1) & (new_y[:, 0:3] < 1)
        # ), f"Relative position values out of range [-1, 1] with max {new_y[:,0:3].max()} and min {new_y[:,0:3].min()}"
        # return x, newx_y
        return new_y


class batchDenormalizeRelativePosNOri(bathNormalizeRelativePosNOri):
    """
    Denormalize relative position data from unit length back to original scale.

    Input:
    - x: torch.Tensor of shape (3, ....), where the last dimension represents the relative position data (x, y, z).

    Output:
    - denormalized_x: torch.Tensor of the same shape as input, with relative position data denormalized to original scale.
    """

    def forward(self, y):
        # x, y = batch
        new_y = y.clone()
        new_y[:, 0:3] = y[:, 0:3] * self.std + self.mean
        new_y[:, 3:] = y[:, 3:] * self.ori_std + self.ori_mean
        # sign = torch.sign(y)
        # new_y[:, 0:3] = (2 ** (y[:, 0:3] * math.log2(self.std)) - self.epsilon) * sign[
        #     :, 0:3
        # ]
        # new_y[:, 3:] = (
        #     2 ** (y[:, 3:] * math.log2(self.ori_std)) - self.epsilon
        # ) * sign[:, 3:]

        assert torch.isfinite(
            new_y
        ).all(), "NaN or Inf in denormalized relative position data"
        return new_y


class batchAddQuatChannel4Sensor(nn.Module):
    """
    Add a quaternion channel to sensor data.

    Input:
    - x: torch.Tensor of shape (6, ....), where the last dimension represents the sensor data (x, y, z).

    Output:
    - x_with_quat: torch.Tensor of shape (10, ....), where the first 6 channels are the original sensor data and the next 4 channels are the quaternion (0, 0, 0, 1).
    """

    def __init__(self):
        super(batchAddQuatChannel4Sensor, self).__init__()

    def forward(self, x):
        # x, y = batch
        batch, _, length = x.shape
        device = x.device
        quat_channel = torch.zeros((batch, 1, length), device=device)
        x_with_quat = torch.cat((x, quat_channel), dim=1)
        return x_with_quat
        # return x_with_quat, y


def integrate_orientation(angular_velocity, dt, initial_orientation=None, epsilon=1e-7):
    """
    Integrate angular velocity to compute orientation over time.

    Parameters:
    - angular_velocity: np.ndarray of shape (3, N), angular velocity in rad/s
    - dt: float, time step between samples
    - initial_orientation: scipy Rotation object or None (defaults to identity)
    - epsilon: small value to prevent division by zero

    Returns:
    - euler_angles: np.ndarray of shape (3, N), Euler angles over time
    """
    isTensor = False
    if isinstance(angular_velocity, torch.Tensor):
        angular_velocity = angular_velocity.cpu().numpy()
        isTensor = True
    N = angular_velocity.shape[1]
    orientations = []

    # Start with identity or provided initial orientation
    current_orientation = initial_orientation or Rotation.identity()
    orientations.append(current_orientation)

    for i in range(N):
        omega = angular_velocity[:, i]
        norm_omega = np.linalg.norm(omega)

        # Prevent division by zero
        if norm_omega > epsilon:
            axis = omega / (norm_omega + epsilon)
            angle = norm_omega * dt
            try:
                delta_rotation = Rotation.from_rotvec(axis * angle)
            except ValueError:
                delta_rotation = Rotation.identity()
        else:
            delta_rotation = Rotation.identity()

        # Update orientation
        current_orientation = current_orientation * delta_rotation
        orientations.append(current_orientation)

    # Convert to Euler angles safely
    try:
        euler_angles = np.array(
            [r.as_euler("xyz", degrees=False) for r in orientations]
        )
    except ValueError:
        # Fallback: use identity if conversion fails
        euler_angles = np.zeros((len(orientations), 3))

    # Transpose to shape (3, N)
    euler_angles = euler_angles[:-1].T
    if isTensor:
        euler_angles = torch.from_numpy(euler_angles)
    return euler_angles


class F32ToINT8(torch.autograd.Function):
    """
    Convert float32 tensor to uint8 tensor by mod into 4 times tensor
    scaling and quantization.
    ----------------------
    input : torch.float32 tensor (batch,channel,time)
    output: torch.int8 tensor (batch,channel,time, byte)
    """

    def __init__(self):
        super(F32ToINT8, self).__init__()

    @staticmethod
    def forward(self, x):
        x[:, :3] = x[:, :3] / 0.000061 / 9.8
        x[:, 3:] = x[:, 3:] / 0.0076 / (torch.pi / 180)

        x = x.to(torch.int32)
        byte0 = x % 256
        byte1 = (x // 256) % 256
        byte2 = (x // 65536) % 256
        byte3 = (x // 16777216) % 256

        x_int8 = torch.stack((byte0, byte1, byte2, byte3), dim=-1).to(torch.uint8)
        return x_int8.float()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (...)
        # We approximate the gradient as if the operation was linear:
        # val = b0 * 2^24 + b1 * 2^16 + b2 * 2^8 + b3
        scales = torch.tensor(
            [2**24, 2**16, 2**8, 1], device=grad_output.device, dtype=grad_output.dtype
        )
        # Expand grad_output to match input shape (..., 4) and scale
        return grad_output.unsqueeze(-1) * scales


class INT8ToF32(nn.Module):
    """
    Convert uint8 tensor to float32 tensor by mod into 4 times tensor
    scaling and quantization.
    ----------------------
    input : torch.uint8 tensor (batch,channel,time, byte)
    output: torch.float32 tensor (batch,channel,time)
    """

    def __init__(self):
        super(INT8ToF32, self).__init__()

    def forward(self, x):
        # byte0 = x[..., 0].to(torch.int32) << 24
        # byte1 = x[..., 1].to(torch.int32) << 16
        # byte2 = x[..., 2].to(torch.int32) << 8
        # byte3 = x[..., 3].to(torch.int32)
        byte0 = x[..., 0].to(torch.int32) * 1
        byte1 = x[..., 1].to(torch.int32) * 256
        byte2 = x[..., 2].to(torch.int32) * 65536
        byte3 = x[..., 3].to(torch.int32) * 16777216

        x_float = byte0 + byte1 + byte2 + byte3
        x_float = x_float.float()
        x_float[:, :3] = x_float[:, :3] * 0.000061 * 9.8
        x_float[:, 3:] = x_float[:, 3:] * 0.0076 * (torch.pi / 180)
        return x_float


def test_inverse_relationship():
    config = json.load(open("./config.json"))
    from torch.utils.data import DataLoader

    from utils.mdatasets import RIDIDataset

    datasets = RIDIDataset(
        root="datasets/RIDI/",
        mode="train",
        transform=None,
        window_size=config["window_size"],
        stride=config["stride"],
        useStep=config["useStep"],
        encoding=False,
    )
    loader = DataLoader(datasets, batch_size=32, shuffle=True)
    imu_data = next(iter(loader))
    imu_data = imu_data["dataY"]

    # Create random IMU data
    batch_size = 10
    frames = 100
    channels = 6  # 3 for accelerometer, 3 for gyroscope
    window_size = config["window_size"]
    n_fft = config["n_fft"]
    # n_fft = 14
    # imu_data = torch.randn(batch_size, channels, window_size)

    # random winthin the range of -1, 1
    # imu_data = torch.rand(batch_size, 6, window_size) * 2 - 1

    # Initialize the transformation classes
    imu_to_yuv = nn.Sequential(
        batchTimeToFrequency(n_fft=n_fft),
        toObservation(),
    )
    yuv_to_imu = nn.Sequential(
        deObservation(),
        batchFrequencyToTime(output_length=window_size, n_fft=n_fft),
    )

    # Convert IMU to YUV
    yuv_data = imu_to_yuv(imu_data)
    print("YUV shape : ", yuv_data.shape)
    # Convert YUV back to IMU
    imu_data_reconstructed = yuv_to_imu(yuv_data)

    # Check if the original IMU data and the reconstructed IMU data are close
    if torch.allclose(imu_data, imu_data_reconstructed, atol=1e-12):
        print("The functions are inverse relationships.")
    else:
        # Calculate the absolute difference
        abs_diff = torch.abs(imu_data - imu_data_reconstructed)
        abs_diff = abs_diff[abs_diff != 0]  # Exclude zero differences
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        print(f"The functions are NOT inverse relationships.")
        print(f"Max difference: {max_diff:.8e}")
        print(f"STD difference: {abs_diff.std().item():.8e}")
        print(f"Mean difference: {mean_diff:.8e}")
        print(f"min difference: {abs_diff.min().item():.8e}")
        print(
            f"Sum error: {(imu_data_reconstructed.cumsum(dim=-1)- imu_data.cumsum(dim=-1)).abs()[:,-1].max():.8e}"
        )

        # Print the indices where the difference is the largest
        # max_diff_indices = torch.argmax(abs_diff, dim=-1)
        # for i in range(batch_size):
        #     print(f"Batch {i}: Max difference at index {max_diff_indices[i].item()}")


class eulerToQuaternion(mModule):
    """
    Convert Euler angles to quaternions.
    Expects input tensors with Euler angles (roll, pitch, yaw) representing rotations.
    """

    def __init__(self, config):
        super(eulerToQuaternion, self).__init__(config)
        # Assuming config contains:
        # self.euler_indices: list of 3 indices [roll_idx, pitch_idx, yaw_idx]
        # self.replace: boolean, whether to replace the euler angles or append quaternions
        if not hasattr(self, "euler_indices"):
            self.euler_indices = [3, 4, 5]  # Default indices

    def euler_to_quat(self, roll, pitch, yaw):
        """
        Convert (roll, pitch, yaw) measured in radians to quaternion (x, y, z, w).
        """
        qx = torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.cos(
            yaw / 2
        ) - torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
        qy = torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.cos(
            yaw / 2
        ) + torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2)
        qz = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.sin(
            yaw / 2
        ) - torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2)
        qw = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.cos(
            yaw / 2
        ) + torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)

        return torch.stack([qx, qy, qz, qw], dim=0)

    def forward(self, data):
        if not self.should_apply():
            return data

        x, y = data

        # # Extract roll, pitch, yaw based on configured indices
        # yaw = x[self.euler_indices[0]]
        # pitch = x[self.euler_indices[1]]
        # roll = x[self.euler_indices[2]]

        # # Convert to quaternion shape (4, time)
        # x_quats = self.euler_to_quat(roll, pitch, yaw)

        # x_new = torch.cat([x, x_quats], dim=0)

        yaw = y[self.euler_indices[0]]
        pitch = y[self.euler_indices[1]]
        roll = y[self.euler_indices[2]]

        # Convert to quaternion shape (4, time)
        y_quats = self.euler_to_quat(roll, pitch, yaw)

        y_new = torch.cat([y[:3], y_quats], dim=0)

        return x, y_new


if __name__ == "__main__":
    # for _ in range(10):
    test_inverse_relationship()
