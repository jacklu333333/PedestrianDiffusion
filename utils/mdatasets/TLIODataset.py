import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .common_imports import *
from .mDataset import mDataset
from .utils import *


class TLIODataset(mDataset):
    """
    TLIO dataset class
    """

    def __init__(
        self,
        root,
        useStep,
        mode="train",
        transform=None,
        window_size=200,
        stride=10,
        sampling_rate=200,
        keep_filters: list = None,
        skip_filters: list = None,
        GravityRemoval=False,
        encoding=False,
        next_window=False,
        precision=torch.float32,
    ):
        super().__init__(
            root=root,
            useStep=useStep,
            mode=mode,
            transform=transform,
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            keep_filters=keep_filters,
            skip_filters=skip_filters,
            GravityRemoval=GravityRemoval,
            encoding=encoding,
            next_window=next_window,
            precision=precision,
        )
        self._label = DATASET_DICT["TLIO"]

        # Adjust root to point to tlio_golden if not already
        if self.root.name != "tlio_golden":
            self.root = self.root.joinpath("tlio_golden")

        if mode == "train":
            self.files = self.read_list(self.root.joinpath("train_list.txt"))
        elif mode == "val":
            self.files = self.read_list(self.root.joinpath("val_list.txt"))
        elif mode == "test":
            self.files = self.read_list(self.root.joinpath("test_list.txt"))
            # rank_zero_info(
            #     cl.Fore.yellow
            #     + "Warning: Using only 1 test file for quick testing."
            #     + cl.Style.reset
            # )
        else:
            raise ValueError(f"Unknown mode {mode}")

        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()

        rank_zero_info(
            cl.Fore.green + f"Loaded {self.chunk_index[-1]} samples" + cl.Style.reset
        )

    def read_list(self, file):
        if not file.exists():
            rank_zero_info(cl.Fore.red + f"List file {file} not found" + cl.Style.reset)
            return []
        with open(file, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        return [self.root.joinpath(f) for f in files]

    def load_files(self):
        self.Data = []
        rank_zero_info(cl.Fore.green + f"Loading {self.mode} data" + cl.Style.reset)
        for file in tqdm(self.files):
            newBatches = self._load_single_file(file, self.window_size, self.stride)
            self.Data.extend(newBatches)
        self.endLoading()

    def _load_single_file(self, file, window_size, stride):
        # if self.mode == "test":
        #     return self._load_raw(file, window_size, stride)
        # else:
        #     return self._load_vio(file, window_size, stride)
        return self._load_vio(file, window_size, stride)

    def _load_raw(self, file, window_size, stride):
        csv_file = file.joinpath("imu_samples_0.csv")
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file {csv_file} not found")

        from scipy.interpolate import interp1d

        imu_data = pd.read_csv(csv_file)
        ts = np.copy(imu_data.iloc[:, 0].values) * 1e-3
        gyro = np.copy(imu_data.iloc[:, 2:5].values)
        accel = np.copy(imu_data.iloc[:, 5:8].values)

        # Skip first 50 samples as in load_all
        idx_start = 50
        ts = ts[idx_start:]
        gyro = gyro[idx_start:]
        accel = accel[idx_start:]

        # Load Ground Truth from NPY
        npy_file = file.joinpath("imu0_resampled.npy")
        vio_data = np.load(npy_file)
        vio_ts = vio_data[:, 0]  # us
        vio_rot = vio_data[:, -10:-6]  # quat
        vio_pos = vio_data[:, -6:-3]
        vio_vel = vio_data[:, -3:]

        # Interpolate
        # We use linear interpolation for pos and vel
        f_pos = interp1d(
            vio_ts,
            vio_pos,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )
        _pos_full = f_pos(ts)

        f_vel = interp1d(
            vio_ts,
            vio_vel,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )
        _vel_full = f_vel(ts)

        # For rotation, simple linear interpolation + normalization
        f_rot = interp1d(
            vio_ts,
            vio_rot,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )
        _rot_full = f_rot(ts)
        # Normalize quaternions
        norm = np.linalg.norm(_rot_full, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm < 1e-6] = 1.0
        _rot_full = _rot_full / norm

        # Estimate sampling rate
        dt = np.mean(np.diff(ts)) * 1e-9  # ts is in us, convert to s
        original_sampling_rate = 1.0 / dt
        print(
            cl.Fore.red
            + f"average sampling rate is {original_sampling_rate}"
            + cl.Style.reset
        )

        jump = int(round(original_sampling_rate / self.sampling_rate))
        rag = 1 if (hasattr(self, "endToEnd") or self.mode == "test") else jump
        newBatches = []
        for i in range(rag):
            _ts = ts[i::jump]
            _gyro = gyro[i::jump]
            _accel = accel[i::jump]

            _pos = _pos_full[i::jump]
            _vel = _vel_full[i::jump]
            _rot = _rot_full[i::jump]

            # Convert to torch tensors and move to device
            _acc = torch.from_numpy(_accel).to(dtype=self.precision, device=self.device)
            _gyr = torch.from_numpy(_gyro).to(dtype=self.precision, device=self.device)
            _vel = torch.from_numpy(_vel).to(dtype=self.precision, device=self.device)
            _pos = torch.from_numpy(_pos).to(dtype=self.precision, device=self.device)
            _rot = torch.from_numpy(_rot).to(dtype=self.precision, device=self.device)

            # Dummy mag and acceleration as they are not in the npy file but required by select_input_output signature
            _mag = torch.zeros_like(_acc).to(self.device)
            # rank_zero_info(
            #     cl.Fore.yellow
            #     + f"Warning: Using dummy magnetometer data for file {file.name}"
            #     + cl.Style.reset
            # )
            assert (
                self.useStep == False
            ), "TLIO dataset has not implemented step data yet."
            if self.GravityRemoval:
                _acc = GravityRemoval(acc=_acc, location="TLIO")

            _, _acceleration = self.vel_acc_generator(
                location=_pos, sample_rate=self.sampling_rate
            )
            # Use select_input_output to format data
            # We pass the whole sequence
            observation, label = self.select_input_output(
                acc=_acc,
                gyr=_gyr,
                mag=_mag,
                location=_pos,
                velocity=_vel,
                acceleration=_acceleration,
                orientation=_rot,
            )

            # Parse sequence ID from folder name
            try:
                seq_id = int(file.name)
            except ValueError:
                seq_id = 0
            if self.encoding:
                encoding, attention_mask = get_enconding(
                    {
                        "person": "unknown",
                        "action": "unknown",
                        "device": "unknown",
                        "mounted": "chestmounted",
                        "environment": "unknown",  # TODO: fill in environment
                        "dataset": "TLIO",
                        "annotation": "VIO",
                    },
                    mode=self.mode,
                )
            else:
                encoding = []
                attention_mask = []

            newBatch = {
                "dataX": observation,  # [length, 6]
                "dataY": label,  # [length, 6] (vel + ang_vel)
                "dataL": torch.tensor(window_size),
                # "dataSteps": [],  # Not using steps for now
                # "classes": [],
                "encoding": encoding,
                "attention_mask": attention_mask,
                "name": torch.tensor(self.files.index(file), dtype=torch.int32),
                "action": torch.tensor(0),
                "device": torch.tensor(0),
                "user": torch.tensor(0),
                "mounted": torch.tensor(0),
                "size": _acc.shape[0] - window_size + 1,
            }
            newBatches.append(newBatch)

        return newBatches

    def _load_vio(self, file, window_size, stride):
        npy_file = file.joinpath("imu0_resampled.npy")
        if not npy_file.exists():
            return []

        try:
            data = np.load(npy_file)
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            return []

        # data structure: ts (1), gyro (3), accel (3), ..., rot (4), pos (3), vel (3)
        # indices: ts: 0, gyro: 1:4, accel: 4:7, rot: -10:-6, pos: -6:-3, vel: -3:
        original_sampling_rate = json.load(
            open(file.joinpath("imu0_resampled_description.json"), "r")
        )["approximate_frequency_hz"]
        jump = int(round(original_sampling_rate / self.sampling_rate))
        rag = 1 if (hasattr(self, "endToEnd") or self.mode == "test") else jump
        newBatches = []
        for i in range(rag):
            ts = data[:, 0]
            gyro = data[i::jump, 1:4]
            accel = data[i::jump, 4:7]
            rot = data[i::jump, -10:-6]
            pos = data[i::jump, -6:-3]
            vel = data[i::jump, -3:]
            # Convert to torch tensors and move to device
            _acc = torch.from_numpy(accel).to(dtype=self.precision, device=self.device)
            _gyr = torch.from_numpy(gyro).to(dtype=self.precision, device=self.device)
            _vel = torch.from_numpy(vel).to(dtype=self.precision, device=self.device)
            _pos = torch.from_numpy(pos).to(dtype=self.precision, device=self.device)
            _rot = torch.from_numpy(rot).to(dtype=self.precision, device=self.device)

            # Dummy mag and acceleration as they are not in the npy file but required by select_input_output signature
            _mag = torch.zeros_like(_acc).to(self.device)
            # rank_zero_info(
            #     cl.Fore.yellow
            #     + f"Warning: Using dummy magnetometer data for file {file.name}"
            #     + cl.Style.reset
            # )
            assert (
                self.useStep == False
            ), "TLIO dataset has not implemented step data yet."
            if self.GravityRemoval:
                _acc = GravityRemoval(acc=_acc, location="TLIO")

            _, _acceleration = self.vel_acc_generator(
                location=_pos, sample_rate=self.sampling_rate
            )
            # Use select_input_output to format data
            # We pass the whole sequence
            observation, label = self.select_input_output(
                acc=_acc,
                gyr=_gyr,
                mag=_mag,
                location=_pos,
                velocity=_vel,
                acceleration=_acceleration,
                orientation=_rot,
            )

            # Parse sequence ID from folder name
            try:
                seq_id = int(file.name)
            except ValueError:
                seq_id = 0
            if self.encoding:
                encoding, attention_mask = get_enconding(
                    {
                        "person": "unknown",
                        "action": "unknown",
                        "device": "unknown",
                        "mounted": "chestmounted",
                        "environment": "unknown",  # TODO: fill in environment
                        "dataset": "TLIO",
                        "annotation": "MSCKF",
                    },
                    mode=self.mode,
                )
            else:
                encoding = []
                attention_mask = []
            newBatch = {
                "dataX": observation,  # [length, 6]
                "dataY": label,  # [length, 6] (vel + ang_vel)
                "dataL": torch.tensor(window_size),
                # "dataSteps": [],  # Not using steps for now
                # "classes": [],
                "encoding": encoding,
                "attention_mask": attention_mask,
                "name": torch.tensor(self.files.index(file), dtype=torch.int32),
                "action": torch.tensor(0),
                "device": torch.tensor(0),
                "user": torch.tensor(0),
                "mounted": torch.tensor(0),
                "size": _acc.shape[0] - window_size + 1,
            }
            newBatches.append(newBatch)

        return newBatches
