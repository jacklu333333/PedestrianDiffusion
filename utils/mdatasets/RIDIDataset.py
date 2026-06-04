from .common_imports import *
from .mDataset import mDataset
from .utils import *


class RIDIDataset(mDataset):
    """
    RIDI: Robust IMU Double Integration
    link: https://www.kaggle.com/datasets/kmader/ridi-robust-imu-double-integration
    """

    def __init__(
        self,
        root,
        useStep,
        mode: str = "train",
        window_size: int = 100,
        stride: int = 1,
        sampling_rate: int = 100,
        transform: torch.nn.modules.container.Sequential = None,
        keep_filters: list = None,
        skip_filters: list = None,
        GravityRemoval=False,
        encoding=False,
        next_window=False,
        precision=torch.float32,
    ) -> None:
        """
        Initialize the RIDI dataset
        ---------------------------------------
        input:
        see mDataset
        """
        super().__init__(
            root=root,
            useStep=useStep,
            mode=mode,
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            transform=transform,
            keep_filters=keep_filters,
            skip_filters=skip_filters,
            GravityRemoval=GravityRemoval,
            encoding=encoding,
            next_window=next_window,
            precision=precision,
        )
        self._label = DATASET_DICT["RIDI"]
        self.user_table.extend(
            [
                "dan",
                "hang",
                "hao",
                "huayi",
                "ma",
                "ruixuan",
                "shali",
                "tang",
                "xiaojing",
                "yajie",
                "zhicheng",
            ]
        )
        self.mounted_table.extend(
            [
                "bag",
                "bag_low",
                "bag_normal",
                "bag_side",
                "bag_speed",
                "bag_stop",
                "bag_test",
                "body",
                "body_backward",
                "body_fast",
                "body_normal",
                "body_side",
                "body_slow",
                "body_stop",
                "body_test",
                "handheld",
                "handheld_normal",
                "handheld_side",
                "handheld_side_test",
                "handheld_speed",
                "handheld_test",
                "leg",
                "leg_front",
                "leg_new",
                "lopata",
            ]
        )

        self.files = self.read_list(self.root.joinpath(f"{mode}_list.txt"))
        if mode == "test":
            # rank_zero_info(
            #     cl.Fore.yellow
            #     + f"- Test mode is shoten to one file for quick evalution."
            #     + cl.Style.reset
            # )
            self.files = self.files
        # self.files = self.files[:2]
        # rank_zero_info(
        #     cl.Fore.yellow
        #     + f"- cut to first two files for quick dev run."
        #     + cl.Style.reset
        # )

        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()
        # self.Data["noise"] = [
        #     x - y for x, y in zip(self.Data["dataX"], self.Data["dataY"])
        # ]
        # assert (
        #     self.dataX.shape[0] == self.dataY.shape[0]
        # ), f"{self.dataX.shape} != {self.dataY.shape}"
        # assert self.dataX.shape[1] == 6, f"{self.dataX.shape[1]} != 6"
        # assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        # assert (
        #     self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        # ), f"{self.config['window_size']} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"
        rank_zero_info(
            cl.Fore.green
            + f'- Loaded "{self.chunk_index[-1]}" samples'
            + cl.Style.reset
        )

    def read_list(self, file):
        """
        read the list of the file
        ---------------------------------------
        input:
            file: Path
                the file to read
        ---------------------------------------
        return:
            files: list
                the list of the file
        """
        with open(file, "r") as f:
            files = f.readlines()
            files = [
                self.root.joinpath("datasets/data_publish_v2/" + f.strip())
                for f in files
            ]
        return files
        rank_zero_info(
            cl.Fore.red
            + f'only use one File "{files[0].name}" for all mode'
            + cl.Style.reset
        )
        return files[:1]

    def load_files(self):
        """
        Load the files from the list
        ---------------------------------------
        input:
            None
        ---------------------------------------
        return:
            None
        """
        # self.Data = {
        #     "dataX": [],
        #     "dataY": [],
        #     "dataL": [],
        #     "dataSteps": [],
        #     "classes": [],
        #     "encoding": [],
        #     "name": [],
        #     "index": [],
        #     "action": [],
        #     "device": [],
        #     "user": [],
        #     "mounted": [],
        # }
        self.Data = []

        for file in tqdm(self.files):
            newBatches = self._load_single_file(
                file.joinpath("processed/data.csv"),
                window_size=self.window_size,
                stride=self.stride,
            )
            self.Data.extend(newBatches)
            # for key, value in newBatch.items():
            #     if key not in self.Data:
            #         raise ValueError(f"Key {key} not found in Data")
            #     self.Data[key].extend(value)
        # self.dataX = np.concatenate(
        #     [_.detach().cpu().numpy() for _ in self.dataX], axis=0
        # )
        # self.dataY = np.concatenate(
        #     [_.detach().cpu().numpy() for _ in self.dataY], axis=0
        # )
        # self.dataL = np.concatenate(
        #     [_.detach().cpu().numpy() for _ in self.dataL], axis=0
        # )

        self.endLoading()

    def _load_single_file(self, file, window_size: int, stride: int, return_wave=False):
        def get_keyword_index(filename, table):
            # action = filename.split("_", 1)[1]
            action_clean = re.sub(r"\d+$", "", filename)
            return table.index(action_clean)

        """
        Load the data from a single file
        ---------------------------------------
        input:
            file: Path
                the file to load
        ---------------------------------------
        return:
            X: torch.Tensor
                the input data
            Y: torch.Tensor
                the output data
        """
        # load the data from a single file
        df = pd.read_csv(file)
        # remove ""processed/data.csv"" from the file
        file = file.parent.parent
        # set time as index
        df.set_index("time", inplace=True)
        # cover nano seconds to seconds
        df.index = df.index / 1e9
        # cover from float to datetime
        df.index = pd.to_datetime(df.index, unit="s")
        # Estimate original sampling rate based on timestamps
        if len(df.index) > 1:
            total_seconds = (df.index[-1] - df.index[0]).total_seconds()
            if total_seconds > 0:
                original_sampling_rate = int(round(len(df) / total_seconds))
            else:
                raise ValueError(
                    "Time difference between first and last index is zero or negative."
                )
        else:
            raise ValueError("Not enough data to estimate sampling rate.")

        # ms = 1000 // self.sampling_rate
        # df = df.resample(f"{ms}ms").mean()
        # df = df.ffill()

        # remove the first column
        # acc = df[["linacce_x", "linacce_y", "linacce_z"]].to_numpy()
        # acc = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
        # acc_norm = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
        # acc_norm = np.linalg.norm(acc_norm, axis=-1)
        # gyr = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        # mag = df[["magnet_x", "magnet_y", "magnet_z"]].to_numpy()
        # orientation = df[["ori_x", "ori_y", "ori_z", "ori_w"]].to_numpy()
        # location = df[["pos_x", "pos_y", "pos_z"]].to_numpy()
        # grav = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
        # Raw sensor values (device frame)
        gyr = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        acc = df[["acce_x", "acce_y", "acce_z"]].to_numpy()
        mag = df[["magnet_x", "magnet_y", "magnet_z"]].to_numpy()
        location = df[["pos_x", "pos_y", "pos_z"]].to_numpy()

        # Use game rotation vector as device orientation; align to initial Tango orientation
        init_tango_ori = quaternion.quaternion(
            *df[["ori_w", "ori_x", "ori_y", "ori_z"]].iloc[0].to_numpy()
        )
        game_rv = quaternion.from_float_array(
            df[["rv_w", "rv_x", "rv_y", "rv_z"]].to_numpy()
        )
        init_rotor = init_tango_ori * game_rv[0].conj()
        ori = init_rotor * game_rv  # sequence of quaternions

        # orientation in (x, y, z, w) for downstream code (rotateToWorldFrame + _orientation2degpersec)
        orientation = quaternion.as_float_array(ori)[:, [1, 2, 3, 0]]

        # Acc norm (for step/motion detection)
        acc_norm = np.linalg.norm(acc, axis=-1)

        # Keep gravity copy if needed
        grav = df[["grav_x", "grav_y", "grav_z"]].to_numpy()

        nz = np.zeros(gyr.shape[0]).reshape(-1, 1)
        gyr = quaternion.from_float_array(np.concatenate([nz, gyr], axis=1))
        acc = quaternion.from_float_array(np.concatenate([nz, acc], axis=1))
        mag = quaternion.from_float_array(np.concatenate([nz, mag], axis=1))

        gyr = quaternion.as_float_array(ori * gyr * ori.conj())[:, 1:]
        acc = quaternion.as_float_array(ori * acc * ori.conj())[:, 1:]
        mag = quaternion.as_float_array(ori * mag * ori.conj())[:, 1:]

        timestamps = df.index.to_numpy()
        # convert from np.float64 (seconds) to datetime64[ns]
        timestamps = timestamps.astype("datetime64[ns]")

        # unit conversion
        # acc = acc * 9.81
        # gyr = gyr
        mag = mag / 100
        # acc, gyr, mag = rotateToWorldFrame(
        #     acc, gyr, mag, rotation=orientation, sample_rate=original_sampling_rate
        # )
        if self.GravityRemoval:
            acc = GravityRemoval(acc=acc, location="RIDI", rescale=False)
        # X, Y = [], []
        # # convert to tensor on cuda and cut the data
        # for ii in range(2):
        #     acc_ = torch.from_numpy(acc).float().cuda()[ii::2]
        #     gyr_ = torch.from_numpy(gyr).float().cuda()[ii::2]
        #     mag_ = torch.from_numpy(mag).float().cuda()[ii::2]
        #     location_ = torch.from_numpy(location).float().cuda()[ii::2]
        #     orientation_ = torch.from_numpy(orientation).float().cuda()[ii::2]
        #     acc_, gyr_, mag_, location_ = self._time_series_filter(
        #         5, acc_, gyr_, mag_, location_
        #     )

        #     velocity_, acceleration_ = self.vel_acc_generator(
        #         location_, sample_rate=100.0
        #     )
        #     assert (
        #         acceleration_.shape == velocity_.shape == location_.shape
        #     ), f"{acceleration_.shape} != {velocity_.shape} != {location_.shape}"
        #     assert acc_.shape[0] == gyr_.shape[0] == velocity_.shape[0]
        #     acc_, gyr_, prevel_, label_ = self.split_data(
        #         acc=acc_,
        #         gyr=gyr_,
        #         velocity=velocity_,
        #         acceleration=acceleration_,
        #         window_size=window_size,
        #         stride=stride,
        #     )
        #     x = torch.cat([acc_, gyr_, prevel_], dim=1)
        #     y = label_
        #     assert x.shape[0] == y.shape[0], f"{x.shape} != {y.shape}"
        #     assert x.shape[1] == 9, f"{x.shape[1]} != 9"
        #     X.append(x)
        #     Y.append(y)
        # X = torch.cat(X, dim=0)
        # Y = torch.cat(Y, dim=0)
        newBatches = []
        jump = int(round(original_sampling_rate / self.sampling_rate))
        rag = 1 if (hasattr(self, "endToEnd") or self.mode == "test") else jump
        for i in range(rag):
            newBatch = {
                "dataX": [],
                "dataY": [],
                "dataL": [],
                "dataSteps": [],
                "encoding": [],
                "name": [],
                "index": [],
                "action": [],
                "device": [],
                "user": [],
                "mounted": [],
            }
            acc_ = (
                torch.from_numpy(acc)
                .to(dtype=self.precision)[i::jump]
                .to(device=self.device, non_blocking=True)
            )
            gyr_ = (
                torch.from_numpy(gyr)
                .to(dtype=self.precision)[i::jump]
                .to(device=self.device, non_blocking=True)
            )
            mag_ = (
                torch.from_numpy(mag)
                .to(dtype=self.precision)[i::jump]
                .to(device=self.device, non_blocking=True)
            )
            location_ = (
                torch.from_numpy(location)
                .to(dtype=self.precision)[i::jump]
                .to(device=self.device, non_blocking=True)
            )
            acc_norm_ = (
                torch.from_numpy(acc_norm[i::jump])
                .to(dtype=self.precision)
                .to(device=self.device, non_blocking=True)
            )
            orientation_ = (
                torch.from_numpy(orientation[i::jump])
                .to(dtype=self.precision)
                .to(device=self.device, non_blocking=True)
            )
            timestamps_ = timestamps[i::jump]
            # acc_, gyr_, mag_, location_ = self._time_series_filter(
            #     5, acc_, gyr_, mag_, location_
            # )
            # if hasattr(self, "endToEnd"):
            #     self._position = (
            #         location_.clone().detach().numpy()
            #         - location_[0:1].clone().detach().numpy()
            #     )
            #     self._orientation = (
            #         R.from_quat(orientation_).inv().as_euler("xyz", degrees=False)
            #     )

            (
                acc_,
                gyr_,
                mag_,
                location_,
                acc_norm_,
                orientation_,
                timestamps_,
            ) = self._first_motion(
                acc_norm_,
                0.5,
                acc_,
                gyr_,
                mag_,
                location_,
                acc_norm_,
                orientation_,
                timestamps_,
            )

            # orientation_ = self._orientation2degpersec(
            #     orientation_, sample_rate=self.sampling_rate, ts=timestamps_
            # )
            # acc_, gyr_, mag_, location_, orientation_ = self._time_series_filter(
            #     30, acc_, gyr_, mag_, location_, orientation_
            # )

            velocity_, acceleration_ = self.vel_acc_generator(
                location=location_, sample_rate=self.sampling_rate, ts=timestamps_
            )
            assert (
                acc_.shape[0]
                == acc_norm_.shape[0]
                == gyr_.shape[0]
                == mag_.shape[0]
                == velocity_.shape[0]
                == acceleration_.shape[0]
                == orientation_.shape[0]
            ), f"{acc_.shape} != {acc_norm_.shape} != {gyr_.shape} != {mag_.shape} != {velocity_.shape} != {acceleration_.shape} != {orientation_.shape}"
            valleys, acc_norm_ = self._step_finder(acc_norm=acc_norm_, window_size=5)
            observation, label = self.select_input_output(
                acc=acc_,
                gyr=gyr_,
                mag=mag_,
                location=location_,
                velocity=velocity_,
                acceleration=acceleration_,
                orientation=orientation_,
            )

            newBatch["dataX"] = observation
            newBatch["dataY"] = label
            newBatch["dataL"] = torch.tensor(window_size)
            # newBatch["dataSteps"] = torch.tensor(valleys, dtype=torch.int32)
            newBatch["dataSteps"] = self.get_pairIndex(
                stepIdx=valleys, window_size=window_size, modes=self.mode
            )
            newBatch["name"] = torch.tensor(self.files.index(file), dtype=torch.int32)
            newBatch["index"] = torch.arange(len(acc_), dtype=torch.int32)
            newBatch["action"] = torch.tensor(0, dtype=torch.int32)
            newBatch["device"] = torch.tensor(0, dtype=torch.int32)
            user_idx = torch.tensor(
                get_keyword_index(file.name.split("_", 1)[0], self.user_table),
                dtype=torch.int32,
            )
            newBatch["user"] = user_idx
            mounted_idx = torch.tensor(
                get_keyword_index(file.name.split("_", 1)[1], self.mounted_table),
                dtype=torch.int32,
            )
            newBatch["mounted"] = mounted_idx
            newBatch["size"] = acc_.shape[0] - window_size + 1
            if self.encoding:
                encoding, attention_mask = get_enconding(
                    {
                        "person": str(file.name.split("_", 1)[0]),
                        "action": "walking",
                        "device": "unknown",
                        "mounted": str(file.name.split("_", 1)[1]),
                        "environment": "unknown",  # TODO: fill in environment
                        "dataset": "RIDI",
                        "annotation": "tango",
                    },
                    mode=self.mode,
                )
                newBatch["encoding"] = encoding
                newBatch["attention_mask"] = attention_mask
            else:
                newBatch["encoding"] = []
                newBatch["attention_mask"] = []
            newBatches.append(newBatch)

            # x, y, length = self.split_by_step(
            #     stepIdx=valleys,
            #     acc=acc_,
            #     gyr=gyr_,
            #     mag=mag_,
            #     orientation=orientation_,
            #     velocity=velocity_,
            #     acceleration=acceleration_,
            #     window_size=window_size,
            #     stride=stride,
            #     modes=self.mode,
            # )

            # newBatch["dataX"].extend(x)
            # newBatch["dataY"].extend(y)
            # newBatch["dataL"].extend(length)
            # newBatch["dataSteps"].append(valleys)
            # encoding = process_Encoder(str(file))
            # # expand the encoding to match the length of X
            # newBatch["encoding"].extend([encoding for _ in range(len(x))])
            # newBatch["name"].extend(
            #     [
            #         torch.tensor(self.files.index(file), dtype=torch.int32)
            #         for _ in range(len(x))
            #     ]
            # )
            # newBatch["index"].extend(list(torch.arange(len(x), dtype=torch.int32)))
            # newBatch["action"].extend([torch.tensor(0) for _ in range(len(x))])
            # newBatch["device"].extend([torch.tensor(0) for _ in range(len(x))])
            # user_idx = torch.tensor(
            #     get_keyword_index(file.name.split("_", 1)[0], self.user_table)
            # )
            # newBatch["user"].extend([user_idx for _ in range(len(x))])
            # mounted_idx = torch.tensor(
            #     get_keyword_index(file.name.split("_", 1)[1], self.mounted_table)
            # )
            # newBatch["mounted"].extend([mounted_idx for _ in range(len(x))])

        return newBatches
