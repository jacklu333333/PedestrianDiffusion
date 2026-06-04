from .common_imports import *
from .mDataset import mDataset
from .utils import *


class RoNINDataset(mDataset):
    """
    RoNIN dataset class
    link: https://ronin.cs.sfu.ca/
    ---------------------------------------
    input:
    see mDataset
    """

    def __init__(
        self,
        root,
        useStep,
        mode="train",
        transform=None,
        window_size=100,
        stride=10,
        sampling_rate=100,
        keep_filters: list = None,
        skip_filters: list = None,
        GravityRemoval=False,
        encoding=False,
        next_window=False,
        precision=torch.float32,
    ):
        """
        Initialize the RoNIN dataset
        ---------------------------------------
        input:
        see mDataset
        """
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
        self._label = DATASET_DICT["RoNIN"]
        self.device_table.extend(
            ["asus3", "asus4", "asus5", "asus6", "asus7", "samsung1"]
        )

        if mode == "train" or mode == "val":
            self.files = self.read_list(self.root.joinpath(f"lists/list_{mode}.txt"))
        elif mode == "test":
            rank_zero_info(
                cl.Fore.green + "Test mode with [seen] data" + cl.Style.reset
            )
            self.files = self.read_list(
                self.root.joinpath(f"lists/list_{mode}_seen.txt")
            )
        elif mode == "test_unseen":
            self._label = DATASET_DICT["RoNIN_unseen"]
            rank_zero_info(
                cl.Fore.green + "Test mode with [unseen] data" + cl.Style.reset
            )
            self.files = self.read_list(self.root.joinpath(f"lists/list_{mode}.txt"))
            self.mode = "test"
        else:
            raise ValueError(f"Unknown mode {mode}")

        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()

        # assert len(self.Data["dataX"]) == len(
        #     self.Data["dataY"]
        # ), f"{len(self.Data['dataX'])} != {len(self.Data['dataY'])}"

        rank_zero_info(
            cl.Fore.green + f"Loaded {self.chunk_index[-1]} samples" + cl.Style.reset
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
                Path(self.root).joinpath("Data").joinpath(f.strip()) for f in files
            ]
        return files

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
        rank_zero_info(cl.Fore.green + f"Loading {self.mode} data" + cl.Style.reset)
        for file in tqdm(self.files):
            newBatches = self._load_single_file(
                file, window_size=self.window_size, stride=self.stride
            )
            self.Data.extend(newBatches)
            # for key, value in newBatch.items():
            #     if key not in self.Data:
            #         # self.Data[key] = []
            #         raise ValueError(f"Key {key} not found in Data")
            #     self.Data[key].extend(value)
        # self.dataX = np.concatenate(self.dataX, axis=0)
        # self.dataY = np.concatenate(self.dataY, axis=0)
        # self.dataL = np.concatenate(self.dataL, axis=0)
        # self.classes = np.concatenate(self.classes, axis=0)

        # assert (
        #     self.dataX.shape[0] == self.dataY.shape[0]
        # ), f"{self.dataX.shape} != {self.dataY.shape}"
        # assert self.dataX.shape[1] == 6, f"{self.dataX.shape[1]} != 6"
        # assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        # assert (
        #     self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        # ), f"{self.window_size} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"

        self.endLoading()

    def get_class(self, name, number):
        trajectory = [
            "a035",
            "a017",
            "a022",
            "a029",
            "a040",
            "a005",
            "a038",
            "a051",
            "a045",
            "a055",
            "a034",
            "a037",
            "a058",
            "a039",
            "a053",
            "a027",
            "a025",
            "a009",
            "a011",
            "a013",
            "a020",
            "a057",
            "a004",
            "a028",
            "a042",
            "a049",
            "a059",
            "a014",
            "a023",
            "a056",
            "a018",
            "a015",
            "a002",
            "a016",
            "a010",
            "a030",
            "a033",
            "a052",
            "a007",
            "a046",
            "a000",
            "a001",
            "a003",
            "a050",
            "a024",
            "a054",
            "a043",
            "a006",
            "a021",
            "a047",
            "a019",
            "a036",
            "a032",
            "a044",
            "a026",
            "a031",
            "a012",
        ]
        # print(len(trajectory))
        for idx, t in enumerate(trajectory):
            if t in name:
                return [idx for _ in range(number)]

        assert False, f"Class not found {name}"

    def _load_single_file(
        self, file: Path, window_size: int, stride: int, return_wave=False
    ):
        """
        Load the data from a single file
        ---------------------------------------
        input:
            file: Path
                the file to load
            window_size: int
                the window size of the data
            stride: int
                the stride of the data
        ---------------------------------------
        return:
            dataX: torch.Tensor
                the input data
            dataY: torch.Tensor
                the output data
        """
        # check directory existence
        if not file.exists():
            print(cl.Fore.red + f'- File "{file}" does not exist' + cl.Style.reset)

        # # read the data from file
        # sensor_device = json.load(open(file.joinpath("info.json")))
        # sensor_device = sensor_device["device"]
        # f = h5py.File(file.joinpath("data.hdf5"), "r")
        # json_info = json.load(open(file.joinpath("info.json")))
        # date = json_info["date"]  # formate mm/dd/yy
        # date = datetime.datetime.strptime(date, "%m/%d/%y")
        # startFrame = int(math.ceil(json_info["start_frame"] / 2))

        # acc = np.array(json_info["imu_acce_scale"]) * (
        #     f["synced"]["linacce"][startFrame:] - np.array(json_info["imu_acce_bias"])
        # )
        # acc_norm = np.array(json_info["imu_acce_scale"]) * (
        #     f["synced"]["acce"][startFrame:] - np.array(json_info["imu_acce_bias"])
        # )
        # gyr = f["synced"]["gyro_uncalib"][startFrame:] - np.array(
        #     json_info["imu_init_gyro_bias"]
        # )
        # mag = f["synced"]["magnet"][startFrame:]
        # location = f["pose"]["tango_pos"][startFrame:]
        # orientation = f["pose"]["ekf_ori"][startFrame:]
        # init_ori = f["pose"]["tango_ori"][startFrame:]
        # timestamps = f["synced"]["time"][startFrame:]

        sensor_device_info = json.load(open(file.joinpath("info.json")))
        sensor_device = sensor_device_info["device"]

        f = h5py.File(file.joinpath("data.hdf5"), "r")
        json_info = sensor_device_info

        date = json_info["date"]  # format mm/dd/yy
        date = datetime.datetime.strptime(date, "%m/%d/%y")

        # Use the true start_frame at the raw (200 Hz) rate; downsampling happens later
        startFrame = int(json_info.get("start_frame", 0))

        # Read raw sensors
        gyro_uncalib = np.array(f["synced"]["gyro_uncalib"])  # [N,3]
        acce_uncalib = np.array(f["synced"]["acce"])  # [N,3] raw accel
        if self.GravityRemoval:
            linacce = np.array(f["synced"]["linacce"])  # [N,3] gravity-removed
        else:
            linacce = acce_uncalib

        mag_all = np.array(f["synced"]["magnet"])  # [N,3]
        tango_pos = np.array(f["pose"]["tango_pos"])  # [N,3]
        # game_ori = np.array(f["synced"]["game_rv"])  # [N,4] (x,y,z,w)
        ori_name, ori, ori_error = select_orientation_source(
            data_path=file,
            max_ori_error=20.0,
            grv_only=True if self.mode == "test" else False,
        )
        ts = np.array(f["synced"]["time"])  # [N]

        # Apply biases/scales
        gyr = gyro_uncalib - np.array(json_info["imu_init_gyro_bias"])
        # acc (model input): use linear acceleration with the same bias/scale correction as original code
        acc = np.array(json_info["imu_acce_scale"]) * (
            linacce - np.array(json_info["imu_acce_bias"])
        )
        # acc_norm helper (from raw accel, consistent with original intent)
        acc_raw_scaled = np.array(json_info["imu_acce_scale"]) * (
            acce_uncalib - np.array(json_info["imu_acce_bias"])
        )
        init_tango_ori = quaternion.quaternion(*f["pose/tango_ori"][0])

        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*json_info["start_calibration"])
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        ori_q = init_rotor * ori_q

        gyr = quaternion.from_float_array(
            np.concatenate([np.zeros([gyr.shape[0], 1]), gyr], axis=1)
        )
        acc = quaternion.from_float_array(
            np.concatenate([np.zeros([acc.shape[0], 1]), acc], axis=1)
        )
        mag_all = quaternion.from_float_array(
            np.concatenate([np.zeros([mag_all.shape[0], 1]), mag_all], axis=1)
        )
        gyr = quaternion.as_float_array(ori_q * gyr * ori_q.conj())[:, 1:]
        acc = quaternion.as_float_array(ori_q * acc * ori_q.conj())[:, 1:]
        mag_all = quaternion.as_float_array(ori_q * mag_all * ori_q.conj())[:, 1:]

        # Slice from startFrame (still at 200 Hz; decimation handled later)
        acc = acc[startFrame:]
        acc_norm = acc_raw_scaled[startFrame:]
        gyr = gyr[startFrame:]
        mag = mag_all[startFrame:]
        location = tango_pos[startFrame:]
        orientation = ori[startFrame:]
        timestamps = ts[startFrame:]

        # convert the np.float64 (seconds) to np.datetime64[ns]
        timestamps = (timestamps * 1e9).astype("datetime64[ns]")
        original_sampling_rate = 200
        newBatches = []
        jump = int(round(original_sampling_rate / self.sampling_rate))
        rag = 1 if (hasattr(self, "endToEnd") or self.mode == "test") else jump
        for i in range(rag):
            newBatch = {
                "dataX": [],
                "dataY": [],
                "dataL": [],
                "dataSteps": [],
                "classes": [],
                "encoding": [],
                "name": [],
                "index": [],
                "action": [],
                "device": [],
                "user": [],
                "mounted": [],
            }
            # get the data from 200Hz to 100Hz
            _acc = acc[i::jump, :]
            _acc_norm = acc_norm[i::jump, :]
            _gyr = gyr[i::jump, :]
            _mag = mag[i::jump, :]
            _location = location[i::jump, :]
            _orientation = orientation[i::jump, :]
            _timestamps = timestamps[i::jump]

            _orientation = R.from_quat(_orientation).inv().as_quat()
            # # rotate to world frame
            # _acc, _gyr, _mag = rotateToWorldFrame(
            #     _acc, _gyr, _mag, sample_rate=self.sampling_rate, rotation=_orientation
            # )

            # convert to tensor on cuda and cut the data
            _acc = torch.from_numpy(_acc).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            _gyr = torch.from_numpy(_gyr).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            _mag = torch.from_numpy(_mag).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            _acc_norm = torch.from_numpy(_acc_norm).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            _location = torch.from_numpy(_location).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            _orientation = torch.from_numpy(_orientation).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )

            assert _acc.shape[0] == _gyr.shape[0] == _mag.shape[0] == _location.shape[0]
            # acc, gyr, mag, location, acc_norm = self._time_series_filter(
            #     5, acc, gyr, mag, location, acc_norm
            # )

            # acc = GravityRemoval(acc=acc, location="RoNIN", rescale=False, date=date)
            _gyr = _gyr  # / np.pi
            _mag = MagneticRemoval(mag=_mag, location="RoNIN", rescale=True, date=date)[
                :, :3
            ]

            assert (
                _acc.shape[0]
                == _gyr.shape[0]
                == _mag.shape[0]
                == _location.shape[0]
                == _orientation.shape[0]
            )

            # acc_, gyr_, mag_, label = self.split_data(
            #     acc=acc,
            #     gyr=gyr,
            #     mag=mag,
            #     velocity=velocity,
            #     acceleration=acceleration,
            #     window_size=window_size,
            #     stride=stride,
            # )
            # del acc, gyr, mag, velocity
            # x = torch.cat([acc_, gyr_, mag_], dim=1)
            # y = label
            # length = torch.tensor(
            #     [window_size] * len(x), dtype=torch.int32, device="cpu"
            # )

            _acc_norm = torch.norm(_acc, dim=1)
            assert _acc_norm.shape[0] == _acc.shape[0] == _orientation.shape[0]
            # _acc, _gyr, _mag, _location, _acc_norm, _orientation, _timestamps = (
            #     self._first_motion(
            #         _acc_norm,
            #         0.98,
            #         _acc,
            #         _gyr,
            #         _mag,
            #         _location,
            #         _acc_norm,
            #         _orientation,
            #         _timestamps,
            #     )
            # )
            # if hasattr(self, "endToEnd"):
            #     self._position = _location - _location[0:1]
            #     self._orientation = (
            #         R.from_quat(_orientation).inv().as_euler("xyz", degrees=False)
            #     )

            # _orientation = self._orientation2degpersec(
            #     _orientation, sample_rate=self.sampling_rate, ts=_timestamps
            # )

            _velocity, _acceleration = self.vel_acc_generator(
                _location, sample_rate=self.sampling_rate
            )

            if return_wave:
                return _acc, _gyr, _mag, _velocity, _acceleration

            # valleys, _acc_norm = self._step_finder(_acc_norm, window_size=5)
            assert not self.useStep, "Step information is not used in RoNINDataset"
            assert _acc_norm.shape[0] == _acc.shape[0] == _orientation.shape[0]
            # acc, gyr, mag, orientation, velocity, acceleration = (
            #     self._time_series_filter(
            #         30, acc, gyr, mag, orientation, velocity, acceleration
            #     )
            # )

            # x, y, length = self.split_by_step(
            #     stepIdx=valleys,
            #     acc=_acc,
            #     gyr=_gyr,
            #     mag=_mag,
            #     orientation=_orientation,
            #     velocity=velocity,
            #     acceleration=acceleration,
            #     window_size=window_size,
            #     stride=stride,
            #     modes=self.mode,
            # )

            # assert all(
            #     a.shape[-2:] == b.shape[-2:] for a, b in zip(x, y)
            # ), "x and y must have the same shape "
            observation, label = self.select_input_output(
                acc=_acc,
                gyr=_gyr,
                mag=_mag,
                location=_location,
                velocity=_velocity,
                acceleration=_acceleration,
                orientation=_orientation,
            )
            newBatch["dataX"] = observation
            newBatch["dataY"] = label
            newBatch["dataL"] = torch.tensor(window_size)
            # newBatch["dataSteps"] = torch.tensor(valleys, dtype=torch.int32)
            # newBatch["dataSteps"] = self.get_pairIndex(
            #     stepIdx=valleys, window_size=window_size, modes=self.mode
            # )
            newBatch["dataSteps"] = []
            newBatch["classes"] = self.get_class(str(file), 1)
            newBatch["name"] = torch.tensor(self.files.index(file), dtype=torch.int32)
            # newBatch["index"] = torch.arange(len(_acc), dtype=torch.int32)
            newBatch["action"] = torch.tensor(0, dtype=torch.int32)
            sensor_device_idx = torch.tensor(
                self.device_table.index(sensor_device), dtype=torch.int32
            )
            newBatch["device"] = sensor_device_idx
            newBatch["user"] = torch.tensor(0, dtype=torch.int32)
            newBatch["mounted"] = torch.tensor(0, dtype=torch.int32)
            newBatch["size"] = _acc.shape[0] - window_size + 1

            # newBatch["encoding"] = process_Encoder(str(file) + f" {sensor_device}")
            if self.encoding:
                encoding, attention_mask = get_enconding(
                    {
                        "person": "unknown",
                        "action": "walking",
                        "device": sensor_device,
                        "mounted": "handheld",
                        "environment": "unknown",  # TODO: fill in environment
                        "dataset": "RoNIN",
                        "annotation": f"ARCore",
                    },
                    mode=self.mode,
                )
                newBatch["encoding"] = encoding
                newBatch["attention_mask"] = attention_mask
            else:
                newBatch["encoding"] = []
                newBatch["attention_mask"] = []
            newBatches.append(newBatch)

            # newBatch["dataY"].extend(y)
            # newBatch["dataL"].extend(length)
            # newBatch["dataSteps"].append(valleys)
            # newBatch["classes"].extend(self.get_class(str(file), len(x)))
            # encoding = process_Encoder(str(file) + f" {sensor_device}")
            # newBatch["encoding"].extend([encoding for _ in range(len(x))])
            # newBatch["name"].extend(
            #     [
            #         torch.tensor(self.files.index(file), dtype=torch.int32)
            #         for _ in range(len(x))
            #     ]
            # )
            # newBatch["index"].extend(list(torch.arange(len(x), dtype=torch.int32)))
            # newBatch["action"].extend([torch.tensor(0) for _ in range(len(x))])
            # sensor_device_idx = torch.tensor(self.device_table.index(sensor_device))
            # newBatch["device"].extend([sensor_device_idx for _ in range(len(x))])
            # newBatch["user"].extend([torch.tensor(0) for _ in range(len(x))])
            # newBatch["mounted"].extend([torch.tensor(0) for _ in range(len(x))])
            if self.mode == "pred":
                break

        # print file complete in green
        # print(cl.Fore.green + f"File {file.name} Done!" + cl.Style.reset)

        # assert dataX.shape[0] == dataY.shape[0]
        # assert dataX.shape[1] == 6, f"{file} has {dataX.shape[1]} columns"
        # assert dataY.shape[1] == 6, f"{file} has {dataY.shape[1]} columns"

        return newBatches
