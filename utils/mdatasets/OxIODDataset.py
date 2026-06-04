from .common_imports import *
from .mDataset import mDataset
from .utils import *


class OxIODDataset(mDataset):
    """
    Oxford Inertial Odometry Dataset
    link: http://deepio.cs.ox.ac.uk/
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
    ) -> None:
        """
        Initialize the Oxford Inertial Odometry Dataset
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
        self._IMU_HEADERS = [
            "Time",
            "attitude_roll(radians)",
            "attitude_pitch(radians)",
            "attitude_yaw(radians)",
            "rotation_rate_x(radians/s)",
            "rotation_rate_y(radians/s)",
            "rotation_rate_z(radians/s)",
            "gravity_x(G)",
            "gravity_y(G)",
            "gravity_z(G)",
            "user_acc_x(G)",
            "user_acc_y(G)",
            "user_acc_z(G)",
            "magnetic_field_x(microteslas)",
            "magnetic_field_y(microteslas)",
            "magnetic_field_z(microteslas)",
        ]

        self._VI_HEADERS = [
            "Time",
            "Header",
            "translation.x",
            "translation.y",
            "translation.z",
            "rotation.x",
            "rotation.y",
            "rotation.z",
            "rotation.w",
        ]
        # self._label = (
        #     DATASET_DICT["OxIOD_tango"]
        #     if self.keep_filters is not None and "tango" in self.keep_filters
        #     else DATASET_DICT["OxIOD"]
        # )
        if self.keep_filters is not None and "tango" in self.keep_filters:
            self._label = DATASET_DICT["OxIOD_tango"]
        elif self.skip_filters is not None and "tango" in self.skip_filters:
            self._label = DATASET_DICT["OxIOD_VICON"]
        else:
            raise ValueError(
                "You should not use OxIODDataset without specifying tango or VICON, this will break the dataset labeling"
            )

        self.mounted_table.extend(
            [
                "handbag",
                "handheld",
                "large scale",
                "multi devices",
                "multi users",
                "pocket",
                "running",
                "slow walking",
                "trolley",
                "large-scale",
            ]
        )
        # self.files = self.read_list(
        #     self.root.joinpath("lists").joinpath(f"train_list.txt")
        # )
        self.files = self.read_list(
            self.root.joinpath("lists").joinpath(f"{mode}_list.txt")
        )
        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()

        # assert (
        #     self.dataX.shape[0] == self.dataY.shape[0]
        # ), f"{self.dataX.shape} != {self.dataY.shape}"
        # assert self.dataX.shape[1] == 6, f"{self.dataX.shape[1]} != 6"
        # assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        # assert (
        #     self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        # ), f"{self.window_size} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"

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
            lines = f.readlines()
            files = []
            for line in lines:
                imu, vi = line.strip().split(",")
                # check existence of file
                assert Path(self.root).joinpath(imu).exists(), (
                    cl.Fore.red
                    + f"File {Path(self.root).joinpath(imu)} does not exist"
                    + cl.Style.reset
                )
                assert Path(self.root).joinpath(vi).exists(), (
                    cl.Fore.red
                    + f"File {Path(self.root).joinpath(vi)} does not exist"
                    + cl.Style.reset
                )
                files.append((imu, vi))
        rank_zero_info(
            cl.Fore.green
            + f'- Found "{len(files)}" files for "{self.mode}".'
            + cl.Style.reset
        )
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
        counter = 0
        # print(cl.Fore.red, self.files, cl.Style.reset)
        for imu, vi in tqdm(self.files):
            # if "nexus" in str(vi) or "tango" in str(vi):
            #     continue
            if self.skip_filters is not None:
                # skip the file in the filer keyword
                if any([f in str(vi) for f in self.skip_filters]):
                    continue

            if self.keep_filters is not None:
                # only keep the file in the filer keyword
                if not any([f in str(vi) for f in self.keep_filters]):
                    continue
            counter += 1
            # if "trolley" not in str(vi):
            #     continue
            # print(cl.Fore.yellow + f"Loading {imu} and {vi}" + cl.Style.reset)
            newBatches = self._load_single_file(
                imu, vi, window_size=self.window_size, stride=self.stride
            )
            self.Data.extend(newBatches)
            # for key, value in newBatch.items():
            #     if key not in self.Data:
            #         raise ValueError(f"Key {key} not found in Data")
            #     self.Data[key].extend(value)

            # self.classes.append(classes)
        # self.dataX = np.concatenate(self.dataX, axis=0)
        # self.dataY = np.concatenate(self.dataY, axis=0)
        # self.dataL = np.concatenate(self.dataL, axis=0)
        # self.classes = np.concatenate(self.classes, axis=0)
        rank_zero_info(cl.Fore.green + f"- Loaded {counter} files." + cl.Style.reset)
        # assert (
        #     self.dataX.shape[0] == self.dataY.shape[0] == self.dataL.shape[0]
        # ), f"{self.dataX.shape} != {self.dataY.shape}"
        # assert self.dataX.shape[1] == 6, f"{self.dataX.shape[1]} != 6"
        # assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        # assert (
        #     self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        # ), f"{self.window_size} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"

        self.endLoading()

    def get_class(self, name, number):
        activity = [
            "handbag",
            "handheld",
            "large scale",
            "multi devices",
            "multi users",
            "pocket",
            "running",
            "slow walking",
            "trolley",
        ]
        for idx, act in enumerate(activity):
            if act in name:
                return [idx for _ in range(number)]
        assert "label not found"

    def _load_single_file(
        self,
        imu,
        vi,
        window_size: int,
        stride: int,
        return_wave=False,
    ):
        def get_keyword_index(name, table):
            for idx, n in enumerate(table):
                if n == "":
                    continue
                if n in name:
                    return idx
            raise ValueError(f"Keyword not found in {name} from {table}")

        """
        Load the data from a single file
        ---------------------------------------
        input:
            imu: str
                the imu file
            vi: str
                the visual odometry file
        ---------------------------------------
        return:
            x: torch.Tensor
                the input data
            y: torch.Tensor
                the output data
        """
        df_imu = pd.read_csv(Path(self.root).joinpath(imu), names=self._IMU_HEADERS)
        df_vi = pd.read_csv(Path(self.root).joinpath(vi), names=self._VI_HEADERS)
        original_sampling_rate = 100
        # ms = 1000 // self.sampling_rate
        # # resampling the original sampling rate 100Hz to self.sampling rate
        # df_imu.index = pd.timedelta_range(
        #     start="0s", periods=len(df_imu), freq="10ms"
        # )  # 1/100 = 0.01s = 10ms
        # df_imu = df_imu.resample(f"{ms}ms").mean()
        # df_vi.index = pd.timedelta_range(
        #     start="0s", periods=len(df_vi), freq="10ms"
        # )  # 1/100 = 0.01s = 10ms
        # df_vi = df_vi.resample(f"{ms}ms").mean()

        # fill the nan value with forward fill
        df_imu = df_imu.ffill()
        df_vi = df_vi.ffill()
        if len(df_imu) != len(df_vi):
            rank_zero_info(
                cl.Fore.yellow
                + f'- This is a acceptable warning due to the misalignment from the original datasets\n"{imu}"\nand\n"{vi}"\nhave different length "{len(df_imu)}" "{len(df_vi)}"'
                + cl.Style.reset
            )
        minlen = min(len(df_imu), len(df_vi))
        df_imu = df_imu[:minlen]
        df_vi = df_vi[:minlen]

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
            if "nexus" in str(vi):
                acc = df_imu[
                    [
                        "attitude_roll(radians)",
                        "attitude_pitch(radians)",
                        "attitude_yaw(radians)",
                    ]
                ].to_numpy()[i::jump]

            else:
                acc = df_imu[
                    [
                        "user_acc_x(G)",
                        "user_acc_y(G)",
                        "user_acc_z(G)",
                    ]
                ].to_numpy()[i::jump]
                acc_norm = df_imu[
                    [
                        "gravity_x(G)",
                        "gravity_y(G)",
                        "gravity_z(G)",
                    ]
                ].to_numpy()[i::jump]
                if self.GravityRemoval:
                    acc = acc
                else:
                    acc += acc_norm
                acc_norm = np.linalg.norm((acc_norm + acc) * 9.81, axis=-1)
            gyr = df_imu[
                [
                    "rotation_rate_x(radians/s)",
                    "rotation_rate_y(radians/s)",
                    "rotation_rate_z(radians/s)",
                ]
            ].to_numpy()[i::jump]
            mag = df_imu[
                [
                    "magnetic_field_x(microteslas)",
                    "magnetic_field_y(microteslas)",
                    "magnetic_field_z(microteslas)",
                ]
            ].to_numpy()[i::jump]
            location = df_vi[
                ["translation.x", "translation.y", "translation.z"]
            ].to_numpy()[i::jump]
            rotation = df_vi[
                ["rotation.x", "rotation.y", "rotation.z", "rotation.w"]
            ].to_numpy()[i::jump]

            # ---------------------------------------------------------------------------------------

            # unit conversion
            acc = acc * 9.81 if not "nexus" in str(vi) else acc  # G to m/s^2
            gyr = gyr if not "nexus" in str(vi) else gyr / 180 * np.pi  # rad/s to deg/s
            mag = mag / 100  # microteslas to gauss

            # world frame
            acc, gyr, mag = rotateToWorldFrame(
                acc,
                gyr,
                mag,
                rotation=rotation,
                sample_rate=self.sampling_rate,
            )

            # plt.figure(figsize=(10, 4))
            # # plot the location of the trajectory in 2d plane
            # # plt.plot(location[:, 0], location[:, 1], label="Ground Truth")
            # plt.plot(np.linalg.norm(acc, axis=1), label="Acc norm")
            # # plt.plot(acc_norm, label="acc norm")

            # plt.title(f"Trajectory of {vi}")
            # plt.xlabel("Time (s)")
            # # plt.ylabel("Location (m)")
            # plt.legend()
            # plt.grid()
            # plt.tight_layout()
            # plt.show()
            # for idx, name in enumerate(["x", "y", "z"]):
            #     # assume gyro_readings[] in col5 (x-axis) and dt in seconds
            #     angle_deg = np.sum(gyr[:, idx]) * 0.01
            #     angle_rad = angle_deg / 180 * np.pi

            #     print(f"Integrated angle (rad){name}:", angle_rad)
            #     print(f"Integrated angle (deg){name}:", angle_deg)
            # c = input("Continue? [y/n]")
            # if c == "n":
            #     plt.close("all")
            #     exit()
            # plt.close("all")

            # rotation = R.from_quat(rotation).as_euler("xyz")
            # print(cl.Fore.red, "rotation to world frame is blocked", cl.Style.reset)

            # # remove gravity
            if "nexus" in str(vi):
                acc_norm = np.linalg.norm(acc, axis=1)
                if self.GravityRemoval:
                    acc = GravityRemoval(acc=acc, location="Oxford", rescale=False)

            # find the first motion
            acc, gyr, mag, location, acc_norm, rotation = self._first_motion(
                acc_norm,
                0.98,
                acc,
                gyr,
                mag,
                location,
                acc_norm,
                rotation,
            )

            # if hasattr(self, "endToEnd"):
            #     self._position = location.copy() - location[0:1]
            #     self._orientation = (
            #         R.from_quat(rotation).inv().as_euler("xyz", degrees=False)
            #     )

            valleys, smoothed_acc_norm = self._step_finder(acc_norm, window_size=5)

            # convert to tensor on cuda and cut the data
            acc = torch.from_numpy(acc).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            gyr = torch.from_numpy(gyr).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            mag = torch.from_numpy(mag).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            location = torch.from_numpy(location).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )
            rotation = torch.from_numpy(rotation).to(
                dtype=self.precision, device=self.device, non_blocking=True
            )

            velocity, acceleration = self.vel_acc_generator(
                location, sample_rate=self.sampling_rate
            )
            # rotation = self._orientation2degpersec(
            #     rotation, sample_rate=self.sampling_rate
            # )

            assert (
                acceleration.shape == velocity.shape == location.shape
            ), f"{acceleration.shape} != {velocity.shape} != {location.shape}"
            # del location

            assert acc.shape[0] == gyr.shape[0] == velocity.shape[0]
            assert acc.isfinite().all(), f"{imu} has NaN in acc"
            assert gyr.isfinite().all(), f"{imu} has NaN in gyr"
            assert velocity.isfinite().all(), f"{vi} has NaN in vel"

            observation, label = self.select_input_output(
                acc=acc,
                gyr=gyr,
                mag=mag,
                location=location,
                velocity=velocity,
                acceleration=acceleration,
                orientation=rotation,
            )
            newBatch["dataX"] = observation
            newBatch["dataY"] = label
            newBatch["dataL"] = torch.tensor(window_size)
            # newBatch["dataSteps"] = torch.tensor(valleys, dtype=torch.int32)
            newBatch["dataSteps"] = self.get_pairIndex(
                stepIdx=valleys, window_size=window_size, modes=self.mode
            )
            newBatch["classes"] = self.get_class(str(vi), 1)
            newBatch["name"] = torch.tensor(
                self.files.index((imu, vi)), dtype=torch.int32
            )
            # newBatch["index"] = torch.arange(len(acc), dtype=torch.int32)
            newBatch["action"] = torch.tensor(0, dtype=torch.int32)
            newBatch["device"] = torch.tensor(0, dtype=torch.int32)
            newBatch["user"] = torch.tensor(0, dtype=torch.int32)
            idxMount = torch.tensor(
                get_keyword_index(name=vi, table=self.mounted_table), dtype=torch.int32
            )
            newBatch["mounted"] = idxMount
            newBatch["size"] = acc.shape[0] - window_size + 1
            # split the file name at the "Oxford Inertial Odometry Dataset"
            fn = str(vi).split("Oxford Inertial Odometry Dataset")[-1].split("/")[0]
            fn = str(fn).lower()
            if self.encoding:
                encoding, attention_mask = get_enconding(
                    {
                        "person": "unknown",
                        "action": "unknown",
                        "device": "unknown",
                        "mounted": fn,
                        "environment": "unknown",  # TODO: fill in environment
                        "dataset": "OxIOD",
                        "annotation": "VICON" if "vi" in str(vi) else "tango",
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
            #     acc=acc,
            #     gyr=gyr,
            #     mag=mag,
            #     orientation=rotation,
            #     velocity=velocity,
            #     acceleration=acceleration,
            #     window_size=window_size,
            #     stride=stride,
            #     modes=self.mode,
            # )
            # label = self.get_class(vi, len(x))

            # # import matplotlib.pyplot as plt

            # # for samplex, sampley in zip(x, y):
            # #     print(sampley.shape)
            # #     plt.plot(samplex[:3].norm(dim=0).cpu().numpy(), label="acc_imu")
            # #     plt.plot(sampley[:3].norm(dim=0).cpu().numpy(), label="acc_label")
            # #     # plt.plot(samplex[3:].norm(dim=0).cpu().numpy())
            # #     # plt.plot(sampley[3:].norm(dim=0).cpu().numpy())
            # #     plt.legend()
            # #     plt.show()
            # #     c = input("Continue? [y/n]")
            # #     if c == "n":
            # #         plt.close("all")
            # #         break
            # #     plt.close("all")

            # # return x, y, length, valleys, label
            # encoding = process_Encoder(str(vi))
            # newBatch["dataX"].extend(x)
            # newBatch["dataY"].extend(y)
            # newBatch["dataL"].extend(length)
            # newBatch["dataSteps"].extend(valleys)
            # newBatch["classes"].extend(label if label is not None else [])
            # newBatch["encoding"].extend([encoding for _ in range(len(x))])
            # newBatch["name"].extend(
            #     [
            #         torch.tensor(self.files.index((imu, vi)), dtype=torch.int32)
            #         for _ in range(len(x))
            #     ]
            # )
            # newBatch["index"].extend(list(torch.arange(len(x), dtype=torch.int32)))
            # newBatch["action"].extend([torch.tensor(0) for _ in range(len(x))])
            # newBatch["device"].extend([torch.tensor(0) for _ in range(len(x))])
            # newBatch["user"].extend([torch.tensor(0) for _ in range(len(x))])
            # idxMount = torch.tensor(
            #     get_keyword_index(name=vi, table=self.mounted_table)
            # )
            # newBatch["mounted"].extend([idxMount for _ in range(len(x))])
        return newBatches
