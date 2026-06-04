from .common_imports import *
from .utils import *


def _getitem_next_seconds(self, idx):
    idx = idx * self.stride
    chunk_idx = (
        torch.searchsorted(self.chunk_index * self.stride, idx, right=True).item() - 1
    )
    ind = idx - self.chunk_index[chunk_idx] * self.stride
    # print(
    #     cl.Fore.yellow,
    #     f"idx: {idx}, chunk_idx: {chunk_idx}, ind: {ind}",
    #     cl.Style.reset,
    # )
    # print(
    #     cl.Fore.yellow,
    #     f"chunk_x_shape: {self.Data[chunk_idx]['dataX'].shape}, chunk_y_shape: {self.Data[chunk_idx]['dataY'].shape}",
    #     cl.Style.reset,
    # )
    if self.useStep == False:
        start = ind
        end = ind + self.window_size
    else:
        # search the index of the next within the range of window_size
        # print(
        #     cl.Fore.yellow,
        #     f"chunk_index: {self.chunk_index}",
        #     f"stepIdx: {len(self.Data[chunk_idx]['dataSteps'])} ind : {ind}",
        #     cl.Style.reset,
        # )
        start, end = self.Data[chunk_idx]["dataSteps"][ind]
        # print(cl.Fore.yellow, f"start: {start}, end: {end}", cl.Style.reset)

    x = self.Data[chunk_idx]["dataX"][start:end].clone().detach()
    nextX = self.Data[chunk_idx]["dataX"][end : end + self.window_size].clone().detach()
    y = self.Data[chunk_idx]["dataY"][end : end + self.window_size].clone().detach()
    mergeX = torch.cat([x, nextX], dim=-1)
    # y[:, :3] = y[:, :3] - self.Data[chunk_idx]["dataY"][start][:3]  # delta position

    # y[:, :3] = y[:, :3].cumsum(dim=0) / self.sampling_rate  # velocity to position
    # y[:, 3:6] = integrate_orientation(
    #     y[:, 3:6].swapaxes(0, 1), self.sampling_rate
    # ).swapaxes(
    #     0, 1
    # )  # delta angle to orientation
    mergeX, y = self.package_to_window_size(
        length=end - start, observation=mergeX, label=y, window_size=self.window_size
    )

    dataL = (
        self.Data[chunk_idx]["dataL"]
        if not self.useStep
        else torch.tensor(end - start, dtype=torch.int32)
    )
    if self.encoding:
        encoding = self.Data[chunk_idx]["encoding"]
        attention_mask = self.Data[chunk_idx]["attention_mask"]
        # encoding is a list randomly select one
        # encoding = (
        #     encoding[np.random.randint(0, len(encoding))].clone().detach()
        # )  # randomly select one encoding
        # set the last one with 50% and the others with equal probability
        if len(encoding) > 1:
            p = torch.ones(len(encoding)) * 0.5
            p[:-1] = p[:-1] / len(p[:-1]) * 0.5
            choice = torch.multinomial(p, 1).item()
            encoding = encoding[choice].clone().detach()
            attention_mask = attention_mask[choice].clone().detach()
        else:
            encoding = encoding[0].clone().detach()
            attention_mask = attention_mask[0].clone().detach()
    else:
        encoding = torch.tensor(0)
        attention_mask = torch.tensor(0)
    names = self.Data[chunk_idx]["name"].clone().detach().long()
    # indexs = torch.tensor(ind // self.stride, dtype=torch.long, requires_grad=False)
    indexs = (ind // self.stride).clone().detach().int()
    action = self.Data[chunk_idx]["action"].clone().detach().int()
    user = self.Data[chunk_idx]["user"].clone().detach().int()
    devices = self.Data[chunk_idx]["device"].clone().detach().int()
    mounteds = self.Data[chunk_idx]["mounted"].clone().detach().int()

    # x = self.Data["dataX"][idx].clone().detach()
    # y = self.Data["dataY"][idx].clone().detach()
    # dataL = self.Data["dataL"][idx]
    # encoding = self.Data["encoding"][idx].clone().detach()
    # names = self.Data["name"][idx].clone().detach().long()
    # indexs = self.Data["index"][idx].clone().detach().long()
    # action = self.Data["action"][idx].clone().detach().int()
    # user = self.Data["user"][idx].clone().detach().int()
    # devices = self.Data["device"][idx].clone().detach().int()
    # mounteds = self.Data["mounted"][idx].clone().detach().int()

    label = torch.tensor(self._label).int()
    # print(
    #     cl.Fore.red,
    #     f"x: {x.shape}, y: {y.shape}, dataL: {dataL}, encoding: {encoding.shape}",
    #     cl.Style.reset,
    # )
    # check the existence and the module length is not zero
    if self.transform and len(self.transform) > 0:
        mergeX, y = self.transform((mergeX, y))
    # mergeX, y = self.euler2Quaternion((mergeX, y))
    # return x, y, dataL, encoding
    # split merge x and nextX
    x = mergeX[:6]
    nextX = mergeX[6:12]

    return {
        "dataX": x,
        "nextX": nextX,
        "dataY": y,
        "dataL": dataL,
        "encoding": encoding,
        "attention_mask": attention_mask,
        "name": names,
        "index": indexs,
        "label": label,
        "action": action,
        "device": devices,
        "user": user,
        "mounted": mounteds,
        "datasets": torch.tensor(self._label).long(),
    }


def endLoadingMinusWindowSize(self):
    """
    End the loading process
    ----------------------------------------
    input:
        None
    ----------------------------------------
    return:
        None
    """
    self.chunk_index = (
        [i["size"] - self.window_size for i in self.Data]
        if self.useStep == False
        else [len(i["dataSteps"]) for i in self.Data]
    )
    self.chunk_index = torch.tensor(self.chunk_index) // self.stride
    self.chunk_index = self.chunk_index.cumsum(dim=0)
    self.chunk_index = torch.cat([torch.tensor([0]), self.chunk_index], dim=0)
    for key, valule in DATASET_DICT.items():
        if valule == self._label:
            self._name = key
            break

    # Prepare info dictionary
    info_to_print = {
        "Dataset": self._name,
        "Mode": self.mode,
        "Window size": f"{self.window_size}",
        "Stride": f"{self.stride}",
        "Sampling rate": f"{self.sampling_rate} Hz",
        "Effective Duration": f"{self.window_size/self.sampling_rate:.2f}s",
        "Use step": self.useStep,
        "Number of sequences": len(self.files),
        "Total number of samples": self.chunk_index[-1].item(),
        "next_window": self.next_window,
    }

    # Print aligned information
    self.print_aligned_info(info_to_print, f"Dataset Summary")
    # self.save()


class mDataset(torch.utils.data.Dataset):
    """
    Base class for all the dataset
    ---------------------------------------
    input:
        root: str
            the root directory of the dataset
        mode: str
            the mode of the dataset (train, val, test, pred)
        transform: torch.nn.modules.container.Sequential
            the transformation of the dataset
        window_size: int
            the window size of the dataset
        stride: int
            the stride of the dataset
        keep_filters: list
            the list of the keyword to keep
        skip_filters: list
            the list of the keyword to skip
    ---------------------------------------
    implement the following function
    1. vel_acc_generator
    2. split_data
    3. _time_series_filter
    4. _step_finder
    5. get_path_format
    6. check_existence
    7. save
    8. load
    9. __len__
    10. __getitem__
    ---------------------------------------
    need to implement the following function
    1. read_list
    2. load_files
    3. _load_single_file

    """

    def __init__(
        self,
        root,
        useStep=True,
        mode="train",
        transform: torch.nn.modules.container.Sequential = None,
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
        self.useStep = useStep
        self.next_window = next_window
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        # self.transform = videoTransform(transform)
        self.window_size = window_size
        if "test" in self.mode and self.useStep:
            self.stride = 1  # for test mode, we use stride 1 to get all the steps
        elif "test" in self.mode:
            self.stride = self.window_size
        else:
            self.stride = stride

        self.sampling_rate = sampling_rate
        self.keep_filters = keep_filters
        self.skip_filters = skip_filters
        self.GravityRemoval = GravityRemoval
        self.encoding = encoding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = precision
        self.step_gap = []

        self.action_table = ["unknown"]
        self.device_table = ["unknown"]
        self.user_table = ["unknown"]
        self.mounted_table = ["unknown"]
        # self.euler2Quaternion = eulerToQuaternion(
        #     {
        #         "probability": 1.0,
        #     }
        # )

    def __len__(self):
        """
        check the length of the dataset and return the length
        """
        # assert len(self.Data["dataX"]) == len(self.Data["dataY"])
        # return len(self.Data["dataX"])
        # if self.mode != "test":
        return self.chunk_index[-1]
        # else:
        #     chunk_size = torch.diff(self.chunk_index, prepend=torch.tensor([0]))
        #     chunk_size = chunk_size // self.window_size
        #     chunk_size = chunk_size.cumsum(dim=0)
        #     self.chunk_index = chunk_size
        #     return self.chunk_index[-1]

    # def __getitem__(self, idx):
    #     """
    #     get the item from the dataset and convert it to tensor
    #     if transform is not None, apply the transform to the data
    #     ---------------------------------------
    #     input:
    #         idx: int
    #             the index of the dataset
    #     ---------------------------------------
    #     return:
    #         x: torch.Tensor
    #             the input data
    #         y: torch.Tensor
    #             the output data
    #     """
    #     if isinstance(self.dataX[idx], np.ndarray):
    #         x = torch.from_numpy(self.dataX[idx]).float()
    #         y = torch.from_numpy(self.dataY[idx]).float()
    #     else:
    #         x = self.dataX[idx].float().clonse().detach()
    #         y = self.dataY[idx].float().clonse().detach()

    #     if self.transform is not None:
    #         x, y = self.transform((x, y))

    #     return x[:6], y, self.dataL[idx]
    #     # return x, y

    def __getitem__(self, idx):
        if self.next_window:
            return _getitem_next_seconds(self, idx)
        idx = idx * self.stride
        chunk_idx = (
            torch.searchsorted(self.chunk_index * self.stride, idx, right=True).item()
            - 1
        )
        ind = idx - self.chunk_index[chunk_idx] * self.stride
        # print(
        #     cl.Fore.yellow,
        #     f"idx: {idx}, chunk_idx: {chunk_idx}, ind: {ind}",
        #     cl.Style.reset,
        # )
        # print(
        #     cl.Fore.yellow,
        #     f"chunk_x_shape: {self.Data[chunk_idx]['dataX'].shape}, chunk_y_shape: {self.Data[chunk_idx]['dataY'].shape}",
        #     cl.Style.reset,
        # )
        if self.useStep == False:
            start = ind
            end = ind + self.window_size
        else:
            # search the index of the next within the range of window_size
            # print(
            #     cl.Fore.yellow,
            #     f"chunk_index: {self.chunk_index}",
            #     f"stepIdx: {len(self.Data[chunk_idx]['dataSteps'])} ind : {ind}",
            #     cl.Style.reset,
            # )
            start, end = self.Data[chunk_idx]["dataSteps"][ind]
            # print(cl.Fore.yellow, f"start: {start}, end: {end}", cl.Style.reset)

        x = self.Data[chunk_idx]["dataX"][start:end].clone().detach()
        y = self.Data[chunk_idx]["dataY"][start:end].clone().detach()
        # y[:, :3] = y[:, :3] - self.Data[chunk_idx]["dataY"][start][:3]  # delta position

        # y[:, :3] = y[:, :3].cumsum(dim=0) / self.sampling_rate  # velocity to position
        # y[:, 3:6] = integrate_orientation(
        #     y[:, 3:6].swapaxes(0, 1), self.sampling_rate
        # ).swapaxes(
        #     0, 1
        # )  # delta angle to orientation
        x, y = self.package_to_window_size(
            length=end - start, observation=x, label=y, window_size=self.window_size
        )

        dataL = (
            self.Data[chunk_idx]["dataL"]
            if not self.useStep
            else torch.tensor(end - start, dtype=torch.int32)
        )
        if self.encoding:
            encoding = self.Data[chunk_idx]["encoding"]
            attention_mask = self.Data[chunk_idx]["attention_mask"]
            # encoding is a list randomly select one
            # encoding = (
            #     encoding[np.random.randint(0, len(encoding))].clone().detach()
            # )  # randomly select one encoding
            # set the last one with 50% and the others with equal probability
            if len(encoding) > 1:
                p = torch.ones(len(encoding)) * 0.5
                p[:-1] = p[:-1] / len(p[:-1]) * 0.5
                choice = torch.multinomial(p, 1).item()
                encoding = encoding[choice].clone().detach()
                attention_mask = attention_mask[choice].clone().detach()
            else:
                encoding = encoding[0].clone().detach()
                attention_mask = attention_mask[0].clone().detach()
        else:
            encoding = torch.tensor(0)
            attention_mask = torch.tensor(0)
        names = self.Data[chunk_idx]["name"].clone().detach().long()
        # indexs = torch.tensor(ind // self.stride, dtype=torch.long, requires_grad=False)
        indexs = (ind // self.stride).clone().detach().int()
        action = self.Data[chunk_idx]["action"].clone().detach().int()
        user = self.Data[chunk_idx]["user"].clone().detach().int()
        devices = self.Data[chunk_idx]["device"].clone().detach().int()
        mounteds = self.Data[chunk_idx]["mounted"].clone().detach().int()

        # x = self.Data["dataX"][idx].clone().detach()
        # y = self.Data["dataY"][idx].clone().detach()
        # dataL = self.Data["dataL"][idx]
        # encoding = self.Data["encoding"][idx].clone().detach()
        # names = self.Data["name"][idx].clone().detach().long()
        # indexs = self.Data["index"][idx].clone().detach().long()
        # action = self.Data["action"][idx].clone().detach().int()
        # user = self.Data["user"][idx].clone().detach().int()
        # devices = self.Data["device"][idx].clone().detach().int()
        # mounteds = self.Data["mounted"][idx].clone().detach().int()

        label = torch.tensor(self._label).int()
        # print(
        #     cl.Fore.red,
        #     f"x: {x.shape}, y: {y.shape}, dataL: {dataL}, encoding: {encoding.shape}",
        #     cl.Style.reset,
        # )
        # check the existence and the module length is not zero
        if self.transform and len(self.transform) > 0:
            x, y = self.transform((x, y))
        # return x, y, dataL, encoding
        return {
            "dataX": x,
            "dataY": y,
            "dataL": dataL,
            "encoding": encoding,
            "attention_mask": attention_mask,
            "name": names,
            "index": indexs,
            "label": label,
            "action": action,
            "device": devices,
            "user": user,
            "mounted": mounteds,
            "datasets": torch.tensor(self._label).long(),
        }

    # @overrides
    # def __getitem__(self, idx):
    #     x = self.Data["dataX"][idx].clone().detach()
    #     y = self.Data["dataY"][idx].clone().detach()
    #     dataL = self.Data["dataL"][idx]
    #     encoding = self.Data["encoding"][idx].clone().detach()
    #     names = self.Data["name"][idx].clone().detach().long()
    #     indexs = self.Data["index"][idx].clone().detach().long()
    #     action = self.Data["action"][idx].clone().detach().int()
    #     user = self.Data["user"][idx].clone().detach().int()
    #     devices = self.Data["device"][idx].clone().detach().int()
    #     mounteds = self.Data["mounted"][idx].clone().detach().int()

    #     label = torch.tensor(self._label).int()
    #     # check the existence and the module length is not zero
    #     if self.transform and len(self.transform) > 0:
    #         x, y = self.transform((x, y))

    #     # return x, y, dataL, encoding
    #     return {
    #         "dataX": x,
    #         "dataY": y,
    #         "dataL": dataL,
    #         "encoding": encoding,
    #         "name": names,
    #         "index": indexs,
    #         "label": label,
    #         "action": action,
    #         "device": devices,
    #         "user": user,
    #         "mounted": mounteds,
    #     }

    def get_path_format(self):
        """
        get the path format of the dataset
        ---------------------------------------
        input:
            None
        ---------------------------------------
        return:
            str
                the path format of the dataset stored in the root directory
        """
        return f"{self.mode}_{self.window_size}_{self.stride}_dataset"

    def check_existence(self):
        """
        check the existence of the dataset
        ---------------------------------------
        input:
            None
        ---------------------------------------
        return:
            bool
                True if the dataset exists, False otherwise
        """
        return Path(self.root).joinpath(self.get_path_format()).exists()

    def save(self):
        """
        save the dataset to the root directory
        ----------------------------------------
        input:
            None
        ----------------------------------------
        return:
            None
        """
        # path = Path(self.root).joinpath(self.get_path_format())
        # path.mkdir(parents=True, exist_ok=False)
        # torch.save(
        #     self.dataX,
        #     path.joinpath("dataX.pth"),
        #     pickle_protocol=pickle.HIGHEST_PROTOCOL,
        # )
        # torch.save(
        #     self.dataY,
        #     path.joinpath("dataY.pth"),
        #     pickle_protocol=pickle.HIGHEST_PROTOCOL,
        # )
        pass

    def load(self):
        """
        load the dataset from the root directory
        ----------------------------------------
        input:
            None
        ----------------------------------------
        return:
            None
        """
        raise NotImplementedError("This function is not completed yet.")
        path = Path(self.root).joinpath(self.get_path_format())
        print(cl.Fore.red + f"Shortcut" + cl.Style.reset)
        print(cl.Fore.red + f"Shortcut" + cl.Style.reset)
        rank_zero_info(cl.Fore.yellow + f"Loading {path}" + cl.Style.reset)
        self.dataX = torch.load(
            path.joinpath("dataX.pth"), mmap=True, weights_only=False
        )
        self.dataY = torch.load(
            path.joinpath("dataY.pth"), mmap=True, weights_only=False
        )

    def split_data(self, acc, gyr, mag, velocity, acceleration, window_size, stride):
        """
        split the data into window_size with stride
        ----------------------------------------
        input:
            acc: torch.Tensor
                the acceleration data
            gyr: torch.Tensor
                the gyroscope data
            vel: torch.Tensor
                the velocity data or the label of data
            window_size: int
                the window size of the data
            stride: int
                the stride of the data
        ----------------------------------------
        return:
            accSplit: torch.Tensor
                the acceleration data split into window_size with stride
            gyrSplit: torch.Tensor
                the gyroscope data split into window_size with stride
            prevel: torch.Tensor
                the previous velocity data (or label) split into window_size with stride
            label: torch.Tensor
                the label data split into window_size with stride
        """
        # acc, gyr, mag, velocity, acceleration = self._time_series_filter(
        #     5, acc, gyr, mag, velocity, acceleration
        # )
        assert (
            acc.shape[0] == gyr.shape[0] == mag.shape[0] == velocity.shape[0]
        ), f"{acc.shape} != {gyr.shape} != {mag.shape} != {velocity.shape}"
        zeros_front = torch.zeros((window_size, 3))
        prevel = torch.cat([zeros_front, velocity], dim=0)

        # use window_size to split the data with stride
        accSplit = acc.unfold(0, window_size, stride)
        gyrSplit = gyr.unfold(0, window_size, stride)
        magSplit = mag.unfold(0, window_size, stride)
        prevel = prevel.unfold(0, window_size, stride)[: accSplit.shape[0]]
        label = acceleration.unfold(0, window_size, stride)
        assert (
            accSplit.shape[0] == gyrSplit.shape[0] == prevel.shape[0] == label.shape[0]
        )

        return accSplit, gyrSplit, magSplit, label

    def read_list(self):
        """
        need to implement the function to read the list of the dataset
        """
        raise NotImplementedError

    def load_files(self):
        """
        need to implement the function to load the files
        """
        raise NotImplementedError

    def _load_single_file(self):
        """
        need to implement the function to load the single file
        """
        raise NotImplementedError

    def vel_acc_generator(self, location, sample_rate, ts=None):
        """
        Generate the velocity and acceleration from the location data
        ----------------------------------------
        input:
            location: torch.Tensor
                the location data
            sample_rate: float
                the sample rate of the data
        ----------------------------------------
        return:
            velocity: torch.Tensor
                the velocity data
            acceleration: torch.Tensor
                the acceleration data
        """
        assert (
            location.ndim == 2 and location.shape[1] == 3
        ), "Expected location to be [T, 3]"
        # check location is tensor or numpy
        if isinstance(location, np.ndarray):
            location = torch.from_numpy(location).float()
        else:
            location = location.clone().float()
        if ts is None:
            sample_rate = torch.tensor(
                [sample_rate], dtype=torch.float32, device=location.device
            )
        else:
            dt = np.array(ts)
            dt = np.diff(
                dt, prepend=dt[0]
            )  # prepend the first element to make the length the same
            dt = dt / np.timedelta64(1, "s")  # convert from datetime64[s] to seconds
            sample_rate = 1 / torch.tensor(
                dt, dtype=torch.float32, device=location.device
            )
            sample_rate = sample_rate.unsqueeze(1)
            # replace the inf with the mean of the sample_rate
            finit = torch.isfinite(sample_rate)
            sample_rate[~finit] = sample_rate[finit].mean()
            # print(f"Sample rate for vel acc converting : {sample_rate.mean()} Hz")

        location = location - location[:1, :]
        velocity = torch.diff(location, dim=0, prepend=location[:1]) * sample_rate

        acceleration = (
            torch.diff(velocity, dim=0, prepend=torch.zeros_like(velocity[:1]))
            * sample_rate
        )

        """
        already checked please reference to the file test.ipynb
        with restoreation concept check
        """
        # assert (
        #     (location - location[:1, :]) == torch.cumsum(velocity, dim=0) / sample_rate
        # ).all()
        # assert (
        #     (location - location[:1, :])
        #     == torch.cumsum((torch.cumsum(velocity, dim=0) / sample_rate), dim=0)
        #     / sample_rate
        # ).all()
        assert (
            acceleration.shape == velocity.shape == location.shape
        ), f"{acceleration.shape} != {velocity.shape} != {location.shape}"
        assert torch.isfinite(velocity).all(), "Non-finite values in velocity"
        return velocity, acceleration

    def _time_series_filter(self, window_size=5, *args):
        """
        Apply the time series filter to the data
        ----------------------------------------
        input:
            window_size: int
                the window size of the filter
            *args: torch.Tensor
                the data to filter
        ----------------------------------------
        return:
            result: list
                the filtered data
        """
        size = len(args[0])
        weights = torch.ones((window_size)) / window_size
        weights = weights.reshape(1, 1, window_size)
        result = []
        for arg in args:
            filter_data = arg
            # check tensor or numpy
            if isinstance(filter_data, np.ndarray):
                filter_data = torch.from_numpy(arg).float()
            from scipy.signal import savgol_filter

            filter_data = savgol_filter(
                arg.to("cpu", non_blocking=True).numpy(),
                window_length=window_size,
                axis=0,
                polyorder=1,
                mode="nearest",
            )
            filter_data = torch.from_numpy(filter_data).float()
            assert (
                arg.shape == filter_data.shape
            ), f"The shape is not the same after savgol filter {arg.shape} != {filter_data.shape}"

            if not isinstance(filter_data, torch.Tensor):
                filter_data = filter_data.to("cpu", non_blocking=True).numpy()
            filter_data = (
                torch.nn.functional.conv1d(
                    torch.nn.functional.pad(
                        filter_data.swapaxes(0, 1).unsqueeze(1),
                        (window_size // 2, window_size // 2),
                        mode="replicate",
                    ),
                    weights.to(filter_data.device),
                    stride=1,
                )
                .squeeze(1)
                .swapaxes(0, 1)
            )
            filter_data = filter_data[: arg.shape[0]]
            assert (
                arg.shape == filter_data.shape
            ), f"The shape is not the same after moving average filter {arg.shape} != {filter_data.shape}"
            result.append(filter_data)

        return result

    def _step_finder(self, acc_norm, window_size=5):
        """
        Find the step from the acceleration data
        ----------------------------------------

        """
        # for every two end points, find the maximum value
        # repeat the pair
        # Smoothing the data to reduce noise (using a moving average filter)
        if isinstance(acc_norm, torch.Tensor):
            acc_norm = acc_norm.to("cpu", non_blocking=True).numpy()
        smoothed_acc_norm = signal.savgol_filter(acc_norm, window_size, 2)

        # Finding the valley points (local minima)
        # valleys, _ = signal.find_peaks(-smoothed_acc_norm)
        _, valleys = find_steps(smoothed_acc_norm)

        # Adjusting the indices to match the original data length
        valleys += (window_size - 1) // 2

        return valleys, smoothed_acc_norm

    def _first_motion(self, acc_norm, threshold=0.5, *args):
        if isinstance(acc_norm, np.ndarray):
            acc_norm = torch.from_numpy(acc_norm).float()
        # std = acc_norm.std()
        # mean = acc_norm.mean()
        # idx = torch.where(acc_norm > (std * 2 + mean))[0][0]

        # diff = torch.diff(acc_norm)
        # # find the first point reach the threshold
        # idx = torch.where(diff > threshold)[0][0]
        # # return all the data after the first motion

        percent = torch.quantile(acc_norm, threshold)
        idx = torch.where(acc_norm > percent)[0][0]

        result = []
        for arg in args:
            result.append(arg[idx:])

        return result

    def sequence_to_frames(self, batch, mode, max_length=MAX_FRAME_LENGTH):
        """
        Convert a batch of sequences into frames for training.
        ----------------------------------------
        """
        x, y, lengths = batch

        X, Y, L = [], [], []
        # if mode != "test":
        if False:
            for idx in range(len(lengths)):
                for i in range(1, max_length + 1):
                    _x = torch.stack(x[idx : idx + i], dim=0)
                    _y = torch.stack(y[idx : idx + i], dim=0)
                    _length = torch.tensor(lengths[idx : idx + i])
                    X.append(_x), Y.append(_y), L.append(_length)
        else:
            for idx in range(len(lengths)):
                i = min(max_length, len(lengths) - idx)
                _x = torch.stack(x[idx : idx + i], dim=0)
                _y = torch.stack(y[idx : idx + i], dim=0)
                _length = torch.tensor(lengths[idx : idx + i])
                X.append(_x), Y.append(_y), L.append(_length)
        return X, Y, L

    def single_sequence_all_combination(
        self,
        stepIdx,
        observation,
        label,
        window_size,
        stride,
        modes="train",
    ):
        assert np.all(np.diff(stepIdx) > 0), "stepIdx is not strictly increasing"
        assert np.all(
            stepIdx < observation.shape[0]
        ), "stepIdx is out of range of the data"
        assert (
            observation.shape[0] == label.shape[0]
        ), f"{observation.shape} != {label.shape}"

        x = []
        y = []
        length = []
        stepIdx = torch.from_numpy(stepIdx)
        combination = torch.combinations(stepIdx, r=2)
        diffs = torch.diff(combination, dim=1).abs()
        valid = (diffs <= window_size).squeeze(1)
        filtered_stepIdx = combination[valid]

        for i, j in filtered_stepIdx:
            o = observation[i:j]
            l = label[i:j]
            assert o.shape[0] == l.shape[0], f"{o.shape} != {l.shape}"

            o, l = self.package_to_window_size(j - i, o, l, window_size)
            assert o.shape == l.shape, f"{o.shape} != {l.shape}"
            assert o.shape[1] == window_size, f"{o.shape[1]} != {window_size}"

            x.append(o)
            y.append(l)
            length.append(int(j - i))

        assert len(x) == len(y) == len(length), f"{len(x)} != {len(y)} != {len(length)}"
        return x, y, length

    def find_next_index(self, startIdx, idxList, window_size):
        idxList = idxList.copy()
        idxList -= idxList[startIdx]
        # find the largest number in idxList that is less than window_size
        nextIdx = np.searchsorted(idxList, window_size, side="right") - 1
        # print(f"startIdx: {startIdx}, idxList: {idxList}, nextIdx: {nextIdx}")
        return int(nextIdx)

    def single_sequence_split_by_step_withOnlyIndex(
        self,
        stepIdx,
        window_size,
        stride,
        modes="train",
        only_first=False,
    ):
        assert np.all(
            np.diff(stepIdx) > 0
        ), f"stepIdx is not strictly increasing {stepIdx}"
        endIdx = 0
        IndexPair = []
        for i in range(len(stepIdx) - 1):
            i = endIdx
            endIdx = self.find_next_index(i, stepIdx, window_size)
            if (endIdx == i) or ((stepIdx[endIdx] - stepIdx[i]) > window_size):
                continue

            IndexPair.append((stepIdx[i], stepIdx[endIdx]))
            if only_first:
                break
        return IndexPair

    def get_pairIndex(self, stepIdx, window_size, modes):
        result = []
        if modes != "test":
            for i in range(1, len(stepIdx)):
                result.extend(
                    self.single_sequence_split_by_step_withOnlyIndex(
                        stepIdx[i:], window_size, self.stride, modes, only_first=True
                    )
                )
        else:
            result.extend(
                self.single_sequence_split_by_step_withOnlyIndex(
                    stepIdx, window_size, self.stride, modes, only_first=False
                )
            )

        return result

    def single_sequence_split_by_step(
        self,
        stepIdx,
        observation,
        label,
        window_size,
        stride,
        modes="train",
        only_first=False,
    ):
        assert np.all(
            np.diff(stepIdx) > 0
        ), f"stepIdx is not strictly increasing {stepIdx}"
        assert np.all(
            stepIdx <= observation.shape[0]
        ), f"stepIdx is out of range of the data {stepIdx} < {observation.shape[0]}"
        assert (
            observation.shape[0] == label.shape[0]
        ), f"{observation.shape} != {label.shape}"
        observation = observation.cpu()
        label = label.cpu()
        x = []
        y = []
        length = []
        endIdx = 0

        for i in range(len(stepIdx) - 1):
            i = endIdx
            endIdx = self.find_next_index(i, stepIdx, window_size)
            if (endIdx == i) or ((stepIdx[endIdx] - stepIdx[i]) > window_size):
                continue
            # print(
            #     cl.Fore.yellow,
            #     f"{stepIdx[i]} : {stepIdx[endIdx]}",
            #     cl.Style.reset,
            # )

            o = observation[stepIdx[i] : stepIdx[endIdx]]
            l = label[stepIdx[i] : stepIdx[endIdx]]
            assert o.shape[0] == l.shape[0], f"{o.shape} != {l.shape}"

            o, l = self.package_to_window_size(
                stepIdx[endIdx] - stepIdx[i], o, l, window_size
            )
            assert o.shape == l.shape, f"{o.shape} != {l.shape}"
            assert o.shape[1] == window_size, f"{o.shape[1]} != {window_size}"

            x.append(o)
            y.append(l)
            length.append(int(stepIdx[endIdx] - stepIdx[i]))
            if only_first:
                break

        assert len(x) == len(y) == len(length), f"{len(x)} != {len(y)} != {len(length)}"
        return x, y, length

    def chunklize(self, stepIdx, threshold=120):
        """
        Chunk the stepIdx where the gap is larger than a threshold
        ----------------------------------------
        input:
            stepIdx: np.ndarray
                the step index
        ----------------------------------------
        return:
            chunks: list[list[int]]
                the list of chunks, where each chunk is a list of step indices
        """
        chunks = []
        diff = np.diff(stepIdx)
        split_point = (
            np.where(diff > threshold)[0] + 1
        )  # +1 to account for the diff shift
        split_indices = np.concatenate(([0], split_point, [len(stepIdx)]))
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            chunks.append(np.array(stepIdx[start:end]))
        return chunks

    def single_step_each_frame(
        self,
        stepIdx,
        observation,
        label,
        window_size,
        stride,
        modes="train",
    ):
        assert np.all(np.diff(stepIdx) > 0), "stepIdx is not strictly increasing"
        assert np.all(
            stepIdx < observation.shape[0]
        ), "stepIdx is out of range of the data"
        assert (
            observation.shape[0] == label.shape[0]
        ), f"{observation.shape} != {label.shape}"

        x = []
        y = []
        length = []

        # check the step index gap, where it is smaller the the window_size crop  the observation and label accrodingly
        for i in range(len(stepIdx) - 1):
            if stepIdx[i + 1] - stepIdx[i] > window_size:
                continue
            o = observation[stepIdx[i] : stepIdx[i + 1]]
            l = label[stepIdx[i] : stepIdx[i + 1]]

            o, l = self.package_to_window_size(
                stepIdx[i + 1] - stepIdx[i], o, l, window_size
            )
            assert o.shape == l.shape, f"{o.shape} != {l.shape}"
            assert o.shape[1] == window_size, f"{o.shape[1]} != {window_size}"

            x.append(o)
            y.append(l)
            length.append(int(stepIdx[i + 1] - stepIdx[i]))

        assert len(x) == len(y) == len(length), f"{len(x)} != {len(y)} != {len(length)}"
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        length = torch.tensor(length)
        return x, y, length

    def convert_to_step_integration(self, stepIdx, acc):
        velocity = torch.zeros_like(acc)
        for i in range(len(stepIdx) - 1):
            velocity[stepIdx[i] : stepIdx[i + 1]] = (
                acc[stepIdx[i] : stepIdx[i + 1]].cumsum(dim=-1) * 0.01
            )
        return velocity

    def split_by_step(
        self,
        stepIdx,
        acc,
        gyr,
        mag,
        velocity,
        acceleration,
        orientation,
        window_size,
        stride,
        modes="train",
        num_frames=MAX_FRAME_LENGTH,
    ):
        """
        Split the data by step
        ----------------------------------------
        input:
            stepIdx: np.ndarray
                the step index
            acc: torch.Tensor
                the acceleration data
            gyr: torch.Tensor
                the gyroscope data
            vel: torch.Tensor
                the velocity data
        ----------------------------------------
        output:
            accSplit: torch.Tensor
                the acceleration data split by step
            gyrSplit: torch.Tensor
                the gyroscope data split by step
            velSplit: torch.Tensor
                the velocity data split by step
        """
        # check the stepIdx is strictly increasing
        assert np.all(np.diff(stepIdx) > 0), "stepIdx is not strictly increasing"
        # chekc the indeIdx is within the range of the data
        assert np.all(stepIdx < acc.shape[0]), "stepIdx is out of range of the data"
        # check all data are the same length
        assert (
            acc.shape[0]
            == gyr.shape[0]
            == mag.shape[0]
            == velocity.shape[0]
            == acceleration.shape[0]
            == orientation.shape[0]
        ), f"{acc.shape} != {gyr.shape} != {mag.shape} != {velocity.shape} != {acceleration.shape} != {orientation.shape}"
        self.step_gap.extend(np.diff(stepIdx).tolist())
        # psudo_v = self.convert_to_step_integration(stepIdx, acc)
        observation = torch.cat([acc, gyr], dim=1)
        # label = acceleration
        label = torch.cat(
            [velocity, orientation.to(device=velocity.device, non_blocking=True)], dim=1
        )

        # check the length of the data is the same
        assert observation.shape[0] == label.shape[0]
        chunks = self.chunklize(stepIdx, threshold=120)

        X = []
        Y = []
        L = []

        # if modes != "test":
        #     for c in chunks:
        #         for i in range(0, len(c), stride):
        #             x, y, length = self.single_sequence_split_by_step(
        #                 stepIdx=c[i:] - c[i],
        #                 observation=observation[c[i] :],
        #                 label=label[c[i] :],
        #                 window_size=window_size,
        #                 stride=stride,
        #                 only_first=True,
        #             )
        #             # x, y, length = self.sequence_to_frames((x, y, length), mode=modes)
        #             X.extend(x)
        #             Y.extend(y)
        #             L.extend(length)

        # else:
        #     for c in chunks:
        #         x, y, length = self.single_sequence_split_by_step(
        #             stepIdx=c - c[0],
        #             observation=observation[c[0] :],
        #             label=label[c[0] :],
        #             window_size=window_size,
        #             stride=stride,
        #         )
        #         X.extend(x)
        #         Y.extend(y)
        #         L.extend(length)

        if modes != "test":
            # if False:
            for i in range(0, len(stepIdx), stride):
                x, y, length = self.single_sequence_split_by_step(
                    stepIdx=stepIdx[i:] - stepIdx[i],
                    observation=observation[stepIdx[i] :],
                    label=label[stepIdx[i] :],
                    window_size=window_size,
                    stride=stride,
                    only_first=True,
                )
                # x, y, length = self.sequence_to_frames((x, y, length), mode=modes)
                X.extend(x)
                Y.extend(y)
                L.extend(length)
            # x, y, length = self.single_sequence_all_combination(
            #     stepIdx=stepIdx,
            #     observation=observation,
            #     label=label,
            #     window_size=window_size,
            #     stride=stride,
            # )
            # X.extend(x)
            # Y.extend(y)
            # L.extend(length)
        else:
            x, y, length = self.single_sequence_split_by_step(
                stepIdx=stepIdx,
                observation=observation,
                label=label,
                window_size=window_size,
                stride=stride,
            )
            X.extend(x)
            Y.extend(y)
            L.extend(length)
        assert len(X) == len(Y) == len(L), f"{len(X)} != {len(Y)} != {len(L)}"

        # X = torch.stack(X, dim=0)
        # Y = torch.stack(Y, dim=0)
        # L = torch.tensor(L)

        # assert (
        #     X.shape[-1] == Y.shape[-1] == window_size
        # ), f"{X.shape} != {Y.shape} != {window_size}"

        return X, Y, L

    def package_to_window_size(self, length, observation, label, window_size):
        assert length <= window_size, f"{length} > {window_size}"
        observation = observation.swapaxes(0, 1)
        label = label.swapaxes(0, 1)

        # mean_o = observation.mean(dim=-1, keepdim=True)
        # std_o = observation.std(dim=-1, keepdim=True)
        # mean_l = label.mean(dim=-1, keepdim=True)
        # std_l = label.std(dim=-1, keepdim=True)
        # observation = (observation - mean_o) / std_o
        # label = (label - mean_l) / std_l

        # max_o = observation.max(dim=-1, keepdim=True)[0]
        # min_o = observation.min(dim=-1, keepdim=True)[0]
        # max_l = label.max(dim=-1, keepdim=True)[0]
        # min_l = label.min(dim=-1, keepdim=True)[0]
        # observation = (observation - min_o) / (max_o - min_o)
        # label = (label - min_l) / (max_l - min_l)

        # input (c, l) output (c, window_size)
        observation = F.pad(observation, (0, window_size - length), "constant", 0)
        label = F.pad(label, (0, window_size - length), "constant", 0)

        return observation, label

    def get_sequence(self, idx, mode="test"):
        """
        get the sequence of the data
        ----------------------------------------
        input:
            idx: int
                the index of the sequence
        ----------------------------------------
        output:
            x: torch.Tensor
                the input data
            y: torch.Tensor
                the output data
        """

        # check self type, if self is OIODDataset, then return the sequence
        if isinstance(self, OxIODDataset):
            files = self.files
            # remove
            if self.skip_filters is not None:
                if len(self.skip_filters) > 0:
                    files = [
                        (i, v)
                        for i, v in files
                        if all(key not in str(v) for key in self.skip_filters)
                    ]

            if self.keep_filters is not None:
                if len(self.keep_filters) > 0:
                    files = [
                        (i, v)
                        for i, v in files
                        if any(key in str(v) for key in self.keep_filters)
                    ]
            idx %= len(files)
            file = files[idx]
            imu, vi = file
            file = vi
            rank_zero_info(
                cl.Fore.green + f'- Loading "{imu}" and "{vi}"' + cl.Style.reset
            )
            x, y, length = self._load_single_file(
                imu, vi, window_size=self.window_size, stride=self.window_size
            )
        else:
            files = self.files
            file = files[idx]
            rank_zero_info(cl.Fore.green + f'- Loading "{file}"' + cl.Style.reset)
            x, y, length = self._load_single_file(
                file, window_size=self.window_size, stride=self.window_size
            )
        x = torch.cat([x], dim=0)[:, :6]
        y = torch.cat([y], dim=0)[:, :6]
        length = torch.cat([length], dim=0)

        assert x.shape[0] == y.shape[0] == length.shape[0]
        # print(x.shape, y.shape)
        # create lamda dataset from x, t and return the dataset
        dataset = torch.utils.data.TensorDataset(x, y, length)
        return dataset, file

    # def _orientation2degpersec(self, orientation, sample_rate, ts=None):
    #     """
    #     orientation: torch.Tensor
    #         the orientation data
    #         shape: (N, 4) in the form of (x, y, z, w)

    #     return: torch.Tensor
    #         the orientation in radian per second
    #         shape: (N, 3) in the form of (x, y, z)
    #     """
    #     # if ts is not None:
    #     #     average_sample_rate = len(ts) / ((ts[-1] - ts[0]) / np.timedelta64(1, "s"))
    #     # print(
    #     #     f"Average sample rate for angular converting: {average_sample_rate} Hz"
    #     # )
    #     length = orientation.shape[0]
    #     # convert the orientation to radian per second
    #     orientation = R.from_quat(orientation)
    #     # compute the angular_velocity  base on the orientation
    #     angular_velocities = [np.zeros(3)]
    #     for i in range(1, length):
    #         dt = (
    #             1 / sample_rate
    #             if ts is None
    #             else ((ts[i] - ts[i - 1]) / np.timedelta64(1, "s"))
    #         )
    #         # Compute the relative rotation from step i-1 to step i
    #         relative_rotation = orientation[i] * orientation[i - 1].inv()
    #         # Extract the angular velocity in the form of a rotation vector
    #         angular_velocity = relative_rotation.as_rotvec() / dt
    #         angular_velocities.append(angular_velocity)
    #     angular_velocities = np.array(angular_velocities)  # Convert list to numpy array
    #     angular_velocities = torch.tensor(angular_velocities).float()

    #     # check finite
    #     assert torch.isfinite(
    #         angular_velocities
    #     ).all(), "Non-finite values in angular_velocities"
    #     return angular_velocities
    def _orientation2degpersec(self, orientation, sample_rate, ts=None):
        """
        orientation: torch.Tensor
            the orientation data
            shape: (N, 4) in the form of (x, y, z, w)

        return: torch.Tensor
            the orientation in radian per second
            shape: (N, 3) in the form of (x, y, z)
        """

        def quat_to_rot_vec(quat, eps=1e-8):
            # Convert quaternion from (x, y, z, w) to (w, x, y, z)
            q_w, q_xyz = quat[..., -1:], quat[..., :3]

            angle = 2 * torch.acos(torch.clamp(q_w, -1.0, 1.0))
            norm = torch.linalg.norm(q_xyz, dim=-1, keepdim=True)
            axis = q_xyz / (norm + eps)
            return axis * angle

        def quat_multiply(q1, q2):
            x1, y1, z1, w1 = q1.unbind(-1)
            x2, y2, z2, w2 = q2.unbind(-1)
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.stack((x, y, z, w), dim=-1)

        def quat_conjugate(quat):
            return torch.cat([-quat[..., :3], quat[..., -1:]], dim=-1)

        if isinstance(orientation, np.ndarray):
            orientation = torch.from_numpy(orientation).float()

        # Ensure orientation is on the correct device
        device = orientation.device

        q_curr = orientation[1:]
        q_prev = orientation[:-1]

        if ts is not None:
            if isinstance(ts, np.ndarray):
                # Convert numpy timedelta to seconds in a tensor
                dt = torch.from_numpy(np.diff(ts) / np.timedelta64(1, "s")).to(
                    device, dtype=orientation.dtype
                )
            else:
                dt = torch.diff(ts)
            dt = dt.unsqueeze(-1)
        else:
            dt = 1 / sample_rate

        # Compute relative rotation: q_rel = q_curr * q_prev.inv()
        q_prev_inv = quat_conjugate(q_prev)
        relative_rotation = quat_multiply(q_curr, q_prev_inv)

        # Convert relative rotation to rotation vector (angular velocity)
        angular_velocity = quat_to_rot_vec(relative_rotation) / dt

        # Prepend a zero vector for the first time step
        angular_velocities = F.pad(angular_velocity, (0, 0, 1, 0))

        assert torch.isfinite(
            angular_velocities
        ).all(), "Non-finite values in angular_velocities"
        return angular_velocities

    def select_input_output(
        self,
        acc,
        gyr,
        mag,
        location,
        velocity,
        acceleration,
        orientation,
    ):
        """
        Select the input and output data based on the input and output dimensions
        ----------------------------------------
        input:
            acc: torch.Tensor
                the acceleration data
            gyr: torch.Tensor
                the gyroscope data
            mag: torch.Tensor
                the magnetometer data
            vel: torch.Tensor
                the velocity data
            acc: torch.Tensor
                the acceleration data
            orientation: torch.Tensor
                the orientation data
            location : torch.Tensor
                the location data
        ----------------------------------------
        return:
            input_data: torch.Tensor
                the input data
            output_data: torch.Tensor
                the output data
        """
        angular_vel = quaternion_to_angular_velocity(
            orientation, dt=1 / self.sampling_rate
        )
        orientation_euler = R.from_quat(orientation.cpu().numpy()).as_euler(
            "xyz", degrees=False
        )
        orientation_euler = torch.from_numpy(orientation_euler).float()
        observation = torch.cat([acc.cpu(), gyr.cpu()], dim=1).cpu()
        label = torch.cat([velocity.cpu(), angular_vel.cpu()], dim=1).cpu()
        # label = torch.cat([acceleration.cpu(), angular_vel.cpu()], dim=1).cpu()
        # l = velocity.cumsum(dim=0) / self.sampling_rate
        # l = l[:, :2].cpu().numpy()
        # plt.close("all")
        # plt.figure(figsize=(10, 4))
        # plt.plot(l[:, 0], l[:, 1], label="Trajectory")
        # plt.title("2D Trajectory from Velocity Integration")
        # plt.xlabel("X Position")
        # plt.ylabel("Y Position")
        # plt.axis("equal")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        return observation, label

    def endLoading(self):
        """
        End the loading process
        ----------------------------------------
        input:
            None
        ----------------------------------------
        return:
            None
        """
        if self.next_window:
            endLoadingMinusWindowSize(self)
            return

        self.chunk_index = (
            [i["size"] for i in self.Data]
            if self.useStep == False
            else [len(i["dataSteps"]) for i in self.Data]
        )
        self.chunk_index = torch.tensor(self.chunk_index) // self.stride
        self.chunk_index = self.chunk_index.cumsum(dim=0)
        self.chunk_index = torch.cat([torch.tensor([0]), self.chunk_index], dim=0)
        for key, valule in DATASET_DICT.items():
            if valule == self._label:
                self._name = key
                break

        # Prepare info dictionary
        info_to_print = {
            "Dataset": self._name,
            "Mode": self.mode,
            "Window size": f"{self.window_size}",
            "Stride": f"{self.stride}",
            "Sampling rate": f"{self.sampling_rate} Hz",
            "Effective Duration": f"{self.window_size/self.sampling_rate:.2f}s",
            "Use step": self.useStep,
            "Number of sequences": len(self.files),
            "Total number of samples": self.chunk_index[-1].item(),
        }

        # Print aligned information
        self.print_aligned_info(info_to_print, f"Dataset Summary")
        # self.save()

    @rank_zero_only
    def print_aligned_info(self, info_dict, title="Loading Complete", color="green"):
        """
        Print aligned information with dynamic input in a box using the 'rich' library.
        ----------------------------------------
        input:
            info_dict: dict
                Dictionary containing key-value pairs to print
            title: str
                Title to display at the top
            color: str
                Color to use for printing (default: green)
        ----------------------------------------
        return:
            None
        """
        console = Console()
        # style = color

        table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
        table.add_column(justify="right")
        table.add_column(justify="left")

        for key, value in info_dict.items():
            table.add_row(f"{key}:", str(value))

        panel = Panel(
            table,
            title=f"{title}",
            expand=False,
        )
        print()
        console.print(panel)
        print()
