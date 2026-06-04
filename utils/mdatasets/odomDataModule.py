from . import *
from .common_imports import *
from .utils import *


class odomDataModule(pl.LightningDataModule):
    """
    DataModule for the odometry dataset
    ---------------------------------------
    """

    def __init__(self, config, data_dir: str = "./datasets") -> None:
        """
        Initialize the odometry dataset
        ---------------------------------------
        input:
            config: dict
                the configuration of the dataset
            data_dir: str
                the directory of the dataset
        return:
            None
        """
        super().__init__()
        self.config = config
        self.data_dir = Path(data_dir)
        self.Sampler = {}

    def gen_dataset(self, mode, dataset: str = None):
        """
        Generate the dataset
        ---------------------------------------
        input:
            mode: str
                the mode of the dataset
            dataset: str
                the dataset to use
        return:
            dataset: torch.utils.data.Dataset
                the dataset
        """
        if mode == "train":
            if self.config["pre_augmentation"]["enabled"]:
                config = {
                    "probability": self.config["pre_augmentation"]["probability"],
                    "mode": self.config["pre_augmentation"]["mode"],
                    "label_transform": True,
                    "degree": self.config["pre_augmentation"]["degree"],
                }
                transform = torch.nn.Sequential(
                    rotationNoise(config),
                )
                transform.add_module(
                    "rotationNoise",
                    rotationNoise(config=self.config["augmentation"]["rotationNoise"]),
                )
                # transform = videoTransform(
                #     torch.nn.Sequential(
                #         rotationNoise(config),
                #     )
                # )

            else:
                transform = torch.nn.Sequential()
                # transform.add_module(
                #     "gaussianNoise",
                #     gaussianNoise(config=self.config["augmentation"]["gaussianNoise"]),
                # )
                # transform.add_module(
                #     "shiftNoise",
                #     shiftNoise(config=self.config["augmentation"]["shiftNoise"]),
                # )
                # transform.add_module(
                #     "scaleNoise",
                #     scaleNoise(config=self.config["augmentation"]["scaleNoise"]),
                # )
                # transform.add_module(
                #     "rotationNoise",
                #     rotationNoise(config=self.config["augmentation"]["rotationNoise"]),
                # )
                # transform.add_module(
                #     "axisMasking",
                #     axisMasking(config=self.config["augmentation"]["axisMasking"]),
                # )
                # transform.add_module("scaling", scaling(config=self.config["augmentation"]["scaling"]))
        else:
            transform = torch.nn.Sequential()
        if not (
            "ronin" in self.config["model"].lower()
            or "ionet" in self.config["model"].lower()
            or "tlio" in self.config["model"].lower()
            or "time" in self.config["model"].lower()
            or "tinymodule" in self.config["model"].lower()
            or "lliomodule" in self.config["model"].lower()
            or "hlsmodule" in self.config["model"].lower()
            or "hifimodule" in self.config["model"].lower()
            or "vaegeneratormodule" in self.config["model"].lower()
            or "cyclegan" in self.config["model"].lower()
            or "unet" in self.config["model"].lower()
            or "lstm" in self.config["model"].lower()
            or "imudiffusionmodule" in self.config["model"].lower()
            or "inertialbridge" in self.config["model"].lower()
            or "inertialflow" in self.config["model"].lower()
            or "inertialdiffusion" in self.config["model"].lower()
            or "inertialdirectregression" in self.config["model"].lower()
            or "PrimeVAEModule".lower() in self.config["model"].lower()
        ):
            transform.add_module(
                "Time2Frequency",
                Time2Frequency(n_fft=self.config["n_fft"]),
            )

        if dataset is not None:
            print(cl.Fore.red + f'- Using "{dataset}" forcedly' + cl.Style.reset)
        else:
            dataset = self.config["dataset"]

        rank_zero_info(cl.Fore.green + f'- Using "{dataset}" dataset' + cl.Style.reset)
        data_dir = str(self.data_dir)
        if dataset == "ADVIO":
            return ADVIODataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "OxIOD":
            OxIOD_VICON = OxIODDataset(
                root=data_dir.replace("hybrid", "OxIOD"),
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                # skip_filters=["tango", "nexus"],
                skip_filters=["tango"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
            OxIOD_Tango = OxIODDataset(
                root=data_dir.replace("hybrid", "OxIOD"),
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                keep_filters=["tango"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
            return torch.utils.data.ConcatDataset([OxIOD_VICON, OxIOD_Tango])
        elif dataset == "OxIOD_VICON":
            return OxIODDataset(
                root=data_dir.replace("OxIOD_VICON", "OxIOD"),
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                # keep_filters=["trolley", "slow walking", "running"],
                # keep_filters=["tango"] if mode == "test" else None,
                # keep_filters=["trolley"],
                # keep_filters=["nexus"],
                # keep_filters=["tango"],
                # skip_filters=["nexus", "tango"],
                skip_filters=["tango"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "OxIOD_tango":
            return OxIODDataset(
                root=data_dir.replace("OxIOD_tango", "OxIOD"),
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                sampling_rate=self.config["sampling_rate"],
                # keep_filters=["trolley", "slow walking", "running"],
                # keep_filters=["tango"] if mode == "test" else None,
                # keep_filters=["trolley"],
                # keep_filters=["nexus"],
                keep_filters=["tango"],
                # skip_filters=["nexus", "tango"],
                stride=self.config["stride"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "RIDI":
            return RIDIDataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "RoNIN":
            return RoNINDataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "RoNINs":
            # Combine RoNIN and RoNIN_unseen into a single concatenated dataset.
            # When testing, include both the regular test split and the unseen test split.
            ro_main = RoNINDataset(
                root=data_dir.replace("RoNINs", "RoNIN"),
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
            if mode == "test":
                ro_unseen = RoNINDataset(
                    root=data_dir.replace("RoNINs", "RoNIN"),
                    useStep=self.config["useStep"],
                    mode="test_unseen",
                    window_size=self.config["window_size"],
                    stride=self.config["stride"],
                    sampling_rate=self.config["sampling_rate"],
                    transform=transform,
                    GravityRemoval=self.config["GravityRemoval"],
                    encoding=self.config["encoding"],
                    next_window=self.config["next_window"],
                )
                return torch.utils.data.ConcatDataset([ro_main, ro_unseen])
            return ro_main

        elif dataset == "RoNIN_unseen":
            assert mode == "test", "RoNIN_unseen dataset can only be used in test mode"
            return RoNINDataset(
                root=data_dir.replace("RoNIN_unseen", "RoNIN"),
                useStep=self.config["useStep"],
                mode="test_unseen",
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "TLIO":
            return TLIODataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "EuRoC":
            return EuRoCDataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "MSD":
            return MSDDataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "M2DGR":
            return M2DGRDataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "UZH_FPV":
            return UZH_FPVDataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "GrandTour":
            return GrandTourDataset(
                root=data_dir,
                useStep=self.config["useStep"],
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                sampling_rate=self.config["sampling_rate"],
                transform=transform,
                GravityRemoval=self.config["GravityRemoval"],
                encoding=self.config["encoding"],
                next_window=self.config["next_window"],
            )
        elif dataset == "hybrid_Prime":
            base_kwargs = {
                "useStep": self.config["useStep"],
                "mode": mode,
                "window_size": self.config["window_size"],
                "stride": self.config["stride"],
                "sampling_rate": self.config["sampling_rate"],
                "transform": transform,
                "GravityRemoval": self.config["GravityRemoval"],
                "encoding": self.config["encoding"],
                "next_window": self.config["next_window"],
            }

            # RoNINs
            ro_main = RoNINDataset(root=self.data_dir.parent / "RoNIN", **base_kwargs)
            total = [ro_main]
            if mode == "test":
                ronin_unseen_kwargs = base_kwargs.copy()
                ronin_unseen_kwargs["mode"] = "test_unseen"
                total.append(
                    RoNINDataset(
                        root=self.data_dir.parent / "RoNIN", **ronin_unseen_kwargs
                    )
                )

            # TLIO
            total.append(TLIODataset(root=self.data_dir.parent / "TLIO", **base_kwargs))

            # UZH_FPV
            total.append(
                UZH_FPVDataset(root=self.data_dir.parent / "UZH_FPV", **base_kwargs)
            )

            total.sort(key=lambda x: x._label)
            min_length = min([len(d) for d in total])

            concat = torch.utils.data.ConcatDataset(total)
            if self.config["equal_sampling"]:
                self.Sampler[f"{mode}"] = EqualSampler(
                    total,
                    num_samples_per_dataset=min_length,
                    mode=mode,
                )
            return concat
        elif dataset in [
            "hybrid",
            "hybrid_drone",
            "hybrid_robot",
            "hybrid_drone_robot",
        ]:
            base_kwargs = {
                "useStep": self.config["useStep"],
                "mode": mode,
                "window_size": self.config["window_size"],
                "stride": self.config["stride"],
                "sampling_rate": self.config["sampling_rate"],
                "transform": transform,
                "GravityRemoval": self.config["GravityRemoval"],
                "encoding": self.config["encoding"],
                "next_window": self.config["next_window"],
            }

            # ADVIO = ADVIODataset(root=self.data_dir.parent / "ADVIO", **base_kwargs)
            RIDI = RIDIDataset(root=self.data_dir.parent / "RIDI", **base_kwargs)
            RoNIN = RoNINDataset(root=self.data_dir.parent / "RoNIN", **base_kwargs)

            oxiod_kwargs = base_kwargs.copy()
            oxiod_kwargs["skip_filters"] = ["tango"]
            OxIOD_VICON = OxIODDataset(
                root=self.data_dir.parent / "OxIOD", **oxiod_kwargs
            )

            oxiod_kwargs_tango = base_kwargs.copy()
            oxiod_kwargs_tango["keep_filters"] = ["tango"]
            OxIOD_Tango = OxIODDataset(
                root=self.data_dir.parent / "OxIOD", **oxiod_kwargs_tango
            )

            TLIO = TLIODataset(root=self.data_dir.parent / "TLIO", **base_kwargs)

            total = [
                # ADVIO,
                RIDI,
                RoNIN,
                OxIOD_VICON,
                OxIOD_Tango,
                TLIO,
            ]

            if "drone" in dataset:
                UZH_FPV = UZH_FPVDataset(
                    root=self.data_dir.parent / "UZH_FPV", **base_kwargs
                )
                total.append(UZH_FPV)
                # EuRoC = EuRoCDataset(root=self.data_dir.parent / "EuRoC", **base_kwargs)
                # total.append(EuRoC)

            if "robot" in dataset:
                GrandTour = GrandTourDataset(
                    root=self.data_dir.parent / "GrandTour", **base_kwargs
                )
                total.append(GrandTour)

            if mode == "test":
                ronin_unseen_kwargs = base_kwargs.copy()
                ronin_unseen_kwargs["mode"] = "test_unseen"
                RoNIN_unseen = RoNINDataset(
                    root=self.data_dir.parent / "RoNIN", **ronin_unseen_kwargs
                )
                total.append(RoNIN_unseen)

            # sort the total list base on the element of ._label in the datasets
            total.sort(key=lambda x: x._label)
            min_length = min([len(d) for d in total])

            concat = torch.utils.data.ConcatDataset(total)
            if self.config["equal_sampling"]:
                self.Sampler[f"{mode}"] = EqualSampler(
                    total,
                    num_samples_per_dataset=(
                        min_length if mode == "train" else min_length
                    ),
                    mode=mode,
                )
            # merge the dataset
            return concat
        else:
            raise ValueError(f"Invalid dataset {self.config['dataset']}")

    def setup(self, stage: str = "fit") -> None:
        """
        Setup the dataset
        ---------------------------------------
        input:
            stage: str
                the stage of the dataset
        return:
            None
        """
        if stage == "fit":
            if hasattr(self, "train_dataset") and hasattr(self, "val_dataset"):
                return
            self.train_dataset = self.gen_dataset("train")
            self.val_dataset = self.gen_dataset("val")
        elif stage == "test":
            if hasattr(self, "test_dataset"):
                return
            self.test_dataset = self.gen_dataset("test")
        elif stage == "pred":
            if hasattr(self, "pred_dataset"):
                return
            self.pred_dataset = self.gen_dataset("pred")
        else:
            raise ValueError(f"Invalid stage {stage}")

    def train_dataloader(self):
        """
        return the train dataloader
        ---------------------------------------
        input:
            None
        return:
            dataloader: torch.utils.data.DataLoader
                the train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False if "train" in self.Sampler else True,
            pin_memory=self.config["pin_memory"],
            drop_last=False,
            # num_workers=os.cpu_count() // 2,
            num_workers=self.config["num_workers"],
            # num_workers=0,
            # collate_fn=collect_fn,
            sampler=self.Sampler["train"] if "train" in self.Sampler else None,
            persistent_workers=self.config["num_workers"] > 0,
        )

    def val_dataloader(self):
        """
        return the validation dataloader
        ---------------------------------------
        input:
            None
        return:
            dataloader: torch.utils.data.DataLoader
                the validation dataloader
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"] * 4,
            # shuffle=self.config["shuffle"],
            shuffle=False,
            pin_memory=self.config["pin_memory"],
            drop_last=False,
            # num_workers=os.cpu_count() // 2,
            num_workers=self.config["num_workers"],
            # num_workers=0,
            # collate_fn=collect_fn,
            # sampler=self.Sampler["val"] if "val" in self.Sampler else None,
            persistent_workers=self.config["num_workers"] > 0,
        )

    def test_dataloader(self):
        """
        return the test dataloader
        ---------------------------------------
        input:
            None
        return:
            dataloader: torch.utils.data.DataLoader
                the test dataloader
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"] * 4,
            shuffle=False,
            pin_memory=self.config["pin_memory"],
            drop_last=False,
            # num_workers=os.cpu_count() // 2,
            num_workers=self.config["num_workers"],
            # num_workers=0,
            # collate_fn=collect_fn,
            # sampler=self.Sampler["test"] if "test" in self.Sampler else None,
            persistent_workers=self.config["num_workers"] > 0,
        )

    def pred_dataloader(self):
        """
        return the prediction dataloader
        ---------------------------------------
        input:
            None
        return:
            dataloader: torch.utils.data.DataLoader
                the prediction dataloader

        """
        return torch.utils.data.DataLoader(
            self.pred_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=self.config["pin_memory"],
            drop_last=False,
            # num_workers=os.cpu_count() // 2,
            num_workers=self.config["num_workers"],
            # num_workers=0,
            # collate_fn=collect_fn,
            # sampler=self.Sampler["pred"] if "pred" in self.Sampler else None,
            persistent_workers=self.config["num_workers"] > 0,
        )

