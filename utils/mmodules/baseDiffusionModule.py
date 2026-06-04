from .common_imports import *
from .utils import *


class baseDiffusionModule(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.lr = self.config["lr"]
        self.warm_up = self.config["warm_up"]
        # 3. PROCESSORS (Keep your existing ones)
        self.sensorProcessor = nn.Sequential()
        self.labelProcessor = nn.Sequential()
        self.finalLabelProcessor = nn.Sequential()
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            num_train_timesteps=self.config["num_time_steps"],
            beta_start=self.config["scheduler_s"],
            beta_end=self.config["scheduler_e"],
            beta_schedule=self.config["scheduler_mode"],
            # num_train_timesteps=config["num_time_steps"],
            # beta_start=config["scheduler_s"],  # 0.0001,
            # beta_end=config["scheduler_e"],  # = 0.02,
            # beta_schedule=config["scheduler_mode"],  # "linear",
            # trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
            # variance_type="fixed_small",
            # clip_sample=True,
            # prediction_type="epsilon",
            # thresholding=False,
            # dynamic_thresholding_ratio=0.995,
            # clip_sample_range=1.0,
            # sample_max_value=1.0,
            # timestep_spacing="leading",
            # steps_offset=0,
            # rescale_betas_zero_snr=True,
        )
        self.toYUV = nn.Sequential(
            # batchTimeToFrequency(
            #     n_fft=self.config["n_fft"],
            # ),
            # Rearrange("b c f t I -> b c (f I) t", c=6, f=16, t=32, I=2),
            # Rearrange("b c (h w) -> b c h w", h=24, w=24),
            # Rearrange("b c (h w) -> b c w h", h=16, w=16),
        )
        self.toXYZ = nn.Sequential(
            # Rearrange("b c h w -> b c (h w)", h=24, w=24),
            # Rearrange("b c (f I) t -> b c f t I", c=6, f=16, t=32, I=2),
            batchFrequencyToTime(
                output_length=self.config["window_size"],
                n_fft=self.config["n_fft"],
            ),
            # Rearrange("b c w h -> b c (w h)", w=16, h=16),
        )
        self.toObservation = nn.Sequential(
            # toObservation(),
            Rearrange("b c f t I -> b c (I f) t", f=16, t=32, I=2),
            # Rearrange("b c (h w) -> b c h w", h=32, w=32),
            # reshapeNet((6, 24, 24)),
        )
        self.deObservation = nn.Sequential(
            # reshapeNet((6, 8, 36, 2)),
            # Rearrange("b c h w -> b c (h w)", h=32, w=32),
            Rearrange("b c (I f) t -> b c f t I", f=16, t=32, I=2),
            # deObservation(),
        )

        # generate metrics
        self.metrics = {
            "train": self._generate_metrics("train"),
            "val": self._generate_metrics("val"),
            "test": self._generate_metrics("test"),
        }
        if self.config["peek_testing"]["enable"]:
            self.metrics["peek_testing"] = self._generate_metrics("peek_testing")

        # log hyperparameters
        self.save_hyperparameters(self.config)

    def preprocessing(self, batch):
        x, y, dataL = batch

        # x_, y_ = self.transform((x, y))
        x_, y_ = x, y
        x_, y_ = self.to_YUV(x_, y_, dataL)

        assert torch.isfinite(x_).all(), "x is not finite"
        assert torch.isfinite(y_).all(), "y is not finite"

        return (
            x_,
            y_,
            # torch.zeros(list(x.shape[:2]) + [0], dtype=torch.float32).to(x_.device),
            None,
        )

    def forward(self, batch):
        print(
            cl.Fore.red,
            "CRITICAL WARNING: baseDiffusionModule forward called",
            cl.Style.reset,
        )
        return self.sequential_forward(batch, mode="test")

    def postprocessing(self, batch):
        x, y, dataL = batch
        x_, y_ = x, y
        # x_, y_ = self.denormalizer(x_, y_)  # frequency denormalization
        # x_, y_ = self.denormalizer(x_, y_, dataL) # time denormalization
        x_, y_ = self.to_XYZ(x_, y_, dataL)  #

        assert torch.isfinite(x_).all(), "x is not finite"
        assert torch.isfinite(y_).all(), "y is not finite"

        return x_, y_

    def mask_generation(self, dataL, window_size, batch_size, channels):
        mask = (
            torch.arange(window_size, requires_grad=False)
            .expand(batch_size, window_size)
            .to(dataL.device)
        )
        mask = mask < dataL.unsqueeze(1)
        mask = mask.unsqueeze(1).expand(-1, channels, -1)
        mask = torch.logical_not(mask)
        return mask

    def video_mask_generation(self, dataL, window_size, batch_size, channels):
        masks = []
        for i in range(dataL.shape[1]):
            mask = self.mask_generation(dataL[:, i], window_size, batch_size, channels)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)
        # print(cl.Fore.red, f"masks shape: {masks.shape}", cl.Style.reset)
        return ~masks

    def sample_noise(self, batch, pad_size, dataL, do_noise=True):
        # result = self.transform(batch) if do_noise else batch
        # x, y = result
        x, y = batch
        x, y = x.clone().detach(), y.clone().detach()

        x, y, encoding = self.preprocessing((x, y, dataL))

        assert torch.isfinite(x).all(), "x is not finite"
        assert torch.isfinite(y).all(), "y is not finite"

        return x, y, encoding

    def _generate_metrics(self, suffix):
        metrics = {
            # f"metric_MSE/{suffix}": nn.MSELoss(),
            # f"metric_MAE/{suffix}": nn.L1Loss(),
            # #
            # f"metric_mse_acc_X/{suffix}": mMSE(channel_index=[0], norm=False),
            # f"metric_mse_acc_Y/{suffix}": mMSE(channel_index=[1], norm=False),
            # f"metric_mse_acc_Z/{suffix}": mMSE(channel_index=[2], norm=False),
            # f"metric_mse_acc/{suffix}": mMSE(channel_index=[0, 1, 2], norm=False),
            #
            f"metric_naive_distance_error_XY/{suffix}": NaiveDistanceError(
                channel_index=[0, 1],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_X/{suffix}": NaiveDistanceError(
                channel_index=[0],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_Y/{suffix}": NaiveDistanceError(
                channel_index=[1],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError(
                channel_index=[2],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error/{suffix}": NaiveDistanceError(
                channel_index=[0, 1, 2],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            # acc
            f"metric_acc_pearson_X/{suffix}": PearsonMetric(
                channel_index=[0], norm=False
            ),
            f"metric_acc_pearson_Y/{suffix}": PearsonMetric(
                channel_index=[1], norm=False
            ),
            f"metric_acc_pearson_Z/{suffix}": PearsonMetric(
                channel_index=[2], norm=False
            ),
            f"metric_acc_pearson_norm/{suffix}": PearsonMetric(
                channel_index=[0, 1, 2], norm=True
            ),
            #
            f"metric_acc_simVector/{suffix}": CosSimMetric(
                channel_index=[0, 1, 2], norm=False
            ),
            f"metric_acc_simVector_X/{suffix}": CosSimMetric(
                channel_index=[0], norm=False
            ),
            f"metric_acc_simVector_Y/{suffix}": CosSimMetric(
                channel_index=[1], norm=False
            ),
            f"metric_acc_simVector_Z/{suffix}": CosSimMetric(
                channel_index=[2], norm=False
            ),
            f"metric_acc_simVector_norm/{suffix}": CosSimMetric(
                channel_index=[0, 1, 2], norm=True
            ),
            # snr
            f"metric_acc_snr/{suffix}": mSNR(channel_index=[0, 1, 2], norm=False),
            f"metric_acc_snr_X/{suffix}": mSNR(channel_index=[0], norm=False),
            f"metric_acc_snr_Y/{suffix}": mSNR(channel_index=[1], norm=False),
            f"metric_acc_snr_Z/{suffix}": mSNR(channel_index=[2], norm=False),
            #
            # gyr
            # f"metric_mse_gyr_X/{suffix}": mMSE(channel_index=[3], norm=False),
            # f"metric_mse_gyr_Y/{suffix}": mMSE(channel_index=[4], norm=False),
            # f"metric_mse_gyr_Z/{suffix}": mMSE(channel_index=[5], norm=False),
            # f"metric_mse_gyr/{suffix}": mMSE(channel_index=[3, 4, 5], norm=False),
            # gyr
            f"metric_naive_Angular_error_X/{suffix}": NaiveAngularError(
                channel_index=[3],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_Angular_error_Y/{suffix}": NaiveAngularError(
                channel_index=[4],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_Angular_error_Z/{suffix}": NaiveAngularError(
                channel_index=[5],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_Angular_error/{suffix}": NaiveAngularError(
                channel_index=[3, 4, 5],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            #
            f"metric_gyr_pearson_X/{suffix}": PearsonMetric(
                channel_index=[3], norm=False
            ),
            f"metric_gyr_pearson_Y/{suffix}": PearsonMetric(
                channel_index=[4], norm=False
            ),
            f"metric_gyr_pearson_Z/{suffix}": PearsonMetric(
                channel_index=[5], norm=False
            ),
            f"metric_gyr_pearson_norm/{suffix}": PearsonMetric(
                channel_index=[3, 4, 5], norm=True
            ),
            #
            f"metric_gyr_simVector/{suffix}": CosSimMetric(
                channel_index=[3, 4, 5], norm=False
            ),
            f"metric_gyr_simVector_X/{suffix}": CosSimMetric(
                channel_index=[3], norm=False
            ),
            f"metric_gyr_simVector_Y/{suffix}": CosSimMetric(
                channel_index=[4], norm=False
            ),
            f"metric_gyr_simVector_Z/{suffix}": CosSimMetric(
                channel_index=[5], norm=False
            ),
            f"metric_gyr_simVector_norm/{suffix}": CosSimMetric(
                channel_index=[3, 4, 5], norm=True
            ),
            # snr
            f"metric_gyr_snr/{suffix}": mSNR(channel_index=[3, 4, 5], norm=False),
            f"metric_gyr_snr_X/{suffix}": mSNR(channel_index=[3], norm=False),
            f"metric_gyr_snr_Y/{suffix}": mSNR(channel_index=[4], norm=False),
            f"metric_gyr_snr_Z/{suffix}": mSNR(channel_index=[5], norm=False),
            #
            # # physic
            # f"metric_angular_error/{suffix}": angularError(),
            # f"metric_angular_vel_error/{suffix}": angularVelError(),
            # f"metric_acceleration_loss/{suffix}": accelerationError(),
            # f"metric_velocity_loss/{suffix}": velocityError(),
            # f"metric_position_loss/{suffix}": positionError(),
        }

        for key, metric in metrics.items():
            self.add_module(key, metric)
        return metrics

    def zero_one(self, x, y, dataL):
        """
        convert  from [-1,1] space to [0,1] space
        """
        mask = self.mask_generation(dataL, x.shape[-1], x.shape[0])
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = x.masked_fill(mask.unsqueeze(1).expand(-1, x.shape[1], -1), 0)
        y = y.masked_fill(mask.unsqueeze(1).expand(-1, y.shape[1], -1), 0)
        return x, y

    def minus_one(self, x, y, dataL):
        """
        convert  from [0,1] space to [-1,1] space
        """
        mask = self.mask_generation(dataL, x.shape[-1], x.shape[0])
        x = x * 2 - 1
        y = y * 2 - 1
        x = x.masked_fill(mask.unsqueeze(1).expand(-1, x.shape[1], -1), 0)
        y = y.masked_fill(mask.unsqueeze(1).expand(-1, y.shape[1], -1), 0)
        return x, y

    def to_YUV(self, x, y, dataL):
        # x = torch.stack([self.toYUV(_) for _ in x], dim=0)
        # y = torch.stack([self.toYUV(_) for _ in y], dim=0)
        x = self.toYUV(x)
        y = self.toYUV(y)
        x = self.sensorProcessor(x)
        y = self.labelProcessor(y)
        return x, y

    def to_XYZ(self, x, y, dataL):
        # x = torch.stack([self.toXYZ(_) for _ in x], dim=0)
        # y = torch.stack([self.toXYZ(_) for _ in y], dim=0)
        x = self.finalLabelProcessor(x)
        y = self.finalLabelProcessor(y)
        x = self.toXYZ(x)
        y = self.toXYZ(y)
        # x= self.sensorProcessor(x)
        return x, y

    def on_test_end(self):
        # extract this test
        if os.getenv("LOCAL_RANK", "0") == "0" and os.getenv("NODE_RANK", "0") == "0":
            log_dir = self.logger.log_dir
            metrics = extract_metrics(log_dir)

            # extract baseline
            log_dir = log_dir.split("/")
            log_dir = "/".join(log_dir[:-1])
            log_dir = log_dir + "/baseline"
            # check if the baseline exists
            if not os.path.exists(log_dir):
                print(cl.Fore.red + "- Baseline does not exist" + cl.Style.reset)
                return super().on_test_end()
            baseline_metrics = extract_metrics(log_dir)

            # save the hyperparameters with the metrics
            config = self.config.copy()

            # compare the metrics
            for key, value in metrics.items():
                # config[key] = value[-1][1]
                if key not in baseline_metrics or (
                    ("pearson" not in key)
                    and ("simVector" not in key)
                    and ("naive" not in key)
                ):
                    continue
                if "std" in key:
                    continue
                m = value[-1][1]
                bm = baseline_metrics[key][-1][1]

                positive_relation = True
                coefficient = 1
                if "naive" in key:
                    positive_relation = False
                    coefficient = -1

                if (m > bm) == positive_relation:
                    word = "^v^"
                    COLOR = cl.Fore.red
                else:
                    word = "@A@"
                    COLOR = cl.Fore.green
                if abs(bm) < 1e-7:
                    bm = 1e-7
                print(
                    COLOR
                    + f"{self.config['dataset']:5s} {word:5s} {key:40s} {m:>7.4f} with difference of {coefficient*(m-bm)/abs(bm)*100:>7.2f}% with original {bm:>7.4f}"
                    + cl.Style.reset
                )
            # self.save_hyperparameters(
            #     config
            # )  # sve the hyperparameters or metics indexing
        return super().on_test_end()

    def add_noise(self, original, noise, t):
        result = self.scheduler.add_noise(
            original_samples=original,
            noise=noise,
            timesteps=t,
        )
        # # replace the place where is t ==self.config["num_time_steps"] -1 with noise
        # mask = t == self.config["num_time_steps"] - 1
        # if mask.any():
        #     result[mask] = noise[mask].clone()
        # # replace the place where is t ==0 with original
        # mask = t == 0
        # if mask.any():
        #     result[mask] = original[mask].clone()

        return result

    def step_backward(
        self,
        original_input,
        estimate_noise,
        t,
        target_steps,
        get_x0=False,
        different_timestamp=True,
    ):
        batch_size = original_input.shape[0]
        channels = original_input.shape[1]
        result = original_input.clone().detach()
        # self.scheduler.set_timesteps(target_steps)
        # self.scheduler.sigmas = torch.zeros_like(self.scheduler.sigmas)

        if not different_timestamp:
            assert (t == t[0]).all(), "Not all timestamp are the same."
            temp = self.scheduler.step(
                model_output=estimate_noise,
                timestep=t[0].cpu(),
                sample=result,
                # variance_noise=estimate_noise[:, channels:],
            )
            # if t[0] == 0 and hasattr(result, "pred_original_sample"):
            #     return temp.pred_original_sample
            # else:
            #     return temp.prev_sample
            return temp.prev_sample

        for i in range(batch_size):
            self.scheduler.set_timesteps(target_steps)
            # self.scheduler.sigmas = torch.zeros_like(self.scheduler.sigmas)
            temp = self.scheduler.step(
                model_output=estimate_noise[i].unsqueeze(0),
                timestep=t[i].cpu(),
                sample=result[i].unsqueeze(0),
                # variance_noise=estimate_noise[:, channels:].unsqueeze(0),
            )
            # if t[i] == 0 and hasattr(result, "pred_original_sample"):
            #     result[i] = temp.pred_original_sample.squeeze(0)
            # else:
            #     result[i] = temp.prev_sample.squeeze(0)
            result[i] = temp.prev_sample.squeeze(0)

        return result
