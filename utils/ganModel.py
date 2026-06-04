import os

import colored as cl
import diffusers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchmetrics import Accuracy

from .mloss import distanceLabelLoss
from .mmodules import baseDiffusionModule, batchStepBatch, extract_metrics
from .scheduler import CosineWarmupScheduler
from .transform import IMUToYUV, YUVToIMU


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, dim=1):
        super(ResidualBlock, self).__init__()
        if dim == 1:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
        elif dim == 2:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif dim == 3:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_channels)
        else:
            raise ValueError("dim must be 1, 2 or 3")
        self.activation = activation(inplace=False)
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.activation(out1)
        out1 = self.dropout(out1) + x
        out2 = self.conv2(out1)
        out2 = self.dropout(out2)
        out2 = self.bn2(out2) + out1
        out2 = self.activation(out2)
        return out2


class downBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation=nn.ReLU, dim=1, num_steps=None
    ):
        super(downBlock, self).__init__()
        self.residual_block = ResidualBlock(in_channels, in_channels, activation, dim)

        if num_steps is not None:
            self.embedding = nn.Embedding(num_steps, in_channels)
            _in_channels = in_channels * 2
        else:
            _in_channels = in_channels

        if dim == 1:
            self.downsample = nn.Conv1d(
                _in_channels, out_channels, kernel_size=4, stride=2
            )
            self.bn = nn.BatchNorm1d(out_channels)
        elif dim == 2:
            self.downsample = nn.Conv2d(
                _in_channels, out_channels, kernel_size=4, stride=2
            )
            self.bn = nn.BatchNorm2d(out_channels)
        elif dim == 3:
            self.downsample = nn.Conv3d(
                _in_channels, out_channels, kernel_size=4, stride=2
            )
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            raise ValueError("dim must be 1, 2 or 3")
        self.activation = activation(inplace=False)
        self.dim = dim

    def forward(self, x, timesteps=None):
        out = self.residual_block(x)
        if hasattr(self, "embedding"):
            emb = self.embedding(timesteps)
            # repeat the embedding to match the w h l given the dimension
            if self.dim == 1:
                emb = emb.unsqueeze(-1).expand(-1, -1, out.shape[-1])
            elif self.dim == 2:
                emb = (
                    emb.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, out.shape[-2], out.shape[-1])
                )
            elif self.dim == 3:
                emb = (
                    emb.unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, out.shape[-3], out.shape[-2], out.shape[-1])
                )
            else:
                raise ValueError("dim must be 1, 2 or 3")
            out = torch.cat((out, emb), dim=1)
        elif timesteps is not None:
            raise ValueError("timesteps must be None if embedding is not used")
            # pass

        out = self.downsample(out)
        out = self.bn(out)
        out = self.activation(out)
        return out


class upBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super(upBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        final_dim,
        num_class,
        activation=nn.ReLU,
    ):
        super(Discriminator, self).__init__()
        self.down_blocks_time = nn.ModuleList()
        self.down_blocks_freq = nn.ModuleList()

        input_dim_time = input_dim
        hidden_dim_time = hidden_dim
        current_length_time = 512
        while current_length_time > 4:
            self.down_blocks_time.append(
                downBlock(
                    input_dim_time,
                    hidden_dim_time,
                    activation,
                    dim=1,
                    num_steps=num_class,
                )
            )
            input_dim_time = hidden_dim_time
            hidden_dim_time *= 2
            current_length_time //= 2

        input_dim_freq = input_dim
        hidden_dim_freq = hidden_dim
        current_length_freq = 51
        while current_length_freq > 4:
            self.down_blocks_freq.append(
                downBlock(
                    input_dim_freq,
                    hidden_dim_freq,
                    activation,
                    dim=2,
                    num_steps=num_class,
                )
            )
            input_dim_freq = hidden_dim_freq
            hidden_dim_freq *= 2
            current_length_freq //= 2

        self.adaptive_pool_time = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(input_dim_time, output_dim),
            activation(inplace=False),
        )
        self.adaptive_pool_freq = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(input_dim_freq, output_dim),
            activation(inplace=False),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(
                output_dim * 2, final_dim
            ),  # Concatenate time and frequency features
            # nn.ReLU(inplace=False),
            nn.Sigmoid(),
        )
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=100, hop_length=9, win_length=100
        )
        self.num_class = num_class

    def forward(self, x, timesteps=None):
        # Time series tower
        x_time = x
        for block in self.down_blocks_time:
            x_time = block(x_time, timesteps=timesteps)
        x_time = self.adaptive_pool_time(x_time)

        # Frequency series tower
        x_freq = self.spectrogram(x)
        for block in self.down_blocks_freq:
            x_freq = block(x_freq, timesteps=timesteps)
        x_freq = self.adaptive_pool_freq(x_freq)

        # Concatenate time and frequency features
        x_combined = torch.cat((x_time, x_freq), dim=-1)
        x_combined = self.fc(x_combined)
        return x_combined


class GANModule(baseDiffusionModule):
    def __init__(self, config):
        super(GANModule, self).__init__(config=config)
        self.generator = diffusers.models.UNet1DModel(
            sample_size=self.config["window_size"],
            in_channels=22,
            out_channels=6,
            act_fn="mish",
            freq_shift=1.0,
        )
        # self.discriminator = Discriminator(
        #     input_dim=6,
        #     hidden_dim=128,
        #     output_dim=self.config["num_time_steps"],
        #     final_dim=1,
        #     # final_dim=self.config["num_time_steps"],
        #     num_class=self.config["num_time_steps"],
        #     activation=nn.Mish,
        # )

        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = distanceLabelLoss()
        # self.criterion = nn.BCELoss()
        # self.criterion = nn.SmoothL1Loss()
        self.automatic_optimization = False

        self.accuracy = Accuracy(
            task="binary",
            # num_classes=self.config["num_time_steps"],
        )

    def noise_generation(self, target, dataL):
        injection_noise = torch.randn_like(target, device=target.device)
        mask = self.mask_generation(dataL, target.shape[-1], target.shape[0])
        injection_noise = injection_noise.masked_fill(
            mask.unsqueeze(1).expand(-1, injection_noise.shape[1], -1), 0
        )
        return injection_noise

    def gan_forward(self, batch, mode):
        # get the batch
        x, y, dataL = batch
        batch_size, channels, seq_len = x.shape
        valid = torch.ones(batch_size, 1, device=y.device)
        invalid = torch.zeros(batch_size, 1, device=y.device)

        # optimizer and scheduler
        # optimizer_g, optimizer_d = self.optimizers()
        # scheduler_g, scheduler_d = self.lr_schedulers()
        optimizer_g = self.optimizers()
        scheduler_g = self.lr_schedulers()

        # data preparation
        x, y = self.sample_noise(
            (x, y), 0, dataL, do_noise=True if mode == "train" else False
        )

        timesteps = torch.randint(
            0, self.config["num_time_steps"], (batch_size,), device=y.device
        )
        timesteps_prime = (
            (timesteps - 1).clamp(min=0, max=self.config["num_time_steps"] - 1).detach()
        )
        injection_noise = self.noise_generation(y, dataL)
        # injection_noise = x.clone().detach()

        noisy_input = self.scheduler.add_noise(
            original_samples=y,
            noise=injection_noise,
            timesteps=timesteps,
        ).clamp(-1, 1)

        """
        Generator
        """
        #
        if mode == "train":
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
        estimate_noise = self.generator(
            sample=noisy_input,
            timestep=timesteps,
            return_dict=False,
        )[0]
        restored_real = batchStepBatch(
            scheduler=self.scheduler,
            original=noisy_input,
            noise=injection_noise,
            t=timesteps,
        ).clamp(-1, 1)
        restored_fake = batchStepBatch(
            scheduler=self.scheduler,
            original=noisy_input,
            noise=estimate_noise,
            t=timesteps,
        ).clamp(-1, 1)

        # g_loss = self.criterion(
        #     self.discriminator(restored_fake, timesteps_prime),
        #     valid,
        # )
        g_loss = F.mse_loss(estimate_noise, injection_noise)
        if mode == "train":
            self.manual_backward(g_loss)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            optimizer_g.step()
            scheduler_g.step(g_loss)
            self.untoggle_optimizer(optimizer_g)
        ##########################################################################################
        # """
        # Discriminator
        # """
        # if mode == "train":
        #     self.toggle_optimizer(optimizer_d)
        #     optimizer_d.zero_grad()
        # real_y = self.discriminator(restored_real, timesteps_prime)
        # fake_y = self.discriminator(restored_fake.detach(), timesteps_prime)
        # # original_y = self.discriminator(y, torch.zeros_like(timesteps_prime))
        # # original_x = self.discriminator(
        # #     x, torch.ones_like(timesteps_prime) * (self.config["num_time_steps"] - 1)
        # # )
        # d_loss = (
        #     self.criterion(real_y, valid)
        #     + self.criterion(fake_y, invalid)
        #     # + self.criterion(original_y, valid)
        #     # + self.criterion(original_x, valid)
        # )
        # d_loss = d_loss / 2

        # if mode == "train":
        #     # Check for parameters with None gradients
        #     self.manual_backward(d_loss)
        #     torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        #     optimizer_d.step()
        #     scheduler_d.step(d_loss)
        #     self.untoggle_optimizer(optimizer_d)

        # ##########################################################################################
        # self.log_dict(
        #     {
        #         f"acc_real_y/{mode}": self.accuracy(real_y, valid),
        #         f"acc_fake_y/{mode}": self.accuracy(fake_y, invalid),
        #     },
        #     prog_bar=True,
        #     on_step=True,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        self.log(
            f"g_loss/{mode}",
            g_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        # self.log(
        #     f"d_loss/{mode}",
        #     d_loss,
        #     prog_bar=True,
        #     on_step=True,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        self.log(
            f"loss/{mode}",
            g_loss,  # + d_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"g_mse_loss/{mode}",
            F.mse_loss(estimate_noise, injection_noise),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        restored_fake, y = self.postprocessing((restored_fake, y, dataL))
        metrics = {}
        for key, metric in self.metrics[mode].items():
            metrics[key] = metric(restored_fake, y)
        self.log_dict(
            metrics, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True
        )
        return g_loss  # + d_loss

    def training_step(self, batch, batch_idx):
        return self.gan_forward(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.gan_forward(batch, "val")

    def test_step(self, batch, batch_idx):
        # self.only_one_exam(batch, batch_idx)
        # self.every_exam(batch, batch_idx)
        self.old_testing_approach(batch, batch_idx)

    def old_testing_approach(self, batch, batch_idx):
        x, y, dataL = batch
        batch_size, channels, seq_len = x.shape
        x, y = self.preprocessing((x, y, dataL))
        steps = torch.arange(
            start=self.config["num_time_steps"] - 1, end=0, step=-1
        ).to(y.device)

        restored_samples = x.clone().detach()
        for t in steps:
            noise = self.generator(
                sample=restored_samples,
                timestep=torch.ones(batch_size, dtype=torch.int64).to(y.device) * t,
                return_dict=False,
            )[0]
            restored_samples = batchStepBatch(
                scheduler=self.scheduler,
                original=restored_samples,
                noise=noise,
                t=torch.ones(batch_size, dtype=torch.int64) * t.cpu(),
            ).clamp(-1, 1)

        restored_samples, y = self.postprocessing((restored_samples, y, dataL))
        metrics = {}
        for key, metric in self.metrics["test"].items():
            metrics[key] = metric(restored_samples, y)
        self.log_dict(
            metrics, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True
        )

        return restored_samples

    def only_one_exam(self, batch, batch_idx):
        x, y, dataL = batch
        batch_size, channels, seq_len = x.shape
        x, y = self.preprocessing((x, y, dataL))

        # use for loop to estimate the steps of each sample with discriminator
        # then use the output to resteor the original samples
        restored_samples = []
        for i in range(batch_size):
            sample = x[i].unsqueeze(0)

            estimated_steps = self.discriminator(sample).long().view(-1)
            # print(f"Estimated steps: {estimated_steps}")
            # print(f"_" * 100)
            # estimated_steps = torch.argmax(estimated_steps, dim=1)
            steps = torch.arange(estimated_steps[0] + 1, 0, step=-1).to(
                estimated_steps.device
            )
            t = estimated_steps[0]
            restored_sample = sample.clone().detach()
            for t in steps:
                noise = self.generator(
                    sample=restored_sample,
                    timestep=torch.tensor([t]).to(estimated_steps.device),
                    return_dict=False,
                )[0]
                restored_sample = self.scheduler.step(
                    model_output=noise[0],
                    timestep=t,
                    sample=restored_sample[0],
                ).prev_sample.unsqueeze(0)
                restored_sample = torch.clamp(restored_sample, -1, 1)

            restored_samples.append(restored_sample)
        restored_samples = torch.cat(restored_samples, dim=0)

        restored_samples, y = self.postprocessing((restored_samples, y, dataL))
        restored_samples, y = self.postprocessing((restored_samples, y, dataL))
        metrics = {}
        for key, metric in self.metrics["test"].items():
            metrics[key] = metric(restored_samples, y)

        self.log_dict(
            metrics, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True
        )
        return restored_samples

    def every_exam(self, batch, batch_idx):
        x, y, dataL = batch
        batch_size, channels, seq_len = x.shape
        x, y = self.preprocessing((x, y, dataL))

        # use for loop to estimate the steps of each sample with discriminator
        # then use the output to resteor the original samples
        restored_samples = []
        for i in range(batch_size):
            # print("_" * 100)
            sample = x[i].unsqueeze(0)

            estimated_steps = self.discriminator(sample).long()
            # estimated_steps = torch.argmax(estimated_steps, dim=1)
            # steps = torch.arange(estimated_steps[0] + 1, 0, step=-1).to(
            #     estimated_steps.device
            # )
            t = estimated_steps[0]
            count = 0
            restored_sample = sample.clone().detach()
            while t != 0 and count < self.config["num_time_steps"]:
                # print(t)
                count += 1
                noise = self.generator(
                    sample=restored_sample,
                    timestep=torch.tensor([t]).to(estimated_steps.device),
                    return_dict=False,
                )[0]
                restored_sample = (
                    self.scheduler.step(
                        model_output=noise[0],
                        timestep=t,
                        sample=restored_sample[0],
                    )
                    .prev_sample.unsqueeze(0)
                    .cuda()
                )
                restored_sample = torch.clamp(restored_sample, -1, 1)

                estimated_steps = self.discriminator(restored_sample)
                # estimated_steps = torch.argmax(estimated_steps, dim=1)
                t = estimated_steps[0].long()

            restored_samples.append(restored_sample)
        restored_samples = torch.cat(restored_samples, dim=0)

        restored_samples, y = self.postprocessing((restored_samples, y, dataL))
        metrics = {}
        for key, metric in self.metrics["test"].items():
            metrics[key] = metric(restored_samples, y)

        self.log_dict(
            metrics, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True
        )
        return restored_samples

    def configure_optimizers(self):
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        # d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        # d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.1)

        g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer,
            mode="min",
            factor=0.9,
            patience=10,
            cooldown=5,
            min_lr=1e-8,
        )
        # g_scheduler = CosineWarmupScheduler(
        #     optimizer=g_optimizer,
        #     warmup=50,
        #     max_iters=100,
        # )

        # d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     d_optimizer,
        #     mode="min",
        #     factor=0.9,
        #     patience=10,
        #     cooldown=5,
        #     min_lr=1e-8,
        # )
        # d_scheduler = CosineWarmupScheduler(
        #     optimizer=d_optimizer,
        #     warmup=50,
        #     max_iters=100,
        # )

        # return [
        #     {
        #         "optimizer": g_optimizer,
        #         "lr_scheduler": {
        #             "scheduler": g_scheduler,
        #             "monitor": "g_loss/train_step",
        #             "interval": "step",
        #             "frequency": 1,
        #         },
        #     },
        #     {
        #         "optimizer": d_optimizer,
        #         "lr_scheduler": {
        #             "scheduler": d_scheduler,
        #             "monitor": "d_loss/train_step",
        #             "interval": "step",
        #             "frequency": 1,
        #         },
        #     },
        # ]

        return (
            {
                "optimizer": g_optimizer,
                "lr_scheduler": {
                    "scheduler": g_scheduler,
                    "monitor": "g_loss/train_step",
                    "interval": "step",
                    "frequency": 1,
                },
            },
        )

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
                print(
                    COLOR
                    + f"{self.config['dataset']:5s} {word:5s} {key:40s} {m:>7.4f} with difference of {coefficient*(m-bm)/abs(bm)*100:>7.2f}%"
                    + cl.Style.reset,
                )
        return super().on_test_end()
