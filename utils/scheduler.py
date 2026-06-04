from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.schedulers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)
from diffusers.utils import deprecate, is_scipy_available
from diffusers.utils.torch_utils import randn_tensor
from torch import nn, optim


class mDDPMScheduler(DDPMScheduler):
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://huggingface.co/papers/2006.11239
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://huggingface.co/papers/2006.11239
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://huggingface.co/papers/2006.11239
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        # if t > 0:
        #     device = model_output.device
        #     variance_noise = randn_tensor(
        #         model_output.shape,
        #         generator=generator,
        #         device=device,
        #         dtype=model_output.dtype,
        #     )
        #     if self.variance_type == "fixed_small_log":
        #         variance = (
        #             self._get_variance(t, predicted_variance=predicted_variance)
        #             * variance_noise
        #         )
        #     elif self.variance_type == "learned_range":
        #         variance = self._get_variance(t, predicted_variance=predicted_variance)
        #         variance = torch.exp(0.5 * variance) * variance_noise
        #     else:
        #         variance = (
        #             self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
        #         ) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )


class mDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        split_channels = model_output.shape[1] // 2

        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :split_channels]
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
                    "`v_prediction`, or `flow_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :split_channels]
                else:
                    epsilon = model_output
            elif self.config.prediction_type == "sample":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(
                original_samples.device, dtype=torch.float32
            )
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timesteps
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if (
            self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]
            and variance_noise is None
        ):
            noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=torch.float32,
            )
        elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None

        if (
            self.config.solver_order == 1
            or self.lower_order_nums < 1
            or lower_order_final
        ):
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample, noise=noise
            )
        elif (
            self.config.solver_order == 2
            or self.lower_order_nums < 2
            or lower_order_second
        ):
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample, noise=noise
            )
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, sample=sample, noise=noise
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)


class DDPM_Scheduler(nn.Module):
    def __init__(
        self, num_time_steps: int = 1000, modes: str = "linear", s=None, e=None
    ):
        super().__init__()
        if modes == "linear":
            if not s:
                s = 1e-4
            if not e:
                e = 0.02
            self.betas = torch.linspace(s, e, num_time_steps, requires_grad=False)
        elif modes == "exp":
            if not s:
                s = -12
            if not e:
                e = -2
            self.betas = torch.exp(
                torch.linspace(s, e, num_time_steps, requires_grad=False)
            )
        elif modes == "cosine":
            if not s:
                s = 1e-4
            if not e:
                e = 0.008
            steps = torch.arange(num_time_steps, dtype=torch.float32)
            self.betas = (
                0.5 * (1 + torch.sin(np.pi * steps / num_time_steps)) * (e - s) + s
            )
        elif modes == "quad":
            if not s:
                s = 1e-4
            if not e:
                e = 0.02
            self.betas = (
                torch.linspace(s**0.5, e**0.5, num_time_steps, requires_grad=False) ** 2
            )
        else:
            raise ValueError("Invalid scheduler mode")
        # self.betas[0] = 1e-7
        # self.betas[-1] = 1 - 1e-6
        # make betas[1:-1] trainable but still keep the first and last beta untrainable
        # self.betas = nn.Parameter(self.betas, requires_grad=True)

        # if modes != "cosine":
        # self.weight = nn.Parameter(
        #     torch.ones_like(self.betas[1:-1]), requires_grad=True
        # )
        self.weight = torch.ones_like(self.betas[1:-1])
        self.alphas = (1 - self.betas) * torch.cat(
            [torch.ones(1), self.weight, torch.ones(1)]
        )
        self.alpha_products = torch.cumprod(self.alphas, dim=0)
        self.sigma = torch.sqrt(self.betas)
        # self.sigma[0] = 0.0
        # self.sigma = torch.sqrt(
        #     torch.sqrt(1 - alpha_t_minus_1) * beta / torch.sqrt(1 - alpha)
        # )

    def forward(self, t, batch_size, device):
        self.betas = self.betas.to(device)
        self.weight = nn.Parameter(
            torch.ones_like(self.betas[1:-1]), requires_grad=True
        ).to(device)
        self.alphas = (1 - self.betas) * torch.cat(
            [torch.ones(1, device=device), self.weight, torch.ones(1, device=device)]
        ).to(device)
        self.alpha_products = torch.cumprod(self.alphas, dim=0).to(device)

        self.sigma = torch.sqrt(self.betas).to(device)
        # self.sigma = torch.sqrt(
        #     (1 - self.alpha_products[:-1]) / self.alphas[1:] * self.betas[1:]
        # ).to(device)

        return (
            self.betas[t].view(batch_size, 1, 1),
            self.alpha_products[t].view(batch_size, 1, 1),
            self.alphas[t].view(batch_size, 1, 1),
            self.sigma[t].view(batch_size, 1, 1),
        )

    def naive_sample_forward(self, pseudo_noise, ground_truth, t):
        index = t < 0
        t = torch.clamp(t, min=0)
        # assert pseudo_noise.shape == ground_truth.shape
        beta, alpha_product, alpha, sigma = self(
            t, pseudo_noise.shape[0], pseudo_noise.device
        )
        noisy = pseudo_noise.clone().detach()
        # noisy[:, :3] = (alpha_product * ground_truth) + (
        #     (1 - alpha_product) * pseudo_noise[:, :3]
        # )
        # noisy[:, :3] = (beta * pseudo_noise[:, :3]) + ((1 - beta) * ground_truth)
        noisy = (beta * pseudo_noise) + ((1 - beta) * ground_truth)
        if index.any():
            noisy[index] = ground_truth[index]

        return noisy

    def sample_forward(self, pseudo_noise, ground_truth, t):
        index = t < 0
        t = torch.clamp(t, min=0)
        # assert pseudo_noise.shape == ground_truth.shape
        beta, alpha_product, alpha, sigma = self(
            t, pseudo_noise.shape[0], pseudo_noise.device
        )
        noisy = pseudo_noise.clone().detach()
        # noisy[:, :3] = (torch.sqrt(alpha_product) * ground_truth) + (
        #     torch.sqrt(1 - alpha_product) * pseudo_noise[:, :3]
        # )
        noisy = (torch.sqrt(alpha_product) * ground_truth) + (
            torch.sqrt(1 - alpha_product) * pseudo_noise
        )
        if index.any():
            noisy[index] = ground_truth[index]

        return noisy

    # def precise_sample_backward(self, pseudo_observation, noise_estimated, t):
    #     beta, alpha_product, alpha, sigma = self(
    #         t, pseudo_observation.shape[0], pseudo_observation.device
    #     )
    #     denoisy = pseudo_observation.clone().detach()
    #     denoisy[:, :3] = (
    #         pseudo_observation[:, :3] - torch.sqrt(1 - alpha_product) * noise_estimated
    #     ) / torch.sqrt(alpha_product)
    #     return denoisy

    def sample_backward(self, pseudo_observation, noise_estimated, t, z=None):
        # z = None
        # assert pseudo_observation.shape == noise_estimated.shape
        beta, alpha_product, alpha, sigma = self(
            t, pseudo_observation.shape[0], pseudo_observation.device
        )
        index = t < 1
        z[index] = torch.zeros_like(z[index])

        if z is None:
            z = torch.zeros_like(pseudo_observation)
        denoisy = pseudo_observation.clone().detach()

        # denoisy[:, :3] = (
        #     1
        #     / torch.sqrt(alpha)
        #     * (
        #         pseudo_observation[:, :3]
        #         - (1 - alpha) / torch.sqrt(1 - alpha_product) * noise_estimated
        #     )
        #     + sigma * z[:, :3]
        # )
        denoisy = (
            1
            / torch.sqrt(alpha)
            * (
                pseudo_observation
                - (1 - alpha) / torch.sqrt(1 - alpha_product) * noise_estimated
            )
            + sigma * z
        )

        return denoisy


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch, **kwargs):
        # make sure the epoch is int
        epoch = float(epoch)
        # print(f'epoch: {epoch}')
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class WarmupReduceLROnPlateau(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, reduce_factor, patience, min_lr=0):
        self.warmup_epochs = warmup_epochs
        self.reduce_factor = reduce_factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.warmup_scheduler = None
        self.reduce_scheduler = None
        self.current_epoch = 0
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, verbose)

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            return [
                base_lr * (self.current_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            if self.reduce_scheduler is None:
                self.reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor=self.reduce_factor,
                    patience=self.patience,
                    min_lr=self.min_lr,
                )
            return self.reduce_scheduler.optimizer.param_groups[0]["lr"]

    def step(self, metrics=None):
        self.current_epoch += 1
        if self.current_epoch > self.warmup_epochs:
            self.reduce_scheduler.step(metrics)
        else:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr


def gen_linear_beta_t(num_timesteps: int):
    alpha_cumprod = np.linspace(1.0, 0.0, num_timesteps)  # Linear decay from 1 to 0

    # Step 2: Derive betas from alpha_cumprod
    alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]])
    alpha_t = alpha_cumprod / alpha_cumprod_prev
    beta_t = 1 - alpha_t
    beta_t = np.clip(beta_t, 1e-7, 1 - 1e-7)

    return beta_t


def gen_log_beta_t(num_timesteps: int, base: int = 2):
    alpha_cumprod = np.linspace(1, base, num_timesteps)[
        ::-1
    ]  # Linear decay from 1 to 0
    alpha_cumprod = np.log(alpha_cumprod) / np.log(base)
    alpha_cumprod[0] = 1
    alpha_cumprod[-1] = 0

    # Step 2: Derive betas from alpha_cumprod
    alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]])
    alpha_t = alpha_cumprod / alpha_cumprod_prev
    beta_t = 1 - alpha_t
    beta_t = np.clip(beta_t, 1e-7, 1 - 1e-7)  # Avoid numerical instability

    return beta_t


import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)


class IMUFlowScheduler(FlowMatchEulerDiscreteScheduler):
    """
    Custom Scheduler for IMU Diffusion that supports:
    1. prediction_type="sample" (Model predicts CLEAN DATA)
    2. Built-in add_noise function for training
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        prediction_type: str = "sample",
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            time_shift_type=time_shift_type,
            stochastic_sampling=stochastic_sampling,
        )
        self.inverse_sigma = invert_sigmas
        self.prediction_type = prediction_type

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # (Your add_noise implementation was perfect, keep it as is)
        sigmas = (
            timesteps.to(original_samples.device).float()
            / self.config.num_train_timesteps
        )
        shift = self.config.shift
        sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)

        while sigmas.ndim < original_samples.ndim:
            sigmas = sigmas.unsqueeze(-1)

        if self.inverse_sigma:
            sigmas = 1.0 - sigmas

        # noisy_samples = (1.0 - sigmas) * original_samples + sigmas * noise
        noisy_samples = (1.0 - sigmas) * noise + sigmas * original_samples
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator=None,
    ):
        if self.step_index is None:
            self._init_step_index(timestep)

        # -----------------------------------------------------------
        # NEW LOGIC: "Last Step Shortcut"
        # If this is the last step of the inference loop, trust the
        # model's prediction of clean data and return it directly.
        # -----------------------------------------------------------
        if (
            self.step_index == len(self.timesteps) - 1
            and self.prediction_type == "sample"
        ):
            # The model predicts clean data. We are at the end.
            # Just return the prediction.
            prev_sample = model_output

            # Increment index for internal tracking (required by parent class logic)
            self._step_index += 1

            if not return_dict:
                return (prev_sample,)
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

        # -----------------------------------------------------------
        # Standard Logic (for all other steps)
        # -----------------------------------------------------------
        if self.prediction_type == "sample":
            sigma = self.sigmas[self.step_index]

            # Standard clamping is still good for intermediate steps
            # just in case a step lands extremely close to 1.0
            sigma_clamped = torch.clamp(sigma, max=0.999)
            sigma_clamped = sigma_clamped.to(sample.device)

            # Convert Clean Data -> Velocity
            model_output = (model_output - sample) / (1.0 - sigma_clamped)
        # prev_sample = model_output
        # if not return_dict:
        #     return (prev_sample,)
        # return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=return_dict,
            generator=generator,
        )

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )
        if self.inverse_sigma:
            self.timesteps = self.timesteps[::-1]
