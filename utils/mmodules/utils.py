from diffusers.schedulers import *

from .common_imports import *


def extract_metrics(log_dir):
    # Initialize the event accumulator
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get the list of tags
    tags = ea.Tags()["scalars"]

    # Extract the metrics
    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]

    return metrics


def batchStepBatch(scheduler, original, noise, t):
    batch_size = original.shape[0]
    result = original.clone().detach()
    # scheduler.set_timesteps(1)
    for i in range(batch_size):
        result[i] = scheduler.step(
            model_output=noise[i],
            timestep=t[i],
            # use_clipped_model_output=True,
            sample=original[i],
        ).prev_sample

    return result


def scheduler_to_X0(scheduler, noisy, noise, t):
    if isinstance(scheduler, DPMSolverMultistepScheduler):
        return scheduler_to_X0_DPMSolverMultistepScheduler(scheduler, noisy, noise, t)
    elif isinstance(scheduler, EulerDiscreteScheduler):
        return scheduler_to_X0_EulerDiscreteScheduler(scheduler, noisy, noise, t)
    elif isinstance(scheduler, DDIMScheduler):
        return scheduler_to_X0_DDIMScheduler(scheduler, noisy, noise, t)
    elif isinstance(scheduler, DPMSolverSinglestepScheduler):
        return scheduler_to_X0_DPMSolverSinglestepScheduler(scheduler, noisy, noise, t)
    else:
        raise NotImplementedError(
            f"scheduler_to_X0 not implemented for scheduler type {type(scheduler)}"
        )


def scheduler_to_X0_DPMSolverMultistepScheduler(scheduler, noisy, noise, t):
    """
    based on the DPMSolverMultistepScheduler
    Convert a noisy sample to the original sample using the scheduler's formula.
    Based on the inverse of add_noise logic using sigmas.
    """
    # Make sure sigmas and timesteps have the same device and dtype as noisy
    sigmas = scheduler.sigmas.to(device=noisy.device, dtype=noisy.dtype)
    if noisy.device.type == "mps" and torch.is_floating_point(t):
        # mps does not support float64
        schedule_timesteps = scheduler.timesteps.to(noisy.device, dtype=torch.float32)
        t = t.to(noisy.device, dtype=torch.float32)
    else:
        schedule_timesteps = scheduler.timesteps.to(noisy.device)
        t = t.to(noisy.device)

    # Determine step indices
    if scheduler.begin_index is None:
        step_indices = [
            scheduler.index_for_timestep(ts, schedule_timesteps) for ts in t
        ]
    elif scheduler.step_index is not None:
        step_indices = [scheduler.step_index] * t.shape[0]
    else:
        step_indices = [scheduler.begin_index] * t.shape[0]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(noisy.shape):
        sigma = sigma.unsqueeze(-1)

    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)

    # noisy = alpha_t * x0 + sigma_t * noise
    x0 = (noisy - sigma_t * noise) / alpha_t

    return x0


def scheduler_to_X0_EulerDiscreteScheduler(scheduler, noisy, noise, t):
    """
    based on the EulerDiscreteScheduler
    Convert a noisy sample to the original sample using the scheduler's formula.
    Based on the inverse of add_noise logic using sigmas.
    """
    # Make sure sigmas and timesteps have the same device and dtype as noisy
    sigmas = scheduler.sigmas.to(device=noisy.device, dtype=noisy.dtype)
    if noisy.device.type == "mps" and torch.is_floating_point(t):
        # mps does not support float64
        schedule_timesteps = scheduler.timesteps.to(noisy.device, dtype=torch.float32)
        t = t.to(noisy.device, dtype=torch.float32)
    else:
        schedule_timesteps = scheduler.timesteps.to(noisy.device)
        t = t.to(noisy.device)

    # Determine step indices
    if scheduler.begin_index is None:
        step_indices = [
            scheduler.index_for_timestep(ts, schedule_timesteps) for ts in t
        ]
    elif scheduler.step_index is not None:
        step_indices = [scheduler.step_index] * t.shape[0]
    else:
        step_indices = [scheduler.begin_index] * t.shape[0]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(noisy.shape):
        sigma = sigma.unsqueeze(-1)

    # noisy = original + noise * sigma
    # original = noisy - noise * sigma
    x0 = noisy - noise * sigma

    return x0


def scheduler_to_X0_DDIMScheduler(scheduler, noisy, noise, t):
    """
    based on the DDIMScheduler
    Convert a noisy sample to the original sample using the scheduler's formula.
    Based on the inverse of add_noise logic using sigmas.
    """
    # Make sure alphas_cumprod and timestep have same device and dtype as noisy
    # Move the scheduler.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=noisy.dtype)
    t = t.to(noisy.device)

    sqrt_alpha_prod = alphas_cumprod[t] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(noisy.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    # noisy_samples = sqrt_alpha_prod * noisy + sqrt_one_minus_alpha_prod * noise
    # return noisy_samples
    x0 = (noisy - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
    return x0


def scheduler_to_X0_DPMSolverSinglestepScheduler(scheduler, noisy, noise, t):
    """
    based on the DPMSolverSinglestepScheduler
    Convert a noisy sample to the original sample using the scheduler's formula.
    Based on the inverse of add_noise logic using sigmas.
    """
    # Make sure sigmas and timesteps have the same device and dtype as noisy
    sigmas = scheduler.sigmas.to(device=noisy.device, dtype=noisy.dtype)
    if noisy.device.type == "mps" and torch.is_floating_point(t):
        # mps does not support float64
        schedule_timesteps = scheduler.timesteps.to(noisy.device, dtype=torch.float32)
        t = t.to(noisy.device, dtype=torch.float32)
    else:
        schedule_timesteps = scheduler.timesteps.to(noisy.device)
        t = t.to(noisy.device)

    # Determine step indices
    if scheduler.begin_index is None:
        step_indices = [
            scheduler.index_for_timestep(ts, schedule_timesteps) for ts in t
        ]
    elif scheduler.step_index is not None:
        step_indices = [scheduler.step_index] * t.shape[0]
    else:
        step_indices = [scheduler.begin_index] * t.shape[0]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(noisy.shape):
        sigma = sigma.unsqueeze(-1)

    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)

    # noisy = alpha_t * x0 + sigma_t * noise
    x0 = (noisy - sigma_t * noise) / alpha_t

    return x0
