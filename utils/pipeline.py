import torch
from diffusers import DiffusionPipeline


class MyCustomPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, vae=None, tokenizer=None, text_encoder=None):
        super().__init__()
        # Register components so they can be saved/loaded
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            # tokenizer=tokenizer,
            # text_encoder=text_encoder,
        )

    @torch.no_grad()
    def __call__(self, prompt_embeds, num_inference_steps=50, guidance_scale=7.5):
        # 1. Prepare latents (random noise)
        batch_size = prompt_embeds.shape[0]
        latents = torch.randn(
            (
                batch_size,
                self.unet.in_channels,
                self.unet.sample_size,
                self.unet.sample_size,
            ),
            device=self.device,
        )

        # 2. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # 3. Denoising loop
        for t in self.scheduler.timesteps:
            # Predict noise residual
            noise_pred = self.unet(
                latents, t, encoder_hidden_states=prompt_embeds
            ).sample

            # (Optional) Classifier-Free Guidance
            if guidance_scale != 1.0:
                # Run unconditional branch
                uncond_pred = self.unet(latents, t, encoder_hidden_states=None).sample
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Decode latents to image (if using VAE)
        if self.vae is not None:
            images = self.vae.decode(latents / 0.18215).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            return images
        else:
            return latents
