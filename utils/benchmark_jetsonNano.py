import time

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch.cuda.amp import autocast
from tqdm import tqdm
from transform import batchFrequencyToTime, batchTimeToFrequency

NUM_OF_STEPS = 1000


class dummymodel(torch.nn.Module):
    def __init__(self):
        super(dummymodel, self).__init__()
        self.model = UNet2DModel(
            sample_size=(32, 32),
            in_channels=6,
            out_channels=6,
            # center_input_sample = False,
            # time_embedding_type = "positional",
            # freq_shift = 0,
            # flip_sin_to_cos = True,
            down_block_types=(
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                # "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                # "AttnUpBlock2D",
            ),
            # block_out_channels=(np.array([224, 448, 672, 896])).tolist(),
            block_out_channels=(32, 64, 128),
            layers_per_block=3,
            # mid_block_scale_factor= 1,
            downsample_padding=1,
            downsample_type="resnet",
            upsample_type="resnet",
            dropout=0.1,
            act_fn="mish",
            # attention_head_dim = 8,
            norm_num_groups=32,
            # attn_norm_num_groups = None,
            norm_eps=1e-5,
            # resnet_time_scale_shift = "default",
            # add_attention = True,
            # class_embed_type = None,
            # num_class_embeds = None,
            num_train_timesteps=1000,
        )
        # self.toFrequency = TimeToFrequency()
        # self.toTime = FrequencyToTime()

    def forward(self, x, timesteps):
        # x_freq = self.toFrequency(x)
        # x_freq = self.model(x_freq, timesteps)
        # x_time = self.toTime(x_freq)
        return self.model(x, timesteps)


def benchmark_unet2d_jetson(precision="fp32"):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBenchmarking {precision.upper()} precision on {device}")
    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_OF_STEPS,
    )

    # Initialize the model
    model = dummymodel()

    # Convert model to appropriate precision
    if precision == "fp16" and device.type == "cuda":
        model = model.half()

    model = model.to(device)
    model.eval()
    model = torch.compile(model, mode="max-autotune")

    # Create sample input
    batch_size = 1
    sample_input = torch.randn(batch_size, 6, 32, 32).to(device)
    if precision == "fp16" and device.type == "cuda":
        sample_input = sample_input.half()

    # Warm-up run
    with torch.no_grad():
        if precision == "fp16" and device.type == "cuda":
            with autocast():
                for _ in range(10):
                    _ = model(
                        sample_input,
                        torch.randint(
                            0, NUM_OF_STEPS, (sample_input.shape[0],), device=device
                        ),
                    )["sample"]
        else:
            for _ in range(10):
                _ = model(
                    sample_input,
                    torch.randint(
                        0, NUM_OF_STEPS, (sample_input.shape[0],), device=device
                    ),
                )["sample"]

    # Benchmark
    num_steps = NUM_OF_STEPS
    times = []

    print(f"Starting {precision.upper()} benchmark for {num_steps} steps...")

    with torch.no_grad():
        for _ in tqdm(range(num_steps)):
            if precision == "fp16" and device.type == "cuda":
                with autocast():
                    start_time = time.time()
                    output = model(
                        sample_input,
                        torch.randint(
                            0, NUM_OF_STEPS, (sample_input.shape[0],), device=device
                        ),
                    )["sample"]
                    output = scheduler.step(
                        model_output=output,
                        timestep=_,
                        sample=sample_input,
                    )["prev_sample"]
                    end_time = time.time()
            else:
                start_time = time.time()
                output = model(
                    sample_input,
                    torch.randint(
                        0,
                        NUM_OF_STEPS,
                        (sample_input.shape[0],),
                        device=device,
                    ),
                )["sample"]
                output = scheduler.step(
                    model_output=output,
                    timestep=_,
                    sample=sample_input,
                )["prev_sample"]
                end_time = time.time()
            times.append(end_time - start_time)
            sample_input = output

    # Calculate statistics
    total_time = sum(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    inferences_per_second = 1.0 / avg_time

    # Print results
    print(f"\n{precision.upper()} Benchmark Results:")
    print(f"Total time for {num_steps} steps: {total_time:.2f} seconds")
    print(f"Average time per inference: {avg_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Inferences per second: {inferences_per_second:.2f}")

    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"\n{precision.upper()} GPU Memory Usage:")
        print(f"Memory Allocated: {memory_allocated:.2f} MB")
        print(f"Memory Reserved: {memory_reserved:.2f} MB")

    return total_time, avg_time, inferences_per_second


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        # Benchmark FP32
        fp32_total, fp32_avg, fp32_ips = benchmark_unet2d_jetson("fp32")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Benchmark FP16
        fp16_total, fp16_avg, fp16_ips = benchmark_unet2d_jetson("fp16")

        # Print comparison
        print("\nComparison Summary:")
        print(f"FP32 Avg Time: {fp32_avg:.4f}s | FP16 Avg Time: {fp16_avg:.4f}s")
        print(f"FP32 IPS: {fp32_ips:.2f} | FP16 IPS: {fp16_ips:.2f}")
        print(f"Speedup (FP16/FP32): {fp32_avg/fp16_avg:.2f}x")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
