import os
import time

import psutil
import torch
from pytorch_lightning.callbacks import Callback
from torch.profiler import ProfilerActivity, profile

try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False


class MeasureInferenceFlopsCallback(Callback):
    def __init__(self, profile_step=3):
        self.profile_step = profile_step
        self.profiler = None
        self.use_cuda = torch.cuda.is_available()
        self.process = psutil.Process(os.getpid())

    def on_test_start(self, trainer, pl_module):
        if self.use_cuda and HAS_DEEPSPEED:
            self.profiler = FlopsProfiler(pl_module)
        else:
            self.profiler = None

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx != self.profile_step:
            return

        print(f"Starting Profile at Step {batch_idx}...")

        # ---------------- CUDA PATH ----------------
        if self.use_cuda and self.profiler is not None:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            self.sdp_math_context = torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
            self.sdp_math_context.__enter__()

            self.profiler.start_profile()

        # ---------------- CPU / EDGE PATH ----------------
        else:
            self.cpu_start_time = time.time()
            self.cpu_start_mem = self.process.memory_info().rss

            self.cpu_profiler = profile(
                activities=[ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True,
            )
            self.cpu_profiler.start()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx != self.profile_step:
            return

        print("\n" + "=" * 80)

        # ---------------- CUDA PATH ----------------
        if self.use_cuda and self.profiler is not None:
            torch.cuda.synchronize()
            peak_mem_bytes = torch.cuda.max_memory_allocated()

            self.profiler.stop_profile()

            if hasattr(self, "sdp_math_context"):
                self.sdp_math_context.__exit__(None, None, None)

            peak_mem_mb = peak_mem_bytes / 1024 / 1024

            print("DEEPSPEED INFERENCE PROFILE (CUDA)")
            print("=" * 80)

            self.profiler.print_model_profile(
                profile_step=self.profile_step,
                module_depth=1,
                top_modules=1,
                detailed=False,
            )

            gflops = self.profiler.get_total_flops() / 1e9
            gmacs = self.profiler.get_total_macs() / 1e9
            latency_ms = self.profiler.get_total_duration() * 1000

            print(f"\nPeak CUDA memory during inference: {peak_mem_mb:.2f} MB")
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        "inference/peak_memory_cuda_mb": float(peak_mem_mb),
                        "inference/gflops_cuda": float(gflops),
                        "inference/gmacs_cuda": float(gmacs),
                        "inference/latency_cuda_ms": float(latency_ms),
                    },
                    step=batch_idx,
                )

            self.profiler.end_profile()

        # ---------------- CPU / EDGE PATH ----------------
        else:
            self.cpu_profiler.stop()

            cpu_end_time = time.time()
            cpu_peak_mem = self.process.memory_info().rss

            latency_ms = (cpu_end_time - self.cpu_start_time) * 1000
            peak_mem_mb = cpu_peak_mem / 1024 / 1024

            print("CPU / EDGE INFERENCE PROFILE")
            print("=" * 80)
            print(f"Inference time: {latency_ms:.2f} ms")
            print(f"Peak RSS memory: {peak_mem_mb:.2f} MB")

            print("\nTop memory-consuming operations:")
            print(
                self.cpu_profiler.key_averages().table(
                    sort_by="self_cpu_memory_usage", row_limit=10
                )
            )

            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        "inference/latency_cpu_ms": float(latency_ms),
                        "inference/peak_memory_cpu_mb": float(peak_mem_mb),
                    },
                    step=batch_idx,
                )

        if trainer.logger is not None:
            if hasattr(trainer.logger, "save"):
                trainer.logger.save()

        exit(0)
