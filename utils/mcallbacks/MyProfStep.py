from .common_imports import *


class MyProfStep(Callback):
    def __init__(self):
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            with_stack=True,
            record_shapes=True,
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # print(f"MyProfStep Step end, {batch_idx}")
        # print(f"class prof: {id(self.prof)}")
        self.prof.step()

    def on_fit_start(self, trainer, pl_module):
        self.prof.start()

    def on_fit_end(self, trainer, pl_module):
        self.prof.stop()
        # print("fit end")
        if int(trainer.global_rank) == 0:
            # print("exporting trace")
            save_path = os.path.join(
                trainer.logger.log_dir, f"prof_trace{trainer.global_rank}.json"
            )
            rank_zero_info(
                cl.Fore.green + f"exporting trace to {save_path}" + cl.Style.reset
            )
            # self.prof.export_chrome_trace(f"/tmp/example-{trainer.global_rank}.json")
            self.prof.export_chrome_trace(save_path)
