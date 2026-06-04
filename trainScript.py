import argparse
import datetime
import json
import os
import socket
import stat
import traceback
from datetime import timedelta
from pathlib import Path

import colored as cl
import deepspeed
import pytorch_lightning as pl
import torch
import torch.optim.swa_utils

torch.serialization.add_safe_globals([getattr, torch.optim.swa_utils.SWALR])
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

import utils.mmodules as MYMODELs
from utils.ganModel import GANModule
from utils.mcallbacks import (
    MyProfStep,
    RobustCheckpointCallback,
    TelegramCallback,
    TimeFilterProgressBar,
    TrajectoryTestMetricsSummaryCallback,
    TrajectoryTestResultHandler,
    copyFileNLoadWeightLogger,
)
from utils.mdatasets import odomDataModule
from utils.parser import parse_arguments
from utils.utility import set_seed

SEED = 42
set_seed(SEED)


# deepspeed_config = {
#     "train_batch_size": 8,
#     "fp16": {"enabled": True},
#     "zero_optimization": {
#         "stage": 2,
#         "reduce_scatter": True,
#         "allgather_bucket_size": 500000000,
#         "reduce_bucket_size": 500000000,
#         "contiguous_gradients": True,
#         "overlap_comm": True,
#         "offload_optimizer": {"device": "cpu"},
#         "offload_param": {"device": "cpu", "pin_memory": True},
#     },
# }


def trainerConstructor(config=None, logger=None):
    deepspeed_cfg = {
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 5e7,
            # Explicitly disable CPU offloading to keep optimizer on GPU
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        # "fp16": {"enabled": True},
    }
    # Save the original batch size to prevent multiple divisions if trainerConstructor is called again
    if "original_batch_size" not in config:
        config["original_batch_size"] = config["batch_size"]

    # In DDP child processes, torch.cuda.device_count() may return 1 because PL sets CUDA_VISIBLE_DEVICES.
    # We should use WORLD_SIZE from the environment if available.
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    config["batch_size"] = max(1, config["original_batch_size"] // world_size)
    # if config["train_phase"] == "finetune":
    #     config["ema"][
    #         "enable"
    #     ] = False  # Disable EMA during finetuning to allow the model to adapt quickly

    trainer = pl.Trainer(
        # devices=int(os.environ.get("WORLD_SIZE", 1)),
        # devices=torch.cuda.device_count(),
        # num_nodes=int(os.environ.get("NUMBER_OF_NODES", 1)),
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_time=(
            timedelta(hours=5) if "RoNIN" in config["dataset"] else timedelta(hours=12)
        ),
        # overfit_batches=5,
        # limit_train_batches=5,
        # limit_val_batches=1,
        # limit_test_batches=1,
        # min_epochs=100,
        # min_epochs=10,
        # max_epochs=10,
        # max_epochs=config["max_epochs"],
        # max_time=datetgl4xup6
        # enable_checkpointing=False,
        # devices=1,si=53fZaR5PzQCiDnYV
        # strategy=pl.strategies.DDPStrategy(
        #     find_unused_parameters=True,
        # ),
        # strategy=pl.strategies.DeepSpeedStrategy(config=deepspeed_config),
        # strategy=DeepSpeedStrategy(
        #     stage=2,
        #     timeout=timedelta(hours=2),
        #     gradient_as_bucket_view=True,
        #     ddp_bucket_cap_mb=25,
        # ),
        # strategy=(
        #     DeepSpeedStrategy(
        #         config=deepspeed_cfg,
        #         # timeout=timedelta(hours=2),
        #     )
        #     if config["model"] != "cycleGan"
        #     else DDPStrategy(find_unused_parameters=True)
        # ),
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy="auto",
        # num_nodes=int(os.environ.get("WORLD_SIZE", 1)),
        # strategy="deepspeed_stage_2_offload",
        # strategy="deepspeed_stage_3_offload",
        # strategy=DeepSpeedStrategy(config=deepspeed_config),
        # min_epochs=10,
        # add early stopping
        # max_steps=6,
        callbacks=[
            # StochasticWeightAveraging(
            #     # 1. The Learning Rate
            #     # Recommendation: 50% of your max training LR.
            #     # Since your base is 1e-4, use 5e-5.
            #     # This is high enough to explore the basin, but safe for diffusion.
            #     swa_lrs=5e-5,
            #     # 2. When to start (The Plateau)
            #     # Look at your graph. The plateau starts roughly 60-70% through training.
            #     # If you train for 100 epochs, set this to 0.7 or roughly epoch 70.
            #     swa_epoch_start=3,
            #     # 3. Smooth Transition
            #     # Anneal from 1e-4 down to 5e-5 over 10 epochs to prevent "shocking" the model.
            #     annealing_epochs=3,
            #     annealing_strategy="cos",
            # ),
            EarlyStopping(
                # monitor="loss_total/val",
                monitor="metric_naive_distance_error_XY/val_mean",
                patience=10,
                mode="min",
                min_delta=1e-4,  # can be 0.001 if doing good
                verbose=True,
            ),
            # EarlyStopping(
            #     monitor="metric_naive_distance_error_XY/val_mean",
            #     patience=10,
            #     mode="min",
            #     min_delta=1e-3,  # can be 0.001 if doing good
            #     verbose=True,
            # ),
            # EarlyStopping(
            #     monitor="loss_total/train_step",
            #     patience=1000 * config["accumulate_grad_batches"],
            #     mode="min",
            #     min_delta=1e-3,  # can be 0.001 if doing good
            #     verbose=True,
            # ),
            # EarlyStopping(
            #     monitor="loss_TimeSUM_acc/val",
            #     patience=5,
            #     mode="min",
            #     min_delta=1e-3,  # mm
            #     verbose=True,
            # ),
            # EarlyStopping(
            #     monitor="loss_TimeSUM_gyr/val",
            #     patience=5,
            #     mode="min",
            #     min_delta=1,  # degree
            #     verbose=True,
            # ),
            # monitor="metric_naive_distance_error
            # EarlyStopping(
            #     monitor="loss_TimeSIM_acc/train_step",
            #     patience=1000 * config["accumulate_grad_batches"],
            #     mode="min",
            #     min_delta=1e-3,  # can be 0.001 if doing good
            #     verbose=True,
            # ),
            # EarlyStopping(
            #     monitor="loss_TimeSIM_gyr/train_step",
            #     patience=1000 * config["accumulate_grad_batches"],
            #     mode="min",
            #     min_delta=1e-3,  # can be 0.001 if doing good
            #     verbose=True,
            # ),
            # EarlyStopping(
            #     monitor="loss_TimeSIM_acc/val",
            #     patience=5,
            #     mode="min",
            #     min_delta=1e-3,  # can be 0.001 if doing good
            #     verbose=True,
            # ),
            # EarlyStopping(
            #     monitor="loss_TimeSIM_gyr/val",
            #     patience=5,
            #     mode="min",
            #     min_delta=1e-3,  # can be 0.001 if doing good
            #     verbose=True,
            # ),
            # pl.callbacks.EarlyStopping(
            #     monitor="loss/val", patience=3, mode="min", min_delta=0.001
            # ),
            # pl.callbacks.EarlyStopping(
            #     monitor="l1/val", patience=3, mode="min", min_delta=0.0000001
            # ),
            # pl.callbacks.EarlyStopping(
            #     monitor="l2/val", patience=3, mode="min", min_delta=0.0000001
            # ),
            RobustCheckpointCallback(
                # monitor="loss_total/val",
                # filename="model_loss_total_val{loss_total/val:05.8f}_step{step:02d}_epoch{epoch:02d}",
                monitor="metric_naive_distance_error_XY/val_mean",
                filename="model_metric_naive_distance_error_XY_val{metric_naive_distance_error_XY/val_mean:05.8f}_step{step:02d}_epoch{epoch:02d}",
                save_top_k=3,
                mode="min",
                auto_insert_metric_name=False,
            ),
            TimeFilterProgressBar(
                bold_keywords=[
                    "loss_TimeSUM_acc/train_step",
                    "loss_TimeSIM_acc/train_step",
                    "loss_TimeSUM_gyr/train_step",
                    "loss_TimeSIM_gyr/train_step",
                    # "acc_pos/train_step",
                    # "gyr_pos/train_step",
                ],
                keep_keywords=[
                    "acc_pos",
                    "gyr_pos",
                    "v_num",
                    "loss",
                    "time_sim",
                    "freq_sim",
                    "regularization",
                ],
                remove_keywords=["train_epoch", "metric"],
            ),
            copyFileNLoadWeightLogger(main_file_name=Path(__file__).name),
            # TelegramCallback(
            #     bot_token=json.load(open("telegram.json", "r"))["token"],
            #     chat_id=json.load(open("telegram.json", "r"))["chat_id"],
            # ),
            # MyProfStep(),
            TrajectoryTestResultHandler(
                video_rendering=config["video_rendering"],
                sampling_rate=config["sampling_rate"],
            ),
            TrajectoryTestMetricsSummaryCallback(),
        ],
        logger=logger,
        sync_batchnorm=True,
        gradient_clip_val=config["gradient_clip_val"],
        gradient_clip_algorithm=config["gradient_clip_algorithm"],
        # check_val_every_n_epoch=None,
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        # val_check_interval=min(number_of_batches - 1, 1000),
        # val_check_interval=1000,
        num_sanity_val_steps=1,
        detect_anomaly=False,
    )
    return trainer


if __name__ == "__main__":
    try:
        config = parse_arguments()

        name = f'diffusion_{config["dataset"]}'
        if config["model"].lower() != "inertialBridgemodule":
            name = f'diffusion_{config["dataset"]}_{config["inference_method"]}'

        logger = pl.loggers.TensorBoardLogger(
            f"logs_PDFamily/{config['model']}",
            name=name,
        )

        dm = odomDataModule(
            config=config,
            data_dir="./datasets/" + config["dataset"],
        )
        model = getattr(MYMODELs, config["model"])(config=config)


        trainer = trainerConstructor(config=config, logger=logger)

        if config["lr_finder"]:
            tuner = pl.tuner.tuning.Tuner(trainer)
            # find the best learning rate
            lr_finder = tuner.lr_find(
                model,
                dm,
                min_lr=1e-10,
                max_lr=1e-2,
                mode="exponential",
                update_attr=True,
                attr_name="lr",
            )
        if int(trainer.global_rank) == 0:
            # check log_dir exists
            if not os.path.exists(logger.log_dir):
                os.makedirs(logger.log_dir)

            if config["lr_finder"]:
                fig = lr_finder.plot(suggest=True)
                fig.savefig(logger.log_dir + "/lr_finder.png")

        trainer.fit(
            model,
            dm,
            ckpt_path=(
                config["base_weight_path"] if config["base_weight_path"] != "" else None
            ),
        )

        print(
            cl.Fore.green
            + f"Complete ! Rank {trainer.global_rank+1}/{trainer.world_size} Initial Training"
            + cl.Style.reset
        )
        if config["dual_stage"]:
            trainer.test(model, dm, ckpt_path="best")
            # last_lr = trainer.optimizers[0].param_groups[0]["lr"]
            latest_ckpt_path = trainer.checkpoint_callback.best_model_path
            del model, trainer
            logger = pl.loggers.TensorBoardLogger(
                f"logs_PDFamily/{config['model']}",
                name=name,
                version=logger.log_dir.split("/")[-1] + "/second_stage",
            )
            config["train_phase"] = "finetune"
            # config["lr"] = last_lr
            config["base_weight_path"] = latest_ckpt_path
            model = getattr(MYMODELs, config["model"])(config=config)
            trainer = trainerConstructor(config=config, logger=logger)

            trainer.fit(
                model=model,
                datamodule=dm,
            )
            print(
                cl.Fore.green
                + f"Complete ! Rank {trainer.global_rank+1}/{trainer.world_size} finetuning"
                + cl.Style.reset
            )
        trainer.test(model, dm, ckpt_path="best")
        print(
            cl.Fore.green
            + f"Complete ! Rank {trainer.global_rank+1}/{trainer.world_size} Testing"
            + cl.Style.reset
        )
    except Exception as e:
        print(cl.Fore.red, e, cl.Style.reset)
        traceback.print_exc()
    finally:
        if int(trainer.global_rank) == 0:
            file_name = f"diffusion_{socket.gethostname()}.log"
            os.system(f"cp {file_name} {logger.log_dir}")
            os.chmod(
                os.path.join(logger.log_dir, f"{file_name}"),
                stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH,
            )
            # os.remove(f"{file_name}")
