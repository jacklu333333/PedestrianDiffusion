import argparse
import datetime
import json
import os
import socket
import stat
import traceback
from pathlib import Path

import colored as cl
import deepspeed
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy

import utils.mmodules as MYMODELs
from utils.mcallbacks import (
    PosteriorCollapseStopper,
    RobustCheckpointCallback,
    TimeFilterProgressBar,
    copyFileNLoadWeightLogger,
)
from utils.mdatasets import odomDataModule
from utils.mmodules import DiffusionModelPL, VAESpectrum
from utils.parser import parse_arguments
from utils.utility import set_seed

set_seed(0)


if __name__ == "__main__":
    try:
        config = parse_arguments()
        # config["model"] = "VAETime"
        assert (
            "vae" in config["model"].lower()
        ), "This script is only for VAE with IMU training"

        logger = pl.loggers.TensorBoardLogger(
            f'logs_{config["model"]}/',
            name=f'{config["dataset"]}_imu',
        )

        # model = DiffusionModelPL(config=config)
        model = getattr(MYMODELs, config["model"])(config=config, target_modes="x")
        # if config["base_weight_path"] != "":
        #     # check is it a file or not
        #     if not os.path.isfile(config["base_weight_path"]):
        #         print(cl.Fore.yellow + "Using deepspeed to load the model" + cl.Style.reset)
        #         state_dict = (
        #             deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
        #                 config["base_weight_path"]
        #             )
        #         )
        #     else:
        #         state_dict = torch.load(
        #             config["base_weight_path"],
        #             weights_only=False,
        #         )["state_dict"]
        #     model.load_state_dict(state_dict)

        dm = odomDataModule(
            config=config,
            data_dir="./datasets/" + config["dataset"],
        )


        trainer = pl.Trainer(
            # devices=2,
            strategy="auto",
            # strategy="deepspeed_stage_2",
            # strategy="deepspeed_stage_2_offload",
            # strategy=DeepSpeedStrategy(config=deepspeed_config),
            # min_epochs=200,
            # max_epochs=1,
            # max_time=datetime.timedelta(days=1),
            # add early stopping
            callbacks=[
                EarlyStopping(
                    monitor="loss_total/val",
                    patience=10,
                    mode="min",
                    min_delta=1e-3,  # can be 0.001 if doing good
                ),
                # EarlyStopping(
                #     monitor="loss_total/train",
                #     patience=1000,
                #     mode="min",
                #     min_delta=1e-3,  # can be 0.001 if doing good
                # ),
                # EarlyStopping(
                #     monitor="recon_loss_acc/val",
                #     patience=3,
                #     mode="min",
                #     min_delta=1e-3,
                # ),
                # EarlyStopping(
                #     monitor="recon_loss_gyr/val",
                #     patience=3,
                #     mode="min",
                #     min_delta=1e-3,
                # ),
                # EarlyStopping(
                #     monitor="loss/val", patience=3, mode="min", min_delta=0.001
                # ),
                # EarlyStopping(
                #     monitor="l1/val", patience=3, mode="min", min_delta=0.0000001
                # ),
                # EarlyStopping(
                #     monitor="l2/val", patience=3, mode="min", min_delta=0.0000001
                # ),
                RobustCheckpointCallback(
                    monitor="loss_total/val",
                    save_top_k=3,
                    mode="min",
                    filename="model_loss_val{loss_total/val:05.8f}_step{step:02d}_epoch{epoch:02d}",
                    auto_insert_metric_name=False,
                ),
                TimeFilterProgressBar(
                    bold_keywords=["recon_loss_acc", "recon_loss_gyr"],
                    keep_keywords=["v_num", "loss", "mse"],
                    remove_keywords=[
                        "train_epoch",
                    ],
                ),
                copyFileNLoadWeightLogger(main_file_name=Path(__file__).name),
                PosteriorCollapseStopper(),
            ],
            logger=logger,
            # sync_batchnorm=True,
            gradient_clip_val=config["gradient_clip_val"],
            gradient_clip_algorithm=config["gradient_clip_algorithm"],
            # limit_train_batches=1,
            # limit_val_batches=100,
            # val_check_interval=500,
            # limit_test_batches=10,
            # detect_anomaly=True,
        )
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

        trainer.fit(model, dm)
        print(
            cl.Fore.green
            + f"Complete ! Rank {trainer.global_rank+1}/{trainer.world_size} Initial Training"
            + cl.Style.reset
        )
        if config["dual_stage"]:
            del model
            config["train_phase"] = "finetune"
            model = getattr(MYMODELs, config["model"])(config=config, target_modes="x")
            trainer.fit(
                model=model,
                datamodule=dm,
                ckpt_path=trainer.checkpoint_callback.best_model_path,
            )
            print(
                cl.Fore.green
                + f"Complete ! Rank {trainer.global_rank+1}/{trainer.world_size} finetuning"
                + cl.Style.reset
            )
        # trainer.test(model, dm)
        # trainer.test(model, dm, ckpt_path=config["base_weight_path"])
        trainer.test(model, dm, ckpt_path="best")
        print(
            cl.Fore.green
            + f"Complete ! Rank {trainer.global_rank+1}/{trainer.world_size} testing"
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
            os.remove(f"{file_name}")

    # if os.getenv("LOCAL_RANK", "0") == "0" and os.getenv("NODE_RANK", "0") == "0":
    # switch to last rank is complete
    # if int(trainer.world_size) == (int(trainer.global_rank) + 1):
    # except Exception as e:
    #     print(cl.Fore.red, e, cl.Style.reset)
    # if int(trainer.global_rank) == 0:
    #     file_name = f"diffusion_{socket.gethostname()}.log"
    #     os.system(f"cp {file_name} {logger.log_dir}")
    #     os.chmod(
    #         os.path.join(logger.log_dir, f"{file_name}"),
    #         stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH,
    #     )
    #     os.remove(f"{file_name}")
