import argparse
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
import torch.utils
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import utils.mmodules as MYMODELs
from utils.mcallbacks import (
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

if __name__ == "__main__":
    try:
        config = parse_arguments()

        # if config["base_weight_path"] == "":
        # config["model"] = "baseLineTestingPL"

        model = getattr(MYMODELs, config["model"])(config=config)

        if isinstance(model, MYMODELs.baseLineTestingPL):
            version = "baseline"
        else:
            version = None

        name = f'diffusion_{config["dataset"]}'
        if config["model"].lower() != "inertialBridgemodule":
            name = f'diffusion_{config["dataset"]}_{config["inference_method"]}'

        logger = pl.loggers.TensorBoardLogger(
            (
                "logs/"
                # "baseline/"
                if config["base_weight_path"] == ""
                else f'{Path(config["base_weight_path"]).parent.parent}/testing'
            ),
            name=name,
            version=version,
        )

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
        #     model.load_state_dict(state_dict, strict=False)
        dm = odomDataModule(
            config=config,
            # data_dir=(
            #     "./datasets/" + config["dataset"]
            #     if socket.gethostname().upper() == "ROGUEONE"
            #     else "/home/jack/Downloads/" + config["dataset"]
            # ),
            data_dir="./datasets/" + config["dataset"],
        )


        trainer = pl.Trainer(
            # limit_test_batches=1,
            # devices=1,
            # Use a DDP strategy object to configure the timeout
            strategy=(
                DDPStrategy(
                    timeout=timedelta(hours=2),
                    find_unused_parameters=True,
                )
                if config["model"] != "cycleGan"
                else DDPStrategy(find_unused_parameters=True)
            ),
            # strategy="deepspeed_stage_2",
            # strategy="deepspeed_stage_2_offload",
            # strategy="ddp_find_unused_parameters_true",
            # add early stopping
            callbacks=[
                EarlyStopping(
                    monitor="loss/val", patience=3, mode="min", min_delta=0.001
                ),
                ModelCheckpoint(
                    monitor="loss/val",
                    save_top_k=3,
                    mode="min",
                    filename="model_val_loss{loss/val:05.8f}_epoch{epoch:02d}",
                    auto_insert_metric_name=False,
                ),
                TimeFilterProgressBar(
                    keep_keywords=["v_num", "loss"],
                    remove_keywords=["train_epoch", "total_loss/val"],
                ),
                copyFileNLoadWeightLogger(main_file_name=Path(__file__).name),
                # TelegramCallback(
                #     bot_token=json.load(open("telegram.json", "r"))["token"],
                #     chat_id=json.load(open("telegram.json", "r"))["chat_id"],
                # ),
                TrajectoryTestResultHandler(
                    video_rendering=config["video_rendering"],
                    sampling_rate=config["sampling_rate"],
                ),
                TrajectoryTestMetricsSummaryCallback(),
            ],
            # limit_test_batches=config["limit_test_batches"],
            logger=logger,
            sync_batchnorm=True,
            gradient_clip_val=config["gradient_clip_val"],
            gradient_clip_algorithm=config["gradient_clip_algorithm"],
            # inference_mode=True,
        )

        if "base_weight_path" in config and config["base_weight_path"] != "":
            trainer.test(model, dm)
        else:
            print(
                cl.Fore.red,
                "!!! Using the random weight for Testing !!!",
                cl.Style.reset,
            )
            trainer.test(model, dm, ckpt_path=None)
        print(
            cl.Fore.green
            + f"Complete ! Rank {trainer.global_rank+1}/{trainer.world_size}"
            + cl.Style.reset
        )
    except Exception as e:
        print(cl.Fore.red, e, cl.Style.reset)
        traceback.print_exc()
    # if os.getenv("LOCAL_RANK", "0") == "0" and os.getenv("NODE_RANK", "0") == "0":
    finally:
        if int(trainer.global_rank) == 0:
            file_name = f"diffusion_test_{socket.gethostname()}.log"
            os.system(f"cp {file_name} {logger.log_dir}")
            os.chmod(
                os.path.join(logger.log_dir, f"{file_name}"),
                stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH,
            )
            # os.remove(f"{file_name}")
