import datetime
import glob
import json
import math
import os
import pickle
import random
import re
from pathlib import Path

import colored as cl
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import quaternion
import scipy.signal as signal
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
from einops import rearrange
from einops.layers.torch import Rearrange
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.spatial.transform import Rotation as R
from torch.nn.modules.container import Sequential
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm
from transformers import (  # a generic “perceiver‐io” model that can handle 3D inputs
    CLIPModel,
    CLIPTextModel,
    CLIPTokenizer,
    PerceiverConfig,
    PerceiverModel,
)

from ..correction import rotateToWorldFrame
from ..geoPreprocessor import GravityRemoval, MagneticRemoval
from ..stepDetection import find_steps
from ..transform import *
from ..transform import batchTimeToFrequency

# # check cudf installation
# try:
#     import cudf
#     import cudf.pandas

#     # cudf.pandas.install()
# except:
#     pass


"""
link:https://github.com/rapidsai/cudf
cheat sheet for loading data into cudf dataframes
----------------------------------------------------------------------------------
import cudf

tips_df = cudf.read_csv("https://github.com/plotly/datasets/raw/master/tips.csv")
tips_df["tip_percentage"] = tips_df["tip"] / tips_df["total_bill"] * 100

# display average tip by dining party size
# print(tips_df.groupby("size").tip_percentage.mean())
"""
MAX_FRAME_LENGTH = 100
# SCALE = {
#     # "RoNIN": {
#     #     "acc": 60.0,
#     #     "gyr": 25.0,
#     #     "label": 60,
#     # },
#     "RoNIN": {
#         "acc": 4.0,
#         "gyr": 4.0,
#         "label": 4.0,
#     },
#     # "OIOD": {
#     #     "acc": 130.0,
#     #     "gyr": 10.0,
#     #     "label": 130.0,
#     # },
#     "OxIOD": {
#         "acc": 4.0,
#         "gyr": 4.0,
#         "label": 4.0,
#     },
# }

DATASET_DICT = {
    "unknown": 0,
    "RIDI": 2,
    "RoNIN": 3,
    "RoNIN_unseen": 4,
    "OxIOD_VICON": 5,
    "OxIOD_tango": 6,
    "OxIOD": 7,
    "TLIO": 8,
}
INV_DATASET_DICT = {v: k for k, v in DATASET_DICT.items()}


from os import path as osp
