"""
reference:
https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946

"""

import io
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import colored as cl
import deepspeed
import diffusers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  # for color scales
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

try:
    from cuml.decomposition import PCA
    from cuml.manifold import TSNE, UMAP
except ImportError:
    print(
        "\033[93m[Warning] Unable to find the 'cuml' library. Some features may be unavailable.\033[0m"
    )
from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
)
from einops import rearrange
from einops.layers.torch import Rearrange

# from mixture_of_experts import HeirarchicalMoE, MoE
from overrides import overrides
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
from tensorboard.backend.event_processing import event_accumulator
from timm.utils import ModelEmaV3
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import (
    CosineSimilarity,
    KLDivergence,
    MeanAbsoluteError,
    MeanSquaredError,
)
from torchmetrics.audio import SignalNoiseRatio
from tqdm import tqdm

from utils.transform import *

from ..activation import limiterActivation
from ..IoNet.ionet import CustomMultiLoss, IoNet
from ..IoNet.utils import euler_to_quaternion, quaternion_to_euler
from ..mconditionalUnet2D import mUNet2DConditionModel
from ..mdatasets import DATASET_DICT, INV_DATASET_DICT
from ..mloss import *
from ..plot_trajectory import animation_fun
from ..scheduler import (
    gen_linear_beta_t,
    gen_log_beta_t,
    mDDPMScheduler,
    mDPMSolverMultistepScheduler,
)
from ..transform import (
    IMUToIntensityQuaternion,
    IMUToYUV,
    IntensityQuaternionToIMU,
    YUVToIMU,
    batchFrequencyToTime,
    batchTimeToFrequency,
    deObservation,
    toObservation,
)
from ..unet1d import BiLSTM, Discriminator1D, UNet1D
from ..utility import find_closest_factors, weight_finder
