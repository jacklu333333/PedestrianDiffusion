import datetime
import json
import os
import random
import shutil
import smtplib
import socket
import stat
import time
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional

import colored as cl
import deepspeed
import numpy as np
import pytorch_lightning as pl
import requests
import torch
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch import Tensor

from ..mdatasets import INV_DATASET_DICT
from ..mmodules import VAESpectrum, diffusionSpectrum
from ..plot_trajectory import animation_fun
from ..transform import integrate_orientation

"""
progress bar with time
"""
