from diffusers.models import (
    UNet2DConditionModel,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
)

from .PedestrianDiffusion import PedestrianDiffusion
from .common_imports import *
from .utils import *


class PedestrianDiffusionSingleDomain(PedestrianDiffusion):
    def __init__(self, config):
        super(PedestrianDiffusionSingleDomain, self).__init__(config)
        self.special_loss = spectrumMultiTaskLossSingleDomain(
            dt=1.0 / self.config["sampling_rate"]
        )
