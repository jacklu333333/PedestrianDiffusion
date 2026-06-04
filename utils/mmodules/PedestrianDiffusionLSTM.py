from utils.ronin.source.model_temporal import LSTMSeqNetwork, BilinearLSTMSeqNetwork
from .PedestrianDiffusion import PedestrianDiffusion
from .common_imports import *
from .utils import *


class PedestrianDiffusionLSTM(PedestrianDiffusion):
    def __init__(self, config):
        super().__init__(config)
        self.toObservation = nn.Sequential(
            Rearrange("b c f t I -> b t ( c t I)", I=2, f=16, c=12),
        )
        if hasattr(self, "deObservation"):
            del self.deObservation
        self.deObservation = nn.Sequential(
            Rearrange("b t (c t I) ->  b c f t I", I=2, f=16, c=6),
        )
        del self.model
        self.model = BilinearLSTMSeqNetwork(
            input_size=284,
            out_size=192,
            batch_size=self.config["batch_size"],
            device=torch.device("cpu"),  # will update dynamically
            lstm_size=self.config.get(
                "lstm_size", 100
            ),  # hidden layer size of the LSTM
            lstm_layers=self.config.get("lstm_layers", 3),
            dropout=self.config.get("dropout", 0.1),
        )
