from .common_imports import *


class myspecialLoss(nn.Module):
    def __init__(self, config):
        super(myspecialLoss, self).__init__()
        self.config = config
        self.history = {
            "train": [],
            "val": [],
            "test": [],
        }

        self.hubber_loss = nn.HuberLoss(delta=1)
        self.mse_loss = nn.MSELoss()
        self.pearson_loss = PearsonLoss()
        self.simclr = simclr_loss()
        # self.distance_cumsum_loss = DistanceCumsumLoss()
        self.naive_distance_error = NaiveDistanceError()
        self.naive_distance_error_X = NaiveDistanceError_X()
        self.naive_distance_error_Y = NaiveDistanceError_Y()
        self.naive_distance_error_Z = NaiveDistanceError_Z()
        self.accerlation_loss = accelerationError()
        self.velocity_loss = velocityError()
        self.position_loss = positionError()
        self.restoration_regulization = nn.MSELoss()

    def clear_history(self, mode="train"):
        self.history[mode].clear()

    def forward(
        self, x, y, loss_identity, restoration, l1, l2, pad_size, aux_loss, mode
    ) -> dict:
        l1_out = l1
        l2_out = l2

        hubber_loss = self.hubber_loss(x, y)
        mse_loss = self.mse_loss(x, y)
        # mse_loss = self.mse_loss(x, y)

        pearson_loss = self.pearson_loss(
            # x[:, :3, pad_size:-pad_size],
            # y[:, :3, pad_size:-pad_size],
            x,
            y,
        )
        simclr_loss = self.simclr(
            # x[:, :3, pad_size:-pad_size],
            # y[:, :3, pad_size:-pad_size],
            x,
            y,
        )
        naive_distance_error = self.naive_distance_error(x, y)
        navie_distance_error_X = self.naive_distance_error_X(x, y)
        navie_distance_error_Y = self.naive_distance_error_Y(x, y)
        navie_distance_error_Z = self.naive_distance_error_Z(x, y)

        # distance_cumsum_loss = self.distance_cumsum_loss(x, y)
        accelerationLoss = self.accerlation_loss(x, y)
        velocityLoss = self.velocity_loss(x, y)
        positionLoss = self.position_loss(x, y)

        restoration_regularization = restoration.square().mean()
        # restoration_regularization = self.restoration_regulization(
        #     restoration, torch.zeros_like(restoration)
        # )

        if len(self.history[mode]) == 0:
            self.history[mode].append(
                {
                    "loss_hubber": hubber_loss.clone().detach(),
                    "loss_mse": mse_loss.clone().detach(),
                    "loss_pearson": pearson_loss.clone().detach(),
                    "loss_simclr": simclr_loss.clone().detach(),
                    "loss_l1": l1_out.clone().detach(),
                    "loss_l2": l2_out.clone().detach(),
                    "loss_aux": aux_loss.clone().detach(),
                    "naive_distance_error": naive_distance_error.clone().detach(),
                    "naive_distance_error_X": navie_distance_error_X.clone().detach(),
                    "naive_distance_error_Y": navie_distance_error_Y.clone().detach(),
                    "naive_distance_error_Z": navie_distance_error_Z.clone().detach(),
                    # "loss_distance_cumsum": distance_cumsum_loss.clone().detach(),
                    "loss_acceleration": accelerationLoss.clone().detach(),
                    "loss_velocity": velocityLoss.clone().detach(),
                    "loss_position": positionLoss.clone().detach(),
                    "restoration_regularization": restoration_regularization.clone().detach(),
                    "loss_identity": loss_identity.clone().detach(),
                }
            )
            # return (hubber_loss + mse_loss + pearson_loss + simclr_loss), l1_out, l2_out
            # return hubber_loss, l1_out, l2_out

        # hubber_loss = hubber_loss / self.history[mode][0]["loss_hubber"]
        # mse_loss = mse_loss / self.history[mode][0]["loss_mse"]
        # pearson_loss = pearson_loss / self.history[mode][0]["loss_pearson"]
        # simclr_loss = simclr_loss / self.history[mode][0]["loss_simclr"]
        # naive_distance_error = (
        #     naive_distance_error / self.history[mode][0]["naive_distance_error"]
        # )
        # navie_distance_error_X = (
        #     navie_distance_error_X / self.history[mode][0]["naive_distance_error_X"]
        # )
        # navie_distance_error_Y = (
        #     navie_distance_error_Y / self.history[mode][0]["naive_distance_error_Y"]
        # )
        # navie_distance_error_Z = (
        #     navie_distance_error_Z / self.history[mode][0]["naive_distance_error_Z"]
        # )
        # # distance_cumsum_loss = (
        # #     distance_cumsum_loss / self.history[mode][0]["loss_distance_cumsum"]
        # # )
        # accelerationLoss = accelerationLoss / self.history[mode][0]["loss_acceleration"]
        # velocityLoss = velocityLoss / self.history[mode][0]["loss_velocity"]
        # positionLoss = positionLoss / self.history[mode][0]["loss_position"]

        l1_out = l1_out / self.history[mode][0]["loss_l1"]
        l2_out = l2_out / self.history[mode][0]["loss_l2"]
        aux_loss_ = aux_loss / self.history[mode][0]["loss_aux"]
        restoration_regularization = (
            restoration_regularization
            / self.history[mode][0]["restoration_regularization"]
        )

        # remove first item of history
        # del self.history[mode][0]

        losses = {
            f"loss_hubber/{mode}": hubber_loss,
            f"loss_mse/{mode}": mse_loss,
            f"loss_pearson/{mode}": pearson_loss,
            f"loss_simclr/{mode}": simclr_loss,
            f"loss_l1/{mode}": l1_out,
            f"loss_l2/{mode}": l2_out,
            f"loss_aux/{mode}": aux_loss_,
            f"naive_distance_error/{mode}": naive_distance_error,
            f"naive_distance_error_X/{mode}": navie_distance_error_X,
            f"naive_distance_error_Y/{mode}": navie_distance_error_Y,
            f"naive_distance_error_Z/{mode}": navie_distance_error_Z,
            # f"loss_distance_cumsum/{mode}": distance_cumsum_loss,
            f"loss_acceleration/{mode}": accelerationLoss,
            f"loss_velocity/{mode}": velocityLoss,
            f"loss_position/{mode}": positionLoss,
            f"restoration_regularization/{mode}": restoration_regularization,
            f"loss_identity/{mode}": loss_identity,
        }
        total_loss = 0
        for key, value in losses.items():
            total_loss += value * self.config[key.split("/")[0]]
        total_loss = total_loss / sum(self.config.values())

        losses[f"loss/{mode}"] = total_loss
        return losses


# class NaiveDistanceSquareError(nn.Module):
#     def __init__(self, rescale=1):
#         super(NaiveDistanceSquareError, self).__init__()
#         self.rescale = rescale

#     def forward(self, x, y):
#         loss = (x.sum(dim=-1) - y.sum(dim=-1)).square().sum(
#             dim=-1
#         ).sqrt() * self.rescale  # unit is cm
#         return loss.mean()


# class DistanceCumsumLoss(nn.Module):
#     def __init__(self, rescale=1):
#         super(DistanceCumsumLoss, self).__init__()
#         self.rescale = rescale
#         # self.hubber_loss = nn.HuberLoss(delta=0.0001)

#     def forward(self, x, y):
#         loss = (x.cumsum(dim=-1) - y.cumsum(dim=-1)).square().sum(
#             dim=-1
#         ).sqrt() * self.rescale
#         # loss = self.hubber_loss(x.cumsum(dim=-1), y.cumsum(dim=-1))
#         return loss.mean()
