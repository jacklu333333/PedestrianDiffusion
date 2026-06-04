from .common_imports import *


class CosSimMetric(nn.Module):
    def __init__(self, channel_index: list, norm=False, reduction="mean"):
        super(CosSimMetric, self).__init__()
        # self.loss = torch.vmap(nn.CosineSimilarity(dim=1))
        self.loss = torch.vmap(F.cosine_similarity)
        self.channel_index = channel_index
        self.norm = norm
        self.reduction = reduction

    def forward(self, x, y, steps=None):
        # assert x.dim() == 3, f"{x.shape}"
        # assert y.dim() == 3, f"{y.shape}"

        x = x[:, self.channel_index]
        y = y[:, self.channel_index]

        if self.norm:
            x = x.norm(dim=1, keepdim=True)
            y = y.norm(dim=1, keepdim=True)

        loss = self.loss(x, y)

        if steps is not None:
            sign = torch.sign(loss)
            loss = loss.clamp(min=-1, max=1)
            loss = torch.pow(loss.abs(), steps.reshape(loss.shape[0], 1)).mul(sign)

        if loss.numel() <= 1:
            print(cl.Fore.red)
            print(x.shape, y.shape)
            print(f"Warning: CosSimMetric returning single value {loss.item()}")
            print(cl.Style.reset)

        result = {
            "mean": loss.mean(),
            "std": loss.std(),
        }
        if self.reduction is not "mean":
            result["loss"] = loss
        return result


# class PearsonLoss(nn.Module):
#     def __init__(self):
#         super(PearsonLoss, self).__init__()
#         self.metric_x = PearsonMetric_acc_X()
#         self.metric_y = PearsonMetric_acc_Y()
#         self.metric_z = PearsonMetric_acc_Z()
#         self.metric_norm = PearsonMetric_acc_Norm()

#     def forward(self, x, y):
#         loss_x = 1 - self.metric_x(x, y)
#         loss_y = 1 - self.metric_y(x, y)
#         loss_z = 1 - self.metric_z(x, y)
#         # loss_norm = 1 - self.metric_norm(x, y)
#         loss = loss_x + loss_y + loss_z  # + loss_norm
#         return loss


# class simclr_loss(nn.Module):
#     def __init__(self):
#         super(simclr_loss, self).__init__()
#         self.metric_x = CosSimMetric_acc_X()
#         self.metric_y = CosSimMetric_acc_Y()
#         self.metric_z = CosSimMetric_acc_Z()
#         self.metric_norm = CosSimMetric_acc_Norm()

#     def forward(self, x, y):
#         loss_x = 1 - self.metric_x(x, y)
#         loss_y = 1 - self.metric_y(x, y)
#         loss_z = 1 - self.metric_z(x, y)
#         # loss_norm = 1 - self.metric_norm(x, y)
#         loss = loss_x + loss_y + loss_z  # + loss_norm
#         return loss
