from .common_imports import *


class PearsonMetric(nn.Module):
    def __init__(self, channel_index: list, norm, esp=1e-7):
        super(PearsonMetric, self).__init__()
        self.esp = esp
        self.channel_index = channel_index
        self.norm = norm

    def forward(self, x, y, steps=None):
        batch_size, channels, length = x.shape

        x_temp = x[:, self.channel_index].sum(dim=-1)
        y_temp = y[:, self.channel_index].sum(dim=-1)

        if self.norm:
            x_temp = x_temp.norm(dim=1, keepdim=True)
            y_temp = y_temp.norm(dim=1, keepdim=True)

        x_temp = rearrange(x_temp, "b c -> c b")
        y_temp = rearrange(y_temp, "b c -> c b")

        vx = x_temp - x_temp.mean(dim=-1, keepdim=True)  # (batch, c, l) --> (batch, c)
        vy = y_temp - y_temp.mean(dim=-1, keepdim=True)  # (batch, c, l) --> (batch, c)

        if batch_size != 1:
            if (vx == 0).all():
                print(cl.Fore.yellow + "- all the vx are zeros" + cl.Style.reset)
            if (vy == 0).all():
                print(cl.Fore.yellow + "- all the vy are zeros" + cl.Style.reset)
            if (vx == 0).all() or (vy == 0).all():
                print(cl.Fore.yellow + "-" * 200 + cl.Style.reset)
        if not torch.isfinite(x).all():
            print(cl.Fore.yellow + "- x is not finite" + cl.Style.reset)
        if not torch.isfinite(y).all():
            print(cl.Fore.yellow + "- y is not finite" + cl.Style.reset)

        index = vx == 0
        vx[index] = vx[index] + self.esp

        index = vy == 0
        vy[index] = vy[index] + self.esp

        vx_root_square_sum = torch.sum(vx**2, dim=-1, keepdim=True).sqrt()
        vy_root_square_sum = torch.sum(vy**2, dim=-1, keepdim=True).sqrt()

        assert (vx_root_square_sum != 0).all(), f"{vx}"
        assert (vy_root_square_sum != 0).all(), f"{vy}"

        assert torch.isfinite(vx_root_square_sum).all(), f"{vx}"
        assert torch.isfinite(vy_root_square_sum).all(), f"{vy}"

        loss = torch.sum(vx * vy, dim=-1, keepdim=True) / (
            vx_root_square_sum * vy_root_square_sum
        )

        assert torch.isfinite(loss).all()
        if steps is not None:
            sign = torch.sign(loss)
            loss = loss.clamp(min=-1, max=1)
            loss = torch.pow(loss.abs(), steps).mul(sign)

        return {
            "mean": loss.mean(),
            # "std": loss.std(),
        }
