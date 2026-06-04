import colored as cl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from einops import rearrange
from ipywidgets import Button, HBox, VBox, interactive
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class Drawer:
    def __init__(
        self,
        model: pl.LightningModule,
        dm: pl.LightningDataModule,
        mode: str = "test",
        jump=100,
    ):
        self.model = model.cuda()
        self.dm = dm
        self.mode = mode
        self.jump = jump
        if self.mode == "test":
            self.dm.setup("test")
        else:
            self.dm.setup("fit")

        self.__t = 0.01
        self.line_G_visible = True
        self.line_E_visible = True
        self.line_R_visible = False

    def random_sample(self):
        if self.mode == "test":
            return np.random.randint(0, len(self.dm.test_dataset.files))
        elif self.mode == "val":
            return np.random.randint(0, len(self.dm.val_dataset.files))
        elif self.mode == "train":
            return np.random.randint(0, len(self.dm.train_dataset.files))
        else:
            raise ValueError("Invalid mode")

    def get_sequence(self, sequence: int):
        if self.mode == "test":
            return self.dm.test_dataset.get_sequence(idx=sequence)
        elif self.mode == "val":
            return self.dm.val_dataset.get_sequence(idx=sequence)
        elif self.mode == "train":
            return self.dm.train_dataset.get_sequence(idx=sequence)
        else:
            raise ValueError("Invalid mode")

    def switch_mode(self, mode):
        if mode == self.mode:
            return

        self.mode = mode
        if self.mode == "test":
            self.dm.setup("test")
        else:
            self.dm.setup("fit")

    def draw(self, sequence=None):
        # Function to plot the trajectory at a given timestamp
        print(
            "Drawing Trajectory" + cl.Fore.GREEN + f"Mode: {self.mode}" + cl.Style.reset
        )
        plt.close("all")
        if sequence is None:
            sequence = self.random_sample()

        dataset, name = self.get_sequence(sequence)
        self._file_name = name

        loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)
        estimation = []
        ground_truth = []
        raw_estimation = []
        for batch in loader:
            # batch = batch.to(self.model.device)
            x, y, dataL = batch

            x = x.to(self.model.device)
            y = y.to(self.model.device)
            dataL = dataL.to(self.model.device)
            # y_hat = self.model.seq_sampling((x, y, dataL))
            y_hat = self.model.naive_seq_sampling((x, y, dataL))
            batch_size, channels, length = x.shape
            mask = torch.arange(length).expand(batch_size, length).to(dataL.device)
            mask = mask < dataL.unsqueeze(1)
            mask = torch.logical_not(mask)
            y_hat = y_hat.masked_fill(mask.unsqueeze(1), 0)

            ground_truth.append(y[:, :3].detach().cpu())
            estimation.append(y_hat[:, :3].detach().cpu())
            raw_estimation.append(x[:, :3].detach().cpu())

        # convert [(batch,c,seq)] to (batch,c,seq)

        # cat by batch axis
        ground_truth = torch.cat(ground_truth, dim=0)
        estimation = torch.cat(estimation, dim=0)
        raw_estimation = torch.cat(raw_estimation, dim=0)

        empty = torch.zeros(ground_truth.shape[0], 3, 1)
        # estimation = torch.cat([empty, estimation], dim=2).diff(dim=2) /self.__t
        # ground_truth = torch.cat([empty, ground_truth], dim=2).diff(dim=2)/self.__t
        # raw_estimation = torch.cat([empty, raw_estimation], dim=2).diff(dim=2)/self.__t

        # e = estimation[:,:,-1].cumsum(dim=0).unsqueeze(-1)
        # g = ground_truth[:,:,-1].cumsum(dim=0).unsqueeze(-1)
        # r = raw_estimation[:,:,-1].cumsum(dim=0).unsqueeze(-1)

        # estimation[1:] += e[:-1]
        # ground_truth[1:] += g[:-1]
        # raw_estimation[1:] += r[:-1]

        # # consider the batch is time after sequence
        estimation = rearrange(estimation, "b c seq -> c (b seq)")
        ground_truth = rearrange(ground_truth, "b c seq -> c (b seq)")
        raw_estimation = rearrange(raw_estimation, "b c seq -> c (b seq)")

        # # filter
        # for k in range(3):
        #     estimation[k] = torch.functional.F.conv1d(
        #         estimation[k].unsqueeze(0),
        #         torch.ones(1, 1, 11).to(estimation.device) / 11,
        #         padding=5,
        #     ).squeeze(0)

        estimation = rearrange(estimation, "c t -> t c")
        ground_truth = rearrange(ground_truth, "c t -> t c")
        raw_estimation = rearrange(raw_estimation, "c t -> t c")

        # store the accerlation
        self._acc_estimation = estimation.clone()
        self._acc_ground_truth = ground_truth.clone()
        self._acc_raw_estimation = raw_estimation.clone()

        # acc -> pos

        self._estimation = torch.cumsum(estimation, dim=0) * self.__t
        self._ground_truth = (
            torch.cumsum(torch.cumsum(ground_truth, dim=0), dim=0) * self.__t**2
        )
        self._raw_estimation = (
            torch.cumsum(torch.cumsum(raw_estimation, dim=0), dim=0) * self.__t**2
        )
        # self._estimation = torch.cumsum(estimation, dim=0) * t
        # self._ground_truth = torch.cumsum(ground_truth, dim=0) * t
        # self._raw_estimation = torch.cumsum(raw_estimation, dim=0) * t

        # add zeros front
        self._estimation = torch.cat([torch.zeros(1, 3), self._estimation], dim=0)
        self._ground_truth = torch.cat([torch.zeros(1, 3), self._ground_truth], dim=0)
        self._raw_estimation = torch.cat(
            [torch.zeros(1, 3), self._raw_estimation], dim=0
        )

        # Create interactive slider
        interactive_plot = interactive(
            self.plot_trajectory, timestamp=(1, len(self._estimation) // self.jump)
        )
        output = interactive_plot.children[-1]
        output.layout.height = "600px"
        # Create toggle buttons
        btn_G = Button(description="Toggle Ground Truth")
        btn_E = Button(description="Toggle Estimation")
        btn_R = Button(description="Toggle Raw Estimation")
        # Attach toggle functions
        btn_G.on_click(self.toggle_G)
        btn_E.on_click(self.toggle_E)
        btn_R.on_click(self.toggle_R)
        button_box = HBox([btn_G, btn_E, btn_R])

        return VBox([button_box, interactive_plot])

    def reDraw(self):
        plt.close("all")
        interactive_plot = interactive(
            self.plot_trajectory, timestamp=(1, len(self._estimation) // self.jump)
        )
        output = interactive_plot.children[-1]
        output.layout.height = "600px"
        return interactive_plot

    def toggle_G(self, _):
        self.line_G_visible = not self.line_G_visible
        self.update_plot()

    def toggle_E(self, _):
        self.line_E_visible = not self.line_E_visible
        self.update_plot()

    def toggle_R(self, _):
        self.line_R_visible = not self.line_R_visible
        self.update_plot()

    def plot_trajectory(self, timestamp):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        timestamp = min(int(timestamp * self.jump), len(self._estimation) - 1)

        # Plot trajectories
        if self.line_G_visible:
            ax.plot(
                self._ground_truth[:timestamp, 0],
                self._ground_truth[:timestamp, 1],
                self._ground_truth[:timestamp, 2],
                label="Ground Truth",
                color="blue",
                alpha=0.5,
            )
            ax.scatter(
                self._ground_truth[timestamp - 1, 0],
                self._ground_truth[timestamp - 1, 1],
                self._ground_truth[timestamp - 1, 2],
                color="blue",
                alpha=0.5,
            )
        if self.line_E_visible:
            ax.plot(
                self._estimation[:timestamp, 0],
                self._estimation[:timestamp, 1],
                self._estimation[:timestamp, 2],
                label="Estimation",
                color="red",
                alpha=0.5,
            )
            ax.scatter(
                self._estimation[timestamp - 1, 0],
                self._estimation[timestamp - 1, 1],
                self._estimation[timestamp - 1, 2],
                color="red",
                alpha=0.5,
            )
        if self.line_R_visible:
            ax.plot(
                self._raw_estimation[:timestamp, 0],
                self._raw_estimation[:timestamp, 1],
                self._raw_estimation[:timestamp, 2],
                label="Raw Estimation",
                color="green",
                alpha=0.5,
            )
            ax.scatter(
                self._raw_estimation[timestamp - 1, 0],
                self._raw_estimation[timestamp - 1, 1],
                self._raw_estimation[timestamp - 1, 2],
                color="green",
                alpha=0.5,
            )

        X = (
            torch.min(
                torch.min(
                    self._ground_truth[:timestamp, 0].min(),
                    self._estimation[:timestamp, 0].min(),
                ),
                self._raw_estimation[:timestamp, 0].min(),
            ),
            torch.max(
                torch.max(
                    self._ground_truth[:timestamp, 0].max(),
                    self._estimation[:timestamp, 0].max(),
                ),
                self._raw_estimation[:timestamp, 0].max(),
            ),
        )
        Y = (
            torch.min(
                torch.min(
                    self._ground_truth[:timestamp, 1].min(),
                    self._estimation[:timestamp, 1].min(),
                ),
                self._raw_estimation[:timestamp, 1].min(),
            ),
            torch.max(
                torch.max(
                    self._ground_truth[:timestamp, 1].max(),
                    self._estimation[:timestamp, 1].max(),
                ),
                self._raw_estimation[:timestamp, 1].max(),
            ),
        )
        Z = (
            torch.min(
                torch.min(
                    self._ground_truth[:timestamp, 2].min(),
                    self._estimation[:timestamp, 2].min(),
                ),
                self._raw_estimation[:timestamp, 2].min(),
            ),
            torch.max(
                torch.max(
                    self._ground_truth[:timestamp, 2].max(),
                    self._estimation[:timestamp, 2].max(),
                ),
                self._raw_estimation[:timestamp, 2].max(),
            ),
        )
        for item in [X, Y, Z]:
            if item[0] == item[1]:
                item = (item[0] - 1, item[1] + 1)
        # ax.set_xlim(X)
        # ax.set_ylim(Y)
        # ax.set_zlim(Z)
        ax.set_xlim(self._ground_truth[:, 0].min(), self._ground_truth[:, 0].max())
        ax.set_ylim(self._ground_truth[:, 1].min(), self._ground_truth[:, 1].max())
        ax.set_zlim(self._ground_truth[:, 2].min(), self._ground_truth[:, 2].max())

        # Labels and title
        ax.set_title(f"Trajectories at Timestamp {timestamp}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_aspect("equal", "box")
        ax.legend()
        fig.tight_layout()
        # fig.show()

    def update_plot(self):
        self.plot_trajectory(len(self._estimation) // self.jump)
