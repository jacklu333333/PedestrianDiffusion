from .common_imports import *
from .DiffusionModelPL import DiffusionModelPL
from .utils import *


class baseTrainingModel(DiffusionModelPL):
    def special_forward(self, batch, mode):
        # x, y = batch
        epsilon, x, dataL = batch
        batch_size = x.shape[0]
        epsilon_pad, x_pad = self.sample_noise(
            (epsilon, x),
            0,
            dataL=dataL,
            do_noise=True if mode == "train" else False,
        )

        t = torch.randint(
            0,
            self.config["num_time_steps"],
            (batch_size,),
            requires_grad=False,
            device=x.device,
        )

        noise_input = self.scheduler.sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t,
        )
        # noise_input = epsilon_pad.clone().detach()
        # noise_input[:, :3] = self.scheduler.add_noise(
        #     original_samples=x_pad[:, :3],
        #     noise=epsilon_pad[:, :3],
        #     timesteps=t,
        # )

        assert torch.isfinite(noise_input).all()
        if torch.all(noise_input == 0):
            print(cl.Fore.red + "- All the noise are zeros" + cl.Style.reset)

        # output, aux_loss = self.model(noise_input, t, dataL)
        output = self.model(
            sample=noise_input[:, :3],
            timestep=t,
            return_dict=False,
        )[0]
        mask = self.mask_generation(dataL, x.shape[-1], batch_size)
        if torch.any(mask):
            output[mask.unsqueeze(1).expand(-1, output.shape[1], -1)] = 0
        aux_loss = F.l1_loss(
            torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
        )

        restoration_t_minus_1 = self.scheduler.sample_backward(
            pseudo_observation=noise_input[:, :3],
            noise_estimated=output,
            t=t,
            # z=noise_input[:, :3],
            z=self.sample_noise((epsilon, x), pad_size, dataL)[0],
        )
        # restoration_t_minus_1 = batchStepBatch(
        #     scheduler=self.scheduler,
        #     original=noise_input[:, :3],
        #     noise=output[:, :3],
        #     t=t,
        # )
        # l1_loss = torch.norm(self.model._get_regularization_params(), 1)
        # l2_loss = torch.norm(self.model._get_regularization_params(), 2)
        l1_loss, l2_loss = self._get_regularization()

        t_minus_1 = self.scheduler.sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t,
        )[:, :3]
        # t_minus_1 = epsilon_pad.clone().detach()
        # t_minus_1[:, :3] = self.scheduler.add_noise(
        #     original_samples=x_pad[:, :3],
        #     noise=epsilon_pad[:, :3],
        #     timesteps=t,
        # )
        losses = self.loss(
            # x=output,
            # y=x_pad,
            #
            # x=output,
            # y=restoration_t_minus_1,
            #
            # x=restoration_t_minus_1,
            # y=x_pad,
            #
            x=output,
            y=epsilon_pad[:, :3],  # the original loss definition
            #
            # x=restoration_t_minus_1,
            # y=t_minus_1,
            #
            # x=output,
            # y=t_minus_1,  # this meas estimate the result not the noise
            #
            restoration=restoration_t_minus_1,
            l1=l1_loss,
            l2=l2_loss,
            pad_size=pad_size,
            aux_loss=aux_loss,
            mode=mode,
        )
        noise_mse = F.mse_loss(output, epsilon_pad[:, :3])

        restoration_t_minus_1, x_pad = self.postprocessing(
            (restoration_t_minus_1, x_pad, dataL)
        )
        restoration_t_minus_1 = restoration_t_minus_1[
            :, :3, pad_size : pad_size + self.config["window_size"]
        ]
        x_pad = x_pad[:, :3, pad_size : pad_size + self.config["window_size"]]

        metrics = {}
        if mode in self.metrics.keys():
            for key, metric in self.metrics[mode].items():
                # metrics[key] = metric(output, x_pad)
                metrics[key] = metric(restoration_t_minus_1, x_pad)
                # metrics[key] = metric(restoration_t_minus_1, t_minus_1)
                # metrics[key] = metric(output, t_minus_1)

        self.log(
            f"noise_mse/{mode}",
            noise_mse,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            losses,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            metrics,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
        )

        return restoration_t_minus_1, output, epsilon_pad, losses[f"loss/{mode}"]
        # return restoration_t_minus_1, output, epsilon_pad, loss
