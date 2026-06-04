from .common_imports import *
from .diffusionSpectrum import diffusionSpectrum
from .utils import *


class baseLineTestingPL(diffusionSpectrum):
    @torch.no_grad()
    def baseline_testing(self, batch):
        # epsilon, x, dataL, classes = batch
        # epsilon, x, dataL = batch
        epsilon = batch["dataX"]
        x = batch["dataY"]
        dataL = batch["dataL"]
        encoding = batch["encoding"]
        # epsilon, x = self.sample_noise((epsilon, x), 0, dataL=dataL, do_noise=False)
        # # estimate = epsilon[:, :3]
        # estimate = epsilon
        # estimate, x = self.postprocessing((estimate, x, dataL))
        estimate = epsilon.clone().detach()
        estimate, x = self.postprocessing((estimate, x, dataL))

        estimate[:, :3] = estimate[:, :3].cumsum(dim=-1) * (
            1 / self.config["sampling_rate"]
        )
        # x[:, :3] = x[:, :3].cumsum(dim=-1) * 0.01
        # mask = self.mask_generation(dataL, x.shape[-1], x.shape[0])
        # if torch.any(mask):
        #     estimate = estimate.masked_fill(
        #         mask.unsqueeze(1).expand(-1, estimate.shape[1], -1), 0
        #     )
        #     x = x.masked_fill(mask.unsqueeze(1).expand(-1, x.shape[1], -1), 0)
        masks = ~self.mask_generation(
            dataL=dataL,
            window_size=self.config["window_size"],
            batch_size=estimate.shape[0],
            channels=estimate.shape[1],
        )
        estimate = estimate.detach() * masks
        x = x.detach() * masks

        metrics = {}
        for key, metric in self.metrics["test"].items():
            if "naive" in key:
                # metrics[key] = metric(estimate, x, dataL=dataL)
                dictionary = metric(estimate, x, dataL=dataL)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                # metrics[key] = metric(estimate, x)
                dictionary = metric(estimate, x)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            # metrics[key] = metric(estimate[:, 3:6], x[:, 3:6])

        self.log_dict(
            metrics,
            sync_dist=True,
        )

        return estimate, x

    @overrides
    def on_test_start(self):
        print(
            cl.Fore.yellow
            + "- Warning this is the baseline evaluation approach"
            + cl.Style.reset
        )
        return super().on_test_start()

    @overrides
    def test_step(self, batch, batch_idx):
        dataX, dataY = self.baseline_testing(batch=batch)
        results = batch.copy()
        results["dataX"] = dataX
        results["dataY"] = dataY

        return {"loss": torch.nan, "results": results}

    @overrides
    def on_test_end(self):
        return super(pl.LightningModule, self).on_test_end()
