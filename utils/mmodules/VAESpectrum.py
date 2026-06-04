from .baseDiffusionModule import baseDiffusionModule
from .common_imports import *
from .utils import *


class VAESpectrum(baseDiffusionModule):
    def __init__(self, config, target_modes: str = "x"):
        _scaling = 5
        sensor_norm_config = {
            "acc_mean": 0,
            "acc_std": 5.78218412399292 * _scaling,
            "gyr_mean": 0,
            "gyr_std": 2.383342266082764 * _scaling,
        }
        label_norm_config = {
            "mean": 0,
            "std": 13.15231227874756 * _scaling,
            "ori_mean": 0,
            "ori_std": 2.802832841873169 * _scaling,
        }
        super().__init__(config=config)
        # self.VAE = mVAEModel(self.config["latent_dim"], eps=1e-14)
        self.VAE = VAE3D_1024(
            latent_dim=self.config["latent_dim"],
            in_channels=6,
        )
        # self.loss_function = VAELoss(
        #     latent_space_numel=self.VAE._latent_space_shape.numel(),
        #     kl_threshold=0.1,
        #     kl_beta=1.0,
        # )
        self.loss_function = ControlVAELoss(
            latent_space_numel=self.VAE._latent_space_shape.numel(),
            target_kl=0.1,
        )
        if hasattr(self, "toObservation"):
            del self.toObservation
        self.toObservation = nn.Sequential()
        if hasattr(self, "deObservation"):
            del self.deObservation
        self.deObservation = nn.Sequential()

        _scaling = 5.0
        if hasattr(self, "sensorProcessor"):
            del self.sensorProcessor
        self.sensorProcessor = nn.Sequential(batchNormalizeSensor(**sensor_norm_config))
        if hasattr(self, "finalSensorProcessor"):
            del self.finalSensorProcessor
        self.labelProcessor = nn.Sequential(
            # batchNormalizeQuaternion(),
            bathNormalizeRelativePosNOri(**label_norm_config),
        )
        if hasattr(self, "finalLabelProcessor"):
            del self.finalLabelProcessor
        self.finalLabelProcessor = nn.Sequential(
            # batchNormalizeQuaternion(),
            # batchDenormalizeRelativePosNOri(**label_norm_config),
            batchNormalizeSensor(**sensor_norm_config),
        )

        # self.loss = nn.MSELoss()

        self.target_mode = target_modes

        self.metrics = {
            "train": self._generate_metrics("train"),
            "val": self._generate_metrics("val"),
            "test": self._generate_metrics("test"),
        }

    @overrides
    def _generate_metrics(self, suffix):
        metrics = {
            f"metric_acc_simVector/{suffix}": CosSimMetric(
                channel_index=[0, 1, 2], norm=False
            ),
            f"metric_gyr_simVector/{suffix}": CosSimMetric(
                channel_index=[3, 4, 5], norm=False
            ),
        }
        return metrics

    def forward(self, x):
        x = self.toObservation(x)
        out = self.VAE(x)[0]
        out = self.deObservation(out)
        return out

    def forward_model(self, x):
        x = self.toObservation(x)
        x_hat, _, mu, logvar = self.VAE(x)
        # recon_loss, kl_loss = self.loss_compute(x_hat, x, mu, logvar)
        # loss_dict = self.loss_function(x_hat, x, mu, logvar)
        # return x_hat, x, mu, logvar
        # print(cl.Fore.yellow + f"x_hat shape: {x_hat.shape}" + cl.Style.reset)
        # print(cl.Fore.yellow + f"x shape: {x.shape}" + cl.Style.reset)
        return x_hat, x, mu, logvar

    def regular_forward(self, batch, mode):
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]
        label = batch["label"]

        x, y, _ = self.sample_noise((x, y), 0, dataL=L)

        if self.target_mode == "mix":
            raise NotImplementedError("Mix mode is not implemented yet")
            x_hat, x, x_mu, x_logvar = self.forward_model(x)
            y_hat, y, y_mu, y_logvar = self.forward_model(y)

            # print(cl.Fore.red, x_hat.shape, x.shape, L.shape, cl.Style.reset)

            # x_hat, x = self.postprocessing((x_hat, x, L))
            # y_hat, y = self.postprocessing((y_hat, y, L))
            mse = F.mse_loss(x_hat, x) + F.mse_loss(y_hat, y)

            loss_dict_x = self.loss_function(x_hat, x, x_mu, x_logvar)
            loss_dict_y = self.loss_function(y_hat, y, y_mu, y_logvar)
            loss_dict = {
                key: loss_dict_x[key] + loss_dict_y[key] for key in loss_dict_x.keys()
            }

        else:
            if self.target_mode == "y":
                raise NotImplementedError("Y mode is not implemented yet")
                x, y = y, x
            elif self.target_mode == "x":
                x, y = x, y
            else:
                raise ValueError("The target mode is not defined")
            x_hat, x, mu, logvar = self.forward_model(x)
            mse = F.mse_loss(x_hat, x)
            x_hat_t, x_t = self.postprocessing(
                (self.deObservation(x_hat), self.deObservation(x), L)
            )
            loss_dict = self.loss_function(
                recon_x_f=x_hat, x_f=x, recon_x_t=x_hat_t, x_t=x_t, mu=mu, logvar=logvar
            )

            # if mode == "test":
            if False:
                unique_label = torch.unique(label)
                for i in unique_label:
                    idx = label == i
                    # i_recon_loss, i_kl_loss = self.loss_compute(
                    #     x_hat[idx], x[idx], mu[idx], logvar[idx]
                    # )

                    i_loss_dict = self.loss_function(
                        recon_x_f=x_hat[idx],
                        x_f=x[idx],
                        recon_x_t=x_hat_t[idx],
                        x_t=x_t[idx],
                        mu=mu[idx],
                        logvar=logvar[idx],
                    )
                    i_recon_loss = (
                        i_loss_dict["recon_loss_acc"] + i_loss_dict["recon_loss_gyr"]
                    )
                    i_kl_loss = i_loss_dict["kl_loss"]
                    self.log_dict(
                        {
                            f"recon_loss_{INV_DATASET_DICT[i.item()]}/{mode}": i_recon_loss,
                            f"kl_loss_{INV_DATASET_DICT[i.item()]}/{mode}": i_kl_loss,
                        },
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    metrics = {}
                    for key, metric in self.metrics[mode].items():
                        dictionary = metric(x_hat_t[idx], x_t[idx])
                        for k, v in dictionary.items():
                            # insert {INV_DATASET_DICT[i.item()]} to key befor /
                            key_parts = key.split("/", 1)
                            key_parts[0] = (
                                f"{key_parts[0]}_{INV_DATASET_DICT[i.item()]}"
                            )
                            new_key = "/".join(key_parts) + f"_{k}"
                            metrics[new_key] = v

                    self.log_dict(
                        metrics,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
            # x_hat, x = self.postprocessing((x_hat, x, L))
        # psnr = self.PSNR(x_hat, x)
        # ssim = self.SSIM(x_hat, x)
        # self.log_dict(
        #     {
        #         f"psnr/{mode}": psnr,
        #         f"ssim/{mode}": ssim,
        #     },
        #     on_step=True if mode == "train" else False,
        #     on_epoch=False if mode == "train" else True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        self.log(
            f"mse/{mode}",
            mse,
            on_step=True if mode == "train" else False,
            on_epoch=False if mode == "train" else True,
            prog_bar=True,
            sync_dist=True,
        )
        losses = {}
        for key, value in loss_dict.items():
            losses[f"{key}/{mode}"] = value
        self.log_dict(
            losses,
            on_step=True if mode == "train" else False,
            on_epoch=False if mode == "train" else True,
            prog_bar=True,
            sync_dist=True,
        )
        result_dict = {}
        if mode == "test":
            encoded = self.VAE.encode(x)
            result_dict["encoded"] = encoded
            # result_dict["label"] = batch["label"]

        for key, value in batch.items():
            result_dict[key] = value

        return (
            loss_dict["recon_loss_acc"] + loss_dict["recon_loss_gyr"],
            loss_dict["kl_loss"],
            loss_dict["loss_total"],
            mse,
        ), result_dict

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "ema"):
            self.ema.step(self.VAE.parameters())
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        recon_loss, kl_loss, total_loss, mse = self.regular_forward(batch, "train")[0]
        return total_loss

    def validation_step(self, batch, batch_idx):
        recon_loss, kl_loss, total_loss, mse = self.regular_forward(batch, "val")[0]
        return total_loss

    def test_step(self, batch, batch_idx):
        (recon_loss, kl_loss, total_loss, mse), result_dict = self.regular_forward(
            batch, "test"
        )
        for key, value in result_dict.items():
            if key not in self.test_dict:
                self.test_dict[key] = []
            self.test_dict[key].extend(list(value.detach().cpu()))

        return total_loss

    def on_test_start(self):
        # Clear previous test data
        self.test_dict = {}

    @overrides
    def on_test_end(self):
        # Gather the data from all processes
        if len(self.test_dict) > 0 and "encoded" in self.test_dict:
            # Convert lists to tensors
            # tensors_to_gather = ["encoded", "label", "action"]
            tensors_to_gather = [
                "encoded",
                "label",
                "action",
                "device",
                "user",
                "mounted",
            ]
            gathered_tensors = {}

            for key in tensors_to_gather:
                gathered_tensors[key] = torch.stack(self.test_dict[key])

            # Gather data from all processes
            if self.trainer.world_size > 1:
                for key in tensors_to_gather:
                    gathered_tensor = self.all_gather(gathered_tensors[key])

                    # Flatten the gathered tensors (they come back as [world_size, batch_size, ...])
                    if gathered_tensor.dim() > gathered_tensors[key].dim():
                        gathered_tensor = gathered_tensor.flatten(0, 1)

                    # Update test_dict with gathered data (only on rank 0 to avoid duplication)
                    if self.global_rank == 0:
                        self.test_dict[key] = gathered_tensor.cpu().numpy()

            # Only visualize on rank 0 to avoid multiple processes creating the same plot
            if self.global_rank == 0:
                torch.save(self.test_dict, f"{self.logger.log_dir}/test_dict.pt")
                self.visualize_latent_space()

        return super().on_test_end()

    @rank_zero_only
    def visualize_latent_space(self):
        """Create interactive t-SNE visualization of the latent space with Plotly and save as HTML"""
        # Prepare data
        latents = np.stack(self.test_dict["encoded"])
        labels = np.stack(self.test_dict["label"])
        action = np.stack(self.test_dict["action"])
        device = np.stack(self.test_dict["device"])
        user = np.stack(self.test_dict["user"])
        mounted = np.stack(self.test_dict["mounted"])

        # exchange to the str meaning
        label_str = np.array([INV_DATASET_DICT[label] for label in labels])
        action_str = np.array(["" for _ in range(len(action))], dtype=object)
        device_str = np.array(["" for _ in range(len(device))], dtype=object)
        user_str = np.array(["" for _ in range(len(user))], dtype=object)
        mounted_str = np.array(["" for _ in range(len(mounted))], dtype=object)

        unique_label = np.unique(labels)
        for l in unique_label:
            label_mask = labels == l
            if len(unique_label) > 1:
                for i, query in enumerate(
                    self.trainer.datamodule.test_dataset.datasets
                ):
                    if query._label == l:
                        break
            else:
                query = self.trainer.datamodule.test_dataset
            action_str[label_mask] = np.array(
                [query.action_table[a.item()] for a in action[label_mask]], dtype=object
            )
            device_str[label_mask] = np.array(
                [query.device_table[d.item()] for d in device[label_mask]], dtype=object
            )
            user_str[label_mask] = np.array(
                [query.user_table[u.item()] for u in user[label_mask]], dtype=object
            )
            mounted_str[label_mask] = np.array(
                [query.mounted_table[m.item()] for m in mounted[label_mask]],
                dtype=object,
            )
        metadata = list(
            zip(
                label_str.tolist(),
                action_str.tolist(),
                device_str.tolist(),
                user_str.tolist(),
                mounted_str.tolist(),
            )
        )
        # Flatten latents if needed
        if latents.ndim > 2:
            latents = latents.reshape(latents.shape[0], -1)

        self.logger.experiment.add_embedding(
            latents,
            metadata=metadata,
            global_step=self.current_epoch,
            tag="LatentSpace",
        )

        # mode = "t-SNE"  # "PCA" or "t-SNE"
        for dim in range(2, 4):
            # for mode in ["PCA", "t-SNE", "UMAP"]:
            for mode in ["UMAP"]:
                if dim == 3 and mode == "t-SNE":
                    continue
                # Run t-SNE to 3D
                print(
                    f"- Performing {mode} in {dim}D on {latents.shape[0]} samples with {latents.shape[1]} dimensions"
                )
                if mode == "t-SNE":
                    # tsne = TSNE(
                    #     n_components=3,
                    #     random_state=42,
                    #     perplexity=min(30, latents.shape[0] - 1),
                    # )
                    # Ensure n_neighbors is at least 3 * perplexity
                    perplexity = min(30, latents.shape[0] - 1)
                    n_neighbors = (
                        max(3 * perplexity, min(30, latents.shape[0] - 1) * 3) * 2
                    )

                    tsne = TSNE(
                        n_components=dim,
                        perplexity=perplexity,
                        n_neighbors=n_neighbors,
                    )
                    latents_3d = tsne.fit_transform(latents)
                elif mode == "PCA":
                    # pca = PCA(n_components=3)
                    pca = PCA(n_components=dim)
                    latents_3d = pca.fit_transform(latents)
                elif mode == "UMAP":
                    umap_model = UMAP(n_components=dim, n_neighbors=15, min_dist=0.1)
                    latents_3d = umap_model.fit_transform(latents)
                else:
                    raise ValueError("Mode should be 't-SNE', 'PCA' or 'UMAP'")

                # Assign colors based on labels
                unique_labels = np.unique(labels)
                color_scale = px.colors.qualitative.Dark24
                colors = {
                    lbl: color_scale[i % len(color_scale)]
                    for i, lbl in enumerate(unique_labels)
                }

                # Build Plotly traces
                fig = go.Figure()
                for lbl in unique_labels:
                    mask = labels == lbl
                    mean = latents_3d[mask].mean(axis=0)
                    std = latents_3d[mask].std(axis=0)

                    # log the mean and std
                    for idxAxis, axis in enumerate(["x", "y", "z"][:dim]):
                        self.logger.log_metrics(
                            {
                                f"latents/{INV_DATASET_DICT[lbl]}/{mode}_{axis}_mean": mean[
                                    idxAxis
                                ],
                                f"latents/{INV_DATASET_DICT[lbl]}/{mode}_{axis}_std": std[
                                    idxAxis
                                ],
                            }
                        )

                    if dim == 2:
                        fig.add_trace(
                            go.Scatter(
                                x=latents_3d[mask, 0],
                                y=latents_3d[mask, 1],
                                mode="markers",
                                marker=dict(color=colors[lbl], size=4),
                                name=f"Class {INV_DATASET_DICT[lbl]}<br>mean: { np.char.mod('%.2f', mean)}<br>std: { np.char.mod('%.2f', std)}",
                                hovertemplate="Index: %{pointIndex}<br>Label: %{customdata}",
                                customdata=labels[mask].reshape(-1, 1),
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=latents_3d[mask, 0],
                                y=latents_3d[mask, 1],
                                z=latents_3d[mask, 2],
                                mode="markers",
                                marker=dict(color=colors[lbl], size=4),
                                name=f"Class {INV_DATASET_DICT[lbl]}<br>mean: { np.char.mod('%.2f', mean)}<br>std: { np.char.mod('%.2f', std)}",
                                hovertemplate="Index: %{pointIndex}<br>Label: %{customdata}",
                                customdata=labels[mask].reshape(-1, 1),
                            )
                        )
                    print(
                        f"- Class {INV_DATASET_DICT[lbl]}: {np.sum(mask)} samples with center at {latents_3d[mask].mean(axis=0)}"
                    )

                # Update layout and axes labels
                x_min, x_max = (
                    latents_3d[:, 0].min() - 1,
                    latents_3d[:, 0].max() + 1,
                )
                y_min, y_max = (
                    latents_3d[:, 1].min() - 1,
                    latents_3d[:, 1].max() + 1,
                )
                if dim == 2:
                    fig.update_layout(
                        title=f"{mode} Visualization of VAE Latent Space in {dim}D",
                        xaxis=dict(
                            title=f"{mode} 1",
                            range=[x_min, x_max],
                            constrain="domain",
                        ),
                        yaxis=dict(
                            title=f"{mode} 2",
                            range=[y_min, y_max],
                            constrain="domain",
                            scaleanchor="x",  # This makes the aspect ratio equal
                            scaleratio=1,  # 1:1 aspect ratio (square)
                        ),
                        width=1000,
                        height=1000,
                        legend=dict(itemsizing="constant"),
                    )
                    # set the x y bound according to the max and min value +-1
                else:
                    z_min, z_max = (
                        latents_3d[:, 2].min() - 1,
                        latents_3d[:, 2].max() + 1,
                    )
                    fig.update_layout(
                        title=f"{mode} Visualization of VAE Latent Space in {dim}D",
                        scene=dict(
                            xaxis=dict(title=f"{mode} 1", range=[x_min, x_max]),
                            yaxis=dict(title=f"{mode} 2", range=[y_min, y_max]),
                            zaxis=dict(title=f"{mode} 3", range=[z_min, z_max]),
                            aspectmode="manual",  # or "data" for proportional scaling
                        ),
                        width=1000,
                        height=1000,
                        legend=dict(itemsizing="constant"),
                    )

                # Save as interactive HTML
                save_path = (
                    f"{self.logger.log_dir}/{mode.lower()}_{dim}D_latent_space.html"
                )
                fig.write_html(save_path, include_plotlyjs="cdn")
                print(f"- Interactive {mode} visualization saved to: {save_path}")

                # Optionally log HTML path to TensorBoard text
                if hasattr(self.logger, "experiment"):
                    self.logger.experiment.add_text(
                        f"{mode}_{dim}D/latent_space_html",
                        save_path,
                        self.current_epoch,
                    )
                    image_save_path = (
                        f"{self.logger.log_dir}/{mode.lower()}_{dim}D_latent_space.png"
                    )
                    fig.write_image(image_save_path)
                    self.logger.experiment.add_image(
                        f"{mode}_{dim}D/latent_space_image",
                        torch.tensor(np.array(Image.open(image_save_path))).permute(
                            2, 0, 1
                        ),  # Convert to CHW format
                        self.current_epoch,
                    )

    # def configure_optimizers(self):
    #     # self.lr = 0.1
    #     # optimizer = optim.Adam(self.VAE.parameters(), lr=self.lr)
    #     optimizer = optim.AdamW(self.VAE.parameters(), lr=self.lr, fused=True)
    #     # optimizer = optim.AdamW(self.VAE.parameters(), lr=0.1)
    #     # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     #     optimizer, T_0=50, T_mult=2
    #     # )
    #     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     #     optimizer,
    #     #     mode="min",
    #     #     factor=0.9,
    #     #     patience=1000,
    #     #     cooldown=10,
    #     #     verbose=False,
    #     #     min_lr=5e-7,
    #     # )
    #     accumulation = self.trainer.accumulate_grad_batches if self.trainer else 1
    #     num_batches = len(self.trainer.datamodule.train_dataloader())
    #     # valid_batches = num_batches // num_devices // accumulation
    #     valid_batches = num_batches // self.trainer.world_size

    #     self.ema = diffusers.training_utils.EMAModel(
    #         parameters=self.VAE.parameters(),
    #         decay=1 - 1 / (valid_batches),
    #         update_after_step=valid_batches,
    #         use_ema_warmup=True,
    #         inv_gamma=valid_batches,
    #         foreach=True,
    #     )
    #     self.ema.to(self.device)

    #     scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
    #         optimizer=optimizer,
    #         num_warmup_steps=valid_batches * self.config["warm_up"],
    #         num_training_steps=valid_batches,
    #         # num_warmup_steps=self.warm_up,
    #         # num_training_steps=num_batches,
    #         # num_cycles=0.3,
    #     )

    #     # scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=250)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "loss/train",
    #             "interval": "step",
    #             "frequency": 1,
    #         },
    #     }
    def configure_optimizers(self):
        # both vae parameters and log_vars are optimized
        # params = list(self.VAE.parameters()) + [self.log_vars]
        params = list(self.VAE.parameters()) + list(self.loss_function.parameters())
        optimizer = optim.AdamW(params, lr=self.lr, fused=True)

        # Phase 1: Initial training with Cosine Warmup and EMA
        if self.config.get("train_phase", "initial") == "initial":
            accumulation = self.trainer.accumulate_grad_batches if self.trainer else 1
            num_batches = len(self.trainer.datamodule.train_dataloader())
            valid_batches = num_batches // self.trainer.world_size // accumulation

            self.ema = diffusers.training_utils.EMAModel(
                parameters=self.VAE.parameters(),
                decay=1 - 1 / (valid_batches),
                update_after_step=valid_batches,
                use_ema_warmup=True,
                inv_gamma=valid_batches,
                foreach=True,
            )
            self.ema.to(self.device)

            scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=valid_batches * self.config["warm_up"],
                num_training_steps=(
                    self.trainer.max_steps
                    if self.trainer.max_steps > 0
                    else self.trainer.max_epochs * valid_batches
                ),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss/train",
                    "interval": "step",
                    "frequency": accumulation,
                },
            }

        # Phase 2: Fine-tuning with ReduceLROnPlateau and no EMA
        elif self.config.get("train_phase") == "finetune":
            # Disable EMA for fine-tuning
            if hasattr(self, "ema"):
                del self.ema

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=100,
                cooldown=10,
                min_lr=1e-8,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss_total/train",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(
                f"Invalid train_phase: {self.config.get('train_phase')}. Must be 'initial' or 'finetune'."
            )
