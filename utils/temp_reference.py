class VAETrainerModulebyAction(VAESpectrum):
    def on_test_start(self):
        # Clear previous test data
        self.test_dict = {}

    def test_step(self, batch, batch_idx):
        (recon_loss, kl_loss, total_loss, mse), result_dict = self.regular_forward(
            batch, "test"
        )
        for key, value in result_dict.items():
            if key not in self.test_dict:
                self.test_dict[key] = []
            self.test_dict[key].extend(list(value.detach().cpu()))

        return total_loss

    def on_test_end(self):
        # Gather the data from all processes
        if len(self.test_dict) > 0 and "encoded" in self.test_dict:
            # Convert lists to tensors
            encoded_tensor = torch.stack(self.test_dict["encoded"])
            labels_tensor = torch.tensor(self.test_dict["label"])
            actions_tensor = torch.tensor(self.test_dict["action"])

            # Gather data from all processes
            if self.trainer.world_size > 1:
                # Use all_gather to collect data from all processes
                gathered_encoded = self.all_gather(encoded_tensor)
                gathered_labels = self.all_gather(labels_tensor)
                gathered_actions = self.all_gather(actions_tensor)

                # Flatten the gathered tensors (they come back as [world_size, batch_size, ...])
                if gathered_encoded.dim() > encoded_tensor.dim():
                    gathered_encoded = gathered_encoded.flatten(0, 1)
                    gathered_labels = gathered_labels.flatten(0, 1)
                    gathered_actions = gathered_actions.flatten(0, 1)

                # Update test_dict with gathered data (only on rank 0 to avoid duplication)
                if self.global_rank == 0:
                    self.test_dict["encoded"] = gathered_encoded.cpu().numpy()
                    self.test_dict["label"] = gathered_labels.cpu().numpy()
                    self.test_dict["action"] = gathered_actions.cpu().numpy()

            # Only visualize on rank 0 to avoid multiple processes creating the same plot
            if self.global_rank == 0:
                torch.save(self.test_dict, f"{self.logger.log_dir}/test_dict.pt")
                self.visualize_latent_space_by_action()

        return super().on_test_end()

    @rank_zero_only
    def visualize_latent_space_by_action(self):
        """Create interactive t-SNE, UMAP, and PCA visualizations of the latent space grouped by action."""
        # Prepare data
        latents = np.array(self.test_dict["encoded"])
        labels = np.array(self.test_dict["label"])
        actions = np.array(self.test_dict["action"])

        # Flatten latents if needed
        if latents.ndim > 2:
            latents = latents.reshape(latents.shape[0], -1)

        for dim in range(2, 4):
            for mode in ["PCA", "t-SNE", "UMAP"]:
                if dim == 3 and mode == "t-SNE":
                    continue
                # Run dimensionality reduction
                print(
                    f"Performing {mode} in {dim}D on {latents.shape[0]} samples with {latents.shape[1]} dimensions"
                )
                if mode == "t-SNE":
                    perplexity = min(30, latents.shape[0] - 1)
                    n_neighbors = max(3 * perplexity, min(30, latents.shape[0] - 1) * 3)
                    tsne = TSNE(
                        n_components=dim,
                        perplexity=perplexity,
                        n_neighbors=n_neighbors,
                    )
                    latents_reduced = tsne.fit_transform(latents)
                elif mode == "PCA":
                    pca = PCA(n_components=dim)
                    latents_reduced = pca.fit_transform(latents)
                elif mode == "UMAP":
                    umap_model = UMAP(n_components=dim, n_neighbors=15, min_dist=0.1)
                    latents_reduced = umap_model.fit_transform(latents)
                else:
                    raise ValueError("Mode should be 't-SNE', 'PCA' or 'UMAP'")

                # Assign colors based on actions
                unique_actions = np.unique(actions)
                color_scale = px.colors.qualitative.Dark24
                colors = {
                    action: color_scale[i % len(color_scale)]
                    for i, action in enumerate(unique_actions)
                }

                # Build Plotly traces
                fig = go.Figure()
                for action in unique_actions:
                    mask = actions == action
                    mean = latents_reduced[mask].mean(axis=0)
                    std = latents_reduced[mask].std(axis=0)
                    # Log the mean and std
                    for idxAxis, axis in enumerate(["x", "y", "z"][:dim]):
                        self.logger.log_metrics(
                            {
                                f"latents_action/{action}/{mode}_{axis}_mean": mean[
                                    idxAxis
                                ],
                                f"latents_action/{action}/{mode}_{axis}_std": std[
                                    idxAxis
                                ],
                            }
                        )

                    if dim == 2:
                        fig.add_trace(
                            go.Scatter(
                                x=latents_reduced[mask, 0],
                                y=latents_reduced[mask, 1],
                                mode="markers",
                                marker=dict(color=colors[action], size=4),
                                name=f"Action {action} mean: {mean} std: {std}",
                                hovertemplate="Index: %{pointIndex}<br>Action: %{customdata}",
                                customdata=actions[mask].reshape(-1, 1),
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=latents_reduced[mask, 0],
                                y=latents_reduced[mask, 1],
                                z=latents_reduced[mask, 2],
                                mode="markers",
                                marker=dict(color=colors[action], size=4),
                                name=f"Action {action} mean: {mean} std: {std}",
                                hovertemplate="Index: %{pointIndex}<br>Action: %{customdata}",
                                customdata=actions[mask].reshape(-1, 1),
                            )
                        )
                    print(
                        f"Action {action}: {np.sum(mask)} samples with center at {latents_reduced[mask].mean(axis=0)}"
                    )

                # Update layout and axes labels
                x_min, x_max = (
                    latents_reduced[:, 0].min() - 1,
                    latents_reduced[:, 0].max() + 1,
                )
                y_min, y_max = (
                    latents_reduced[:, 1].min() - 1,
                    latents_reduced[:, 1].max() + 1,
                )
                if dim == 2:
                    fig.update_layout(
                        title=f"{mode} Visualization of VAE Latent Space by Action in {dim}D",
                        xaxis=dict(
                            title=f"{mode} 1", range=[x_min, x_max], constrain="domain"
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
                else:
                    z_min, z_max = (
                        latents_reduced[:, 2].min() - 1,
                        latents_reduced[:, 2].max() + 1,
                    )
                    fig.update_layout(
                        title=f"{mode} Visualization of VAE Latent Space by Action in {dim}D",
                        scene=dict(
                            xaxis=dict(title=f"{mode} 1", range=[x_min, x_max]),
                            yaxis=dict(title=f"{mode} 2", range=[y_min, y_max]),
                            zaxis=dict(title=f"{mode} 3", range=[z_min, z_max]),
                            aspectmode="cube",  # or "data" for proportional scaling
                        ),
                        width=1000,
                        height=1000,
                        legend=dict(itemsizing="constant"),
                    )

                # Save as interactive HTML
                save_path = f"{self.logger.log_dir}/{mode.lower()}_{dim}D_latent_space_by_action.html"
                fig.write_html(save_path, include_plotlyjs="cdn")
                print(f"Interactive {mode} visualization saved to: {save_path}")

                # Optionally log HTML path to TensorBoard text
                if hasattr(self.logger, "experiment"):
                    self.logger.experiment.add_text(
                        f"{mode}_{dim}D/latent_space_by_action_html",
                        save_path,
                        self.current_epoch,
                    )
                    image_save_path = f"{self.logger.log_dir}/{mode.lower()}_{dim}D_latent_space_by_action.png"
                    fig.write_image(image_save_path)
                    self.logger.experiment.add_image(
                        f"{mode}_{dim}D/latent_space_by_action_image",
                        torch.tensor(np.array(Image.open(image_save_path))).permute(
                            2, 0, 1
                        ),  # Convert to CHW format
                        self.current_epoch,
                    )
