from .common_imports import *


class copyFileNLoadWeightLogger(Callback):
    def __init__(self, main_file_name):
        self.main_file_name = main_file_name
        self.log_dir: Optional[str] = None

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        self.logger = trainer.logger
        self.config = pl_module.config

        # resolve a single, consistent log_dir across ranks
        resolved = getattr(trainer, "log_dir", None)
        if resolved is None and hasattr(self.logger, "log_dir"):
            resolved = self.logger.log_dir
        assert resolved is not None, "Could not resolve log_dir from trainer/logger"
        self.log_dir = str(resolved)

        # copy files only on rank 0, others wait
        if trainer.global_rank == 0:
            utils_dir = os.path.join(self.log_dir, "utils")
            if not os.path.exists(utils_dir):
                self.log_files(trainer, pl_module)
            else:
                rank_zero_info(
                    cl.Fore.yellow
                    + f'- "utils" folder already exists, skip copying files'
                    + cl.Style.reset
                )
        # make sure all ranks see the same files before moving on
        if hasattr(trainer, "strategy") and trainer.strategy is not None:
            trainer.strategy.barrier()

        return super().setup(trainer, pl_module, stage)

    def on_fit_start(self, trainer, pl_module):
        if hasattr(self, "already_loaded") and self.already_loaded:
            return super().on_fit_start(trainer, pl_module)

        self.already_loaded = True
        self.loadWeight(trainer, pl_module)
        if hasattr(trainer, "strategy") and trainer.strategy is not None:
            trainer.strategy.barrier()
        return super().on_fit_start(trainer, pl_module)

    def on_test_start(self, trainer, pl_module):
        if hasattr(self, "already_loaded") and self.already_loaded:
            return super().on_test_start(trainer, pl_module)

        if not hasattr(trainer.datamodule, "train_dataset"):
            self.loadWeight(trainer, pl_module)
            if hasattr(trainer, "strategy") and trainer.strategy is not None:
                trainer.strategy.barrier()
        return super().on_test_start(trainer, pl_module)

    def _load_checkpoint(self, path):
        if os.path.isdir(path):
            state_dict = (
                deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                    path
                )
            )
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        return state_dict

    def loadWeight(self, trainer, pl_module):
        # Prefer copied paths under self.log_dir if present; otherwise fall back to original config paths.
        # base model weight
        if self.config["base_weight_path"] != "":
            orig = self.config["base_weight_path"]
            copied = None
            if self.log_dir is not None:
                copied = os.path.join(
                    self.log_dir, "base_model_weight", os.path.basename(orig)
                )
            path = copied if (copied and os.path.exists(copied)) else orig
            assert os.path.exists(path), f"Base model weight path {path} does not exist"

            state_dict = self._load_checkpoint(path)

            rank_zero_info(
                cl.Fore.yellow
                + f'- Using "{pl_module.__class__.__name__}" model and Load model weight from "{self.config["base_weight_path"]}"'
                + cl.Style.reset
            )
            # pl_module.load_state_dict(state_dict, strict=False)
            missing_key, unexpected_key = pl_module.load_state_dict(
                state_dict, strict=False
            )
            if len(unexpected_key) > 0 or len(missing_key) > 0:
                rank_zero_info(
                    cl.Fore.yellow
                    + f"- When loading base model weights, unexpected keys: {unexpected_key}, missing keys: {missing_key}"
                    + cl.Style.reset
                )

        # VAE weight
        if self.config["vae_weight_path"] != "":
            orig = self.config["vae_weight_path"]
            copied = None
            if self.log_dir is not None:
                copied = os.path.join(
                    self.log_dir, "vae_weight", os.path.basename(orig)
                )
            path = copied if (copied and os.path.exists(copied)) else orig
            assert os.path.exists(path), f"VAE weight path {path} does not exist"

            state_dict = self._load_checkpoint(path)

            rank_zero_info(
                cl.Fore.yellow
                + f'- Using "{pl_module.__class__.__name__}" model and Load VAE weight from "{self.config["vae_weight_path"]}"'
                + cl.Style.reset
            )
            # remove the prefix VAE. from the state_dict keys
            if hasattr(pl_module, "model"):
                state_dict = {
                    k[len("VAE.") :] if k.startswith("VAE.") else k: v
                    for k, v in state_dict.items()
                }

                unexpected_key, missing_key = pl_module.VAE.load_state_dict(
                    state_dict, strict=False
                )

            else:
                unexpected_key, missing_key = pl_module.load_state_dict(
                    state_dict, strict=True
                )

            if len(unexpected_key) > 0 or len(missing_key) > 0:
                rank_zero_info(
                    cl.Fore.yellow
                    + f"- When loading VAE weights, unexpected keys: {unexpected_key}, missing keys: {missing_key}"
                    + cl.Style.reset
                )

        elif hasattr(pl_module, "VAE"):
            if self.config["base_weight_path"] != "":
                rank_zero_info(
                    cl.Fore.yellow
                    + f'- Using "{pl_module.__class__.__name__}" model without loading VAE weights, but you may ignore this if the base model already includes VAE weights.'
                    + cl.Style.reset
                )
                # raise Warning(
                #     f'must have VAE weight path for model name: "{pl_module.__class__.__name__}"'
                # )
            else:
                rank_zero_info(
                    cl.Fore.red
                    + f'- No VAE weight path provided for model name: "{pl_module.__class__.__name__}"'
                    + cl.Style.reset
                )
        else:
            pass

        # latent VAE weight
        if self.config["latent_vae_weight_path"] != "":
            orig = self.config["latent_vae_weight_path"]
            copied = None
            if self.log_dir is not None:
                copied = os.path.join(
                    self.log_dir, "latent_vae_weight", os.path.basename(orig)
                )
            path = copied if (copied and os.path.exists(copied)) else orig
            assert os.path.exists(path), f"latent VAE weight path {path} does not exist"

            state_dict = self._load_checkpoint(path)

            rank_zero_info(
                cl.Fore.yellow
                + f'- Using "{pl_module.__class__.__name__}" model and Load latent VAE weight from "{self.config["latent_vae_weight_path"]}"'
                + cl.Style.reset
            )
            # remove the prefix VAE. from the state_dict keys
            state_dict = {
                k[len("VAE.") :] if k.startswith("VAE.") else k: v
                for k, v in state_dict.items()
            }

            pl_module.latent_VAE.load_state_dict(state_dict, strict=True)
        elif hasattr(pl_module, "latent_VAE"):
            if self.config["base_weight_path"] != "":
                rank_zero_info(
                    cl.Fore.yellow
                    + f'- Using "{pl_module.__class__.__name__}" model without loading latent VAE weights, but you may ignore this if the base model already includes VAE weights.'
                    + cl.Style.reset
                )
            else:
                rank_zero_info(
                    cl.Fore.red
                    + f'- No latent VAE weight path provided for model name: "{pl_module.__class__.__name__}"'
                    + cl.Style.reset
                )
        else:
            pass

    def _copy_weight(self, key, folder, readonly_files, desc):
        if self.config[key] != "":
            if not os.path.exists(self.config[key]):
                raise FileNotFoundError(
                    f"{desc} path {self.config[key]} does not exist"
                )
            os.makedirs(f"{self.log_dir}/{folder}", exist_ok=False)

            rank_zero_info(
                cl.Fore.yellow
                + f'Copy  {desc} to "{self.log_dir}/{folder}"'
                + cl.Style.reset
            )

            src = self.config[key]
            dst = f"{self.log_dir}/{folder}/{os.path.basename(src.rstrip(os.sep))}"
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

            for root, _, files in os.walk(f"{self.log_dir}/{folder}"):
                for f in files:
                    file_path = os.path.join(root, f)
                    readonly_files.append(file_path)

    @rank_zero_only
    def log_files(self, trainer, pl_module):
        # must have self.log_dir resolved in setup
        assert self.log_dir is not None, "log_dir must be set in setup()"

        readonly_files = [
            f"{self.log_dir}/config.json",
            os.path.join(self.log_dir, self.main_file_name),
        ]

        # # # base model weight
        # self._copy_weight(
        #     "base_weight_path", "base_model_weight", readonly_files, "Base model weight"
        # )

        # # # VAE weight
        # self._copy_weight("vae_weight_path", "vae_weight", readonly_files, "VAE weight")

        # main file and utils
        shutil.copy2(self.main_file_name, self.log_dir)
        shutil.copytree("utils", f"{self.log_dir}/utils")
        json.dump(self.config, open(f"{self.log_dir}/config.json", "w"), indent=4)

        # make copied files read-only
        for root, _, files in os.walk(f"{self.log_dir}/utils/"):
            for f in files:
                file_path = os.path.join(root, f)
                readonly_files.append(file_path)

        for file_path in readonly_files:
            if os.path.exists(file_path):
                os.chmod(file_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
