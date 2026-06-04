import argparse
import json
import os
import socket
from argparse import Namespace
from pprint import pprint

import colored as cl
from flatten_dict import flatten, unflatten
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def parse_arguments():
    parser = argparse.ArgumentParser()
    default_config = json.load(open("config.json"))
    # Flatten the nested dict
    flat_config = flatten(default_config, reducer="dot")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file (JSON format)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="dataset name",
    )
    parser.add_argument(
        "--base_weight_path",
        "-w",
        type=str,
        default=None,
        help="base weight path to load the model",
    )
    parser.add_argument(
        "--vae_weight_path",
        "-vae_weight",
        type=str,
        default=None,
        help="VAE weight path to load the VAE model",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=None,
        help="batch size for training",
    )

    for key, value in flat_config.items():
        if key in [
            "vae_weight_path",
            "dataset",
            "base_weight_path",
            "model",
            "batch_size",
        ]:
            continue
        # print(f"{key}: {value}")
        if isinstance(value, bool):
            types = lambda x: (str(x).lower() == "true")
        elif value is not None:
            types = type(value)
        else:
            types = str
        parser.add_argument(
            f"--{key}",
            type=types,
            default=None,
            help=f"({type(value).__name__}) Default: {value}",
        )

    args = parser.parse_args()
    # find the match key in args and replace it
    print(f"Loading config from {args.config}")
    config = json.load(open(args.config))
    flat_config = flatten(config, reducer="dot")
    # if the matching key in args is not None, replace it
    args_dict = vars(args)
    args_dict.pop("config")
    for key, value in args_dict.items():
        if value is not None:
            if key in flat_config:
                rank_zero_info(
                    cl.Fore.yellow
                    + f"Overriding {key}: {flat_config[key]} --> {value}"
                    + cl.Style.reset
                )
                flat_config[key] = value
            else:
                raise KeyError(f"{key} not in config file, and this should not happen")
    config = unflatten(flat_config, splitter="dot")
    # if the model is cycleGan then clip are both None
    if config.get("model") == "cycleGan":
        config["gradient_clip_val"] = None
        config["gradient_clip_algorithm"] = None
    return config


if __name__ == "__main__":
    config = parse_arguments()
    pprint(config)
