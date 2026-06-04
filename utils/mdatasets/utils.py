import functools
import itertools
from typing import List, Union

from .common_imports import *

# from transformers import CLIPTextModel, CLIPTokenizer
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import os.path as osp
# import h5py
# import json


def peeking_trajectory(location):
    # plot 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(location[:, 0], location[:, 1], location[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # set tri-axis equal
    ax.set_box_aspect(
        [np.ptp(location[:, 0]), np.ptp(location[:, 1]), np.ptp(location[:, 2])]
    )
    plt.show()
    c = input("Continue? [y/n]")
    if c == "n":
        exit()
    plt.close("all")


# 1. DEFINE DEVICE
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        except ImportError:
            return torch.device("cpu")


DEVICE = get_device()
# print(f"Using device: {DEVICE}")

# model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_id = "openai/clip-vit-base-patch32"
TOKENIZER = CLIPTokenizer.from_pretrained(
    model_id,
    cache_dir="cache/clip_tokenizer",
    # local_files_only=True,
)
# Load full CLIPModel to avoid "unexpected keys" warnings for vision weights
_full_clip_model = CLIPModel.from_pretrained(
    model_id,
    # use_safetensors=True,
    cache_dir="cache/clip_text_encoder",
    # local_files_only=True,
)
TEXT_ENCODER = _full_clip_model.text_model


@functools.lru_cache(maxsize=2048)
def will_truncate(prompt: str) -> bool:
    # 1. Get raw tokens (no truncation/padding)
    raw = TOKENIZER(prompt, truncation=False, padding=False)
    raw_len = len(raw["input_ids"])
    # print(cl.Fore.green, prompt, "  --> ", raw_len, cl.Style.reset)

    # 2. Compare against max
    return raw_len > TOKENIZER.model_max_length


@functools.lru_cache(maxsize=1024)
def _process_encoder_core(prompts_tuple):
    # Helper to handle the actual encoding logic so it can be cached
    # Convert tuple back to list for tokenizer
    prompts = list(prompts_tuple)

    processed_prompts = []
    for p in prompts:
        # prompt = "datasets/Oxford Inertial Odometry Dataset/test/multi-attachments/trolley/vi1.csv"
        if will_truncate(p):
            raise ValueError(f"Prompt is too long for CLIP tokenizer: {p}")
        p = p.replace("_", " ")
        p = p.replace("-", " ")
        p = p.replace("/", " ")
        processed_prompts.append(p)

    tokens = TOKENIZER(
        processed_prompts,
        padding="max_length",
        truncation=True,
        max_length=TOKENIZER.model_max_length,
        return_tensors="pt",
    )
    # 3. MOVE INPUTS TO GPU & DISABLE GRADIENT CALCULATION
    # Ensure execution on correct device (handling DDP where DEVICE global might be stale or generic)
    active_device = get_device()

    # Move model to device if not already (cheap if already there)
    TEXT_ENCODER.to(active_device)

    tokens = {k: v.to(active_device) for k, v in tokens.items()}

    with torch.no_grad():
        text_outputs = TEXT_ENCODER(**tokens)

    text_embeds = text_outputs.last_hidden_state

    # Return to CPU to save VRAM since this is cached
    return text_embeds.cpu(), tokens["attention_mask"].cpu()
    # return text_embeds, tokens["attention_mask"].cpu()


def process_Encoder(prompt: Union[str, List[str]]):
    # Wrapper to handle types and utilize cache
    if isinstance(prompt, str):
        # Pass as single-item tuple to be hashable
        endcoding, mask = _process_encoder_core((prompt,))
        return endcoding.detach()[0, :, :].detach(), mask[0, :, :].detach()
    else:
        # Convert list to tuple to be hashable
        endcoding, mask = _process_encoder_core(tuple(prompt))
        return endcoding.detach(), mask.detach()


def collect_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    This function pads the sequences to the maximum length in the batch.
    """
    max_length = MAX_FRAME_LENGTH
    # xs, ys, ls = zip(*batch)
    xs = [_["dataX"] for _ in batch]
    ys = [_["dataY"] for _ in batch]
    ls = [_["dataL"] for _ in batch]
    encodings = [_["encoding"] for _ in batch]

    X = torch.stack(
        [F.pad(x, (0, 0, 0, 0, 0, max_length - x.size(0))) for x in xs], dim=0
    )
    Y = torch.stack(
        [F.pad(y, (0, 0, 0, 0, 0, max_length - y.size(0))) for y in ys], dim=0
    )
    L = torch.stack([F.pad(l, (0, max_length - l.size(0))) for l in ls], dim=0)

    encodings = torch.stack(encodings, dim=0)

    # return X, Y, L
    return {
        "dataX": X,
        "dataY": Y,
        "dataL": L,
        "encoding": encodings,
    }


def resample_series(*argv, original_sampling_rate, target_sampling_rate):
    """
    Resample multiple time series arrays from original to target sampling rate.

    Args:
        original_sampling_rate (float): Original sampling rate in Hz.
        target_sampling_rate (float): Target sampling rate in Hz.
        *argv: Variable number of 1D or 2D numpy arrays representing time series data.

    Returns:
        List of numpy arrays with resampled data.
    """
    resampled_arrays = []

    for series in argv:
        n_samples = series.shape[0]
        original_time = pd.timedelta_range(
            start="0s", periods=n_samples, freq=f"{1000/original_sampling_rate}ms"
        )

        # Handle 1D and 2D arrays
        if series.ndim == 1:
            df = pd.DataFrame({"value": series}, index=original_time)
        elif series.ndim == 2:
            col_names = [f"value_{i}" for i in range(series.shape[1])]
            df = pd.DataFrame(series, columns=col_names, index=original_time)
        else:
            raise ValueError("Only 1D or 2D arrays are supported.")

        # Resample and convert to numpy
        target_freq = f"{1000//target_sampling_rate}ms"
        df_resampled = df.resample(target_freq).mean()
        resampled_arrays.append(df_resampled.to_numpy())

    return resampled_arrays


# from the RoNIN Repo
# https://github.com/Sachini/ronin


def gyro_integration(ts, gyro, init_q):
    """
    Integrate gyro into orientation.
    https://www.lucidar.me/en/quaternions/quaternion-and-gyroscope/
    """
    output_q = np.zeros((gyro.shape[0], 4))
    output_q[0] = init_q
    dts = ts[1:] - ts[:-1]
    for i in range(1, gyro.shape[0]):
        output_q[i] = (
            output_q[i - 1]
            + angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1])
            * dts[i - 1]
        )
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q


@functools.lru_cache(
    maxsize=32
)  # Limit cache size to prevent OOM if many datasets exist
def select_orientation_source(
    data_path, max_ori_error=20.0, grv_only=True, use_ekf=True
):
    """
    Select orientation from one of gyro integration, game rotation vector or EKF orientation.

    Args:
        data_path: path to the compiled data. It should contain "data.hdf5" and "info.json".
        max_ori_error: maximum allow alignment error.
        grv_only: When set to True, only game rotation vector will be used.
                  When set to False:
                     * If game rotation vector's alignment error is smaller than "max_ori_error", use it.
                     * Otherwise, the orientation will be whichever gives lowest alignment error.
                  To force using the best of all sources, set "grv_only" to False and "max_ori_error" to -1.
                  To force using game rotation vector, set "max_ori_error" to any number greater than 360.


    Returns:
        source_name: a string. One of 'gyro_integration', 'game_rv' and 'ekf'.
        ori: the selected orientation.
        ori_error: the end-alignment error of selected orientation.
    """
    ori_names = ["gyro_integration", "game_rv"]
    ori_sources = [None, None, None]

    with open(osp.join(data_path, "info.json")) as f:
        info = json.load(f)
        ori_errors = np.array(
            [
                info["gyro_integration_error"],
                info["grv_ori_error"],
                info["ekf_ori_error"],
            ]
        )
        init_gyro_bias = np.array(info["imu_init_gyro_bias"])

    with h5py.File(osp.join(data_path, "data.hdf5")) as f:
        ori_sources[1] = np.copy(f["synced/game_rv"])
        if grv_only or ori_errors[1] < max_ori_error:
            min_id = 1
        else:
            if use_ekf:
                ori_names.append("ekf")
                ori_sources[2] = np.copy(f["pose/ekf_ori"])
            min_id = np.argmin(ori_errors[: len(ori_names)])
            # Only do gyro integration when necessary.
            if min_id == 0:
                ts = f["synced/time"]
                gyro = f["synced/gyro_uncalib"] - init_gyro_bias
                ori_sources[0] = gyro_integration(ts, gyro, ori_sources[1][0])

    return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]


def angular_velocity_to_quaternion_derivative(q, w):
    omega = (
        np.array(
            [
                [0, -w[0], -w[1], -w[2]],
                [w[0], 0, w[2], -w[1]],
                [w[1], -w[2], 0, w[0]],
                [w[2], w[1], -w[0], 0],
            ]
        )
        * 0.5
    )
    return np.dot(omega, q)


# 4. USE TORCH COMPILER DECORATOR
@torch.compile
def quaternion_to_angular_velocity(
    quaternions: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    Convert quaternion time series to angular velocity vectors.

    Args:
        quaternions: torch.Tensor of shape [T, 4], unit quaternions [x, y, z, w]
        dt: float, timestep between samples

    Returns:
        angular_velocities: torch.Tensor of shape [T, 3]
    """
    # Normalize to avoid drift
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)

    # Quaternion conjugate (inverse for unit quaternions)
    q_conj = quaternions.clone()
    q_conj[:, :3] *= -1  # negate vector part

    # Helper: quaternion multiplication
    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )

    # Compute quaternion derivatives with central differences
    dq = torch.zeros_like(quaternions)
    dq[1:-1] = (quaternions[2:] - quaternions[:-2]) / (2 * dt)
    dq[0] = (quaternions[1] - quaternions[0]) / dt  # forward diff
    dq[-1] = (quaternions[-1] - quaternions[-2]) / dt  # backward diff

    # Multiply dq * q_conj to get angular velocity quaternion
    omega_quat = 2.0 * quat_mul(dq, q_conj)

    # Extract vector part (x,y,z) as angular velocity
    angular_velocities = omega_quat[:, :3]
    return angular_velocities


@functools.lru_cache(maxsize=None)
def _get_encoding_cached(attributes_tuple, mode):
    attributes = dict(attributes_tuple)
    combination = {}
    for key, value in attributes.items():
        combination[key] = [value] if value == "unknown" else ["unknown", value]

    keys = list(combination.keys())
    values = list(combination.values())

    sentences = []
    if mode == "train":
        for instance in itertools.product(*values):
            current_attrs = dict(zip(keys, instance))
            # sentence = "This collected by {person} person doing {action} with {device} device mounted on {mounted} in {environment} environment in {dataset} datasets with {annotation} annotation.".format(
            #     person=current_attrs["person"],
            #     action=current_attrs["action"],
            #     device=current_attrs["device"],
            #     mounted=current_attrs["mounted"],
            #     environment=current_attrs["environment"],
            #     dataset=current_attrs["dataset"],
            #     annotation=current_attrs["annotation"],
            # )
            sentence = "This collected by {person} person doing {action} with {device} device mounted on {mounted} in {environment} environment with {annotation} annotation.".format(
                person=current_attrs["person"],
                action=current_attrs["action"],
                device=current_attrs["device"],
                mounted=current_attrs["mounted"],
                environment=current_attrs["environment"],
                annotation=current_attrs["annotation"],
            )
            sentences.append(sentence)
    else:
        # sentence = "This collected by {person} personnel doing {action} with {device} device mounted on {mounted} in {environment} environment with {annotation} annotation.".format(
        #     person="unknown",
        #     action="unknown",
        #     device="unknown",
        #     mounted="unknown",
        #     environment="unknown",
        #     dataset="unknown",
        #     annotation="unknown",
        # )
        sentence = "This collected by {person} personnel doing {action} with {device} device mounted on {mounted} in {environment} environment with {annotation} annotation.".format(
            person=attributes["person"],
            action=attributes["action"],
            device=attributes["device"],
            mounted=attributes["mounted"],
            environment=attributes["environment"],
            annotation=attributes["annotation"],
        )
    sentences.append(sentence)

    encodings_tensor, attention_mask = process_Encoder(sentences)
    # Convert back to list of tensors
    encodings = [encodings_tensor[i] for i in range(encodings_tensor.shape[0])]
    attention_masks = [attention_mask[i] for i in range(attention_mask.shape[0])]
    return encodings, attention_masks


def get_enconding(attributes: dict, mode: str):
    attr_tuple = tuple(sorted(attributes.items()))
    return _get_encoding_cached(attr_tuple, mode)


if __name__ == "__main__":
    encoding = process_Encoder("A person walking on the street")
    print(encoding.shape)
