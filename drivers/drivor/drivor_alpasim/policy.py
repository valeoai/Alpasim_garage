# SPDX-License-Identifier: Apache-2.0
# Adapted from the AlpaSim SimScale LTF sample (Copyright (c) 2026 NVIDIA Corporation).

"""DrivoR policy boundary: predict_batch(list[InferenceInput]) -> list[Prediction].

The network is the upstream DrivoR model (valeoai/DrivoR, copied into ./original), built
from its agent YAML with the overrides of its NAVSIM navtest evaluation, and loaded with
strict=True from the released checkpoint. Only the model is imported, not the NAVSIM
agent, so none of the NAVSIM/nuPlan training and scoring stack is needed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .preprocessing import CAMERA_IDS, build_ego_status, preprocess_images

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "agent._drivor_model."

# agent.config.* overrides of DrivoR's NAVSIM navtest evaluation.
AGENT_CONFIG = os.path.join(os.path.dirname(__file__), "original", "drivor_original_agent.yaml")
CONFIG_OVERRIDES = {
    "proposal_num": 64,
    "refiner_ls_values": 0.0,
    "image_backbone": {"focus_front_cam": False},
    "one_token_per_traj": True,
    "refiner_num_heads": 1,
    "tf_d_model": 256,
    "tf_d_ffn": 1024,
    "area_pred": False,
    "agent_pred": False,
    "ref_num": 4,
    "noc": 1.0,
    "dac": 1.0,
    "ddc": 0.0,
    "ttc": 5.0,
    "ep": 5.0,
    "comfort": 2.0,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class InferenceInput:
    images: Mapping[str, np.ndarray]
    command_one_hot: np.ndarray
    velocity_xy: np.ndarray
    acceleration_xy: np.ndarray

    def __post_init__(self) -> None:
        missing = [
            camera_id for camera_id in CAMERA_IDS if camera_id not in self.images
        ]
        if missing:
            raise ValueError(f"missing required cameras: {', '.join(missing)}")


@dataclass(frozen=True)
class Prediction:
    trajectory: np.ndarray


def build_config() -> DictConfig:
    config = OmegaConf.load(AGENT_CONFIG).config
    config = OmegaConf.merge(config, CONFIG_OVERRIDES)
    # No DINOv2 weights file needed: the checkpoint (strict=True) sets every tensor.
    config.image_backbone.pretrained = False
    return config


def normalized_state_dict(payload: object) -> dict[str, Tensor]:
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("checkpoint payload must be a dict containing 'state_dict'")

    normalized: dict[str, Tensor] = {}
    for key, value in payload["state_dict"].items():
        if not key.startswith(CHECKPOINT_PREFIX):
            raise ValueError(f"unexpected checkpoint key: {key}")
        normalized[key.removeprefix(CHECKPOINT_PREFIX)] = value
    return normalized


def load_checkpoint(model: nn.Module, path: str | PathLike[str]) -> int:
    # Lightning checkpoint (optimizer/loop state included), so not weights_only.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = normalized_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    return len(state_dict)


class DrivoRPolicy:
    def __init__(
        self,
        checkpoint_path: str | PathLike[str],
        device: str | torch.device = "cuda",
        warm_up: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but CUDA is not available"
            )

        from .original.model import DrivoRModel

        self.config = build_config()
        self.model = DrivoRModel(self.config)
        if self.model.num_cams != len(CAMERA_IDS):
            raise RuntimeError(
                f"DrivoR config enables {self.model.num_cams} cameras; "
                f"the adapter feeds {len(CAMERA_IDS)} ({', '.join(CAMERA_IDS)})"
            )
        self.state_dict_count = load_checkpoint(self.model, checkpoint_path)
        LOGGER.info(
            "Loaded %d tensors from checkpoint %s",
            self.state_dict_count,
            checkpoint_path,
        )
        self.model.to(self.device).eval()

        if warm_up:
            zero_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            request = InferenceInput(
                images={camera_id: zero_image for camera_id in CAMERA_IDS},
                command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
                velocity_xy=np.zeros(2, dtype=np.float32),
                acceleration_xy=np.zeros(2, dtype=np.float32),
            )
            self.predict_batch([request])
            LOGGER.info("DrivoR policy warm-up completed")

    def predict_batch(self, requests: list[InferenceInput]) -> list[Prediction]:
        if not requests:
            return []

        # (B, num_cams, 3, 672, 1148) uint8, then the upstream feature builder's
        # /255 + ImageNet normalization, in float32.
        image = torch.stack(
            [preprocess_images(request.images) for request in requests]
        ).to(device=self.device)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 1, 3, 1, 1)
        image = (image.float() / 255.0 - mean) / std

        # Its forward reads ego_status[:, -1] of a (B, T, 11) history.
        ego_status = torch.stack(
            [
                build_ego_status(
                    command_one_hot=request.command_one_hot,
                    velocity_xy=request.velocity_xy,
                    acceleration_xy=request.acceleration_xy,
                )
                for request in requests
            ]
        )[:, None].to(device=self.device)

        # FP32, as in the NAVSIM evaluation (AbstractAgent.compute_trajectory).
        with torch.inference_mode():
            output = self.model({"image": image, "ego_status": ego_status})
        return self._predictions(output, len(requests))

    @staticmethod
    def _predictions(output: Mapping, batch_size: int) -> list[Prediction]:
        trajectory = output["trajectory"].detach().float().cpu().numpy()
        expected_shape = (batch_size, 8, 3)
        if trajectory.shape != expected_shape:
            raise ValueError(
                f"model trajectory must have shape {expected_shape}; "
                f"got {trajectory.shape}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError("model trajectory contains non-finite values")

        return [Prediction(trajectory=sample.copy()) for sample in trajectory]
