# SPDX-License-Identifier: Apache-2.0

"""Inputs for DrivoR, reproducing DrivoR's NAVSIM feature builder at inference time.

Camera order, resize and ego-status layout follow valeoai/DrivoR's feature builder
with its 4-camera agent config (cam_f0, cam_b0, cam_l0, cam_r0 at the current frame).
Images stay uint8 here; policy.py applies the same ImageNet normalization on-device.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

# Order of the feature builder's cameras (f0, b0, l0, l1, l2, r0, r1, r2),
# filtered to the four the agent config enables.
CAMERA_IDS = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0")
EXPECTED_HEIGHT = 1080
EXPECTED_WIDTH = 1920
IMAGE_SIZE = (1148, 672)  # config.image_size, (width, height) as passed to PIL


def _validated_image(camera_id: str, image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    expected_shape = (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"{camera_id} image must be 1920x1080 with shape {expected_shape}; "
            f"got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(
            f"{camera_id} image must have dtype uint8; got dtype {array.dtype}"
        )
    return array


def preprocess_images(images: Mapping[str, np.ndarray]) -> torch.Tensor:
    """Return uint8 images of shape (num_cams, 3, 672, 1148), resized with PIL."""
    missing = [camera_id for camera_id in CAMERA_IDS if camera_id not in images]
    if missing:
        raise ValueError(f"missing required cameras: {', '.join(missing)}")

    resized = [
        np.asarray(Image.fromarray(_validated_image(camera_id, images[camera_id])).resize(IMAGE_SIZE))
        for camera_id in CAMERA_IDS
    ]
    return torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).contiguous()


def _vector(name: str, values: Sequence[float] | np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},); got shape {vector.shape}")
    return vector


def build_ego_status(
    *,
    command_one_hot: Sequence[float] | np.ndarray,
    velocity_xy: Sequence[float] | np.ndarray,
    acceleration_xy: Sequence[float] | np.ndarray,
) -> torch.Tensor:
    """NAVSIM ego_status row: pose(3), velocity(2), acceleration(2), command(4).

    The current pose in its own ego frame is the origin, so it is all zeros.
    """
    status = np.concatenate(
        (
            np.zeros(3, dtype=np.float32),
            _vector("velocity_xy", velocity_xy, 2),
            _vector("acceleration_xy", acceleration_xy, 2),
            _vector("command_one_hot", command_one_hot, 4),
        )
    )
    return torch.from_numpy(status)
