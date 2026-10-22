# SPDX-License-Identifier: Apache-2.0

"""nuPlan camera views synthesized from the PAI (NuRec) camera rig.

NAVSIM models are trained on nuPlan cameras: 1920x1080 pinholes (f = 1545 px,
about 64 deg wide); CAM_F0/B0/L0/R0 look forward, back and 55 deg to either side
(DrivoR uses all four; LTF, DiffusionDrive and GTRS-Dense use F0/L0/R0). The PAI track
renders six f-theta cameras instead, none of which looks straight back. Each
nuPlan view is rendered here by rotating its pixel rays into the rig frame and
sampling the PAI camera that sees each ray closest to its optical axis
(front wide 120 for F0, the cross 120 cameras for L0/R0, the two rear 70 cameras
stitched for B0). Only rotations are used: camera positions differ by up to
about 2 m between the rigs, so close objects shift slightly.

Calibration comes from start_session (available_cameras): the f-theta
intrinsics are for the camera's native resolution and are scaled to each
delivered JPEG, whose size is scene-dependent (1916 or 1920 x 1080).

Also used by the SimScale sample drivers (drivers/simscale/*), which import it
from this package on the PAI track only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping

import numpy as np

try:
    import cv2
except ImportError:  # the challenge image has no OpenCV: numpy bilinear fallback
    cv2 = None

LOGGER = logging.getLogger(__name__)

# PAI cameras that can see the forward views (F0/L0/R0), and all those used
FRONT_SOURCE_IDS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
)
PAI_SOURCE_IDS = FRONT_SOURCE_IDS + (
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
)

# nuPlan intrinsics (identical for all cameras and vehicles) and camera-to-lidar
# rotations (w, x, y, z), from the nuPlan camera table (log 2021.08.17.16.48.45_veh-43).
# The lidar frame is forward-left-up like the AlpaSim rig; F0 yaw 0, L0 +55, R0 -55,
# B0 180 deg.
NUPLAN_HEIGHT = 1080
NUPLAN_WIDTH = 1920
NUPLAN_FOCAL = 1545.0
NUPLAN_PRINCIPAL_POINT = (960.0, 560.0)
NUPLAN_CAMERA_TO_RIG_QUAT = {
    "CAM_F0": (-0.48509849291694507, 0.5131702610820036, -0.506191551918007, 0.49508165798944426),
    "CAM_B0": (0.5008733453455183, -0.5006975413962285, -0.49496101666551, 0.5034296931485133),
    "CAM_L0": (-0.6643395320949048, 0.6842165618728846, -0.2059868110545875, 0.21924898226977985),
    "CAM_R0": (0.21440005434457995, -0.20786136180116455, 0.6801306873263874, -0.6695136437176936),
}


def _quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw_deg(camera_to_rig: np.ndarray) -> float:
    forward = camera_to_rig[:, 2]
    return float(np.degrees(np.arctan2(forward[1], forward[0])))


class _FthetaSource:
    """One PAI camera: rotation and f-theta projection at its native resolution."""

    def __init__(self, camera) -> None:
        spec = camera.intrinsics
        if spec.WhichOneof("camera_param") != "ftheta_param":
            raise ValueError(f"{camera.logical_id}: expected f-theta intrinsics")
        param = spec.ftheta_param
        q = camera.rig_to_camera.quat
        # rig_to_camera is the camera's pose in the rig: it maps camera coordinates
        # (OpenCV: x right, y down, z forward) to rig coordinates.
        self.camera_to_rig = _quat_to_matrix(q.w, q.x, q.y, q.z)
        self.native_hw = (int(spec.resolution_h), int(spec.resolution_w))
        self.principal_point = np.array(
            [param.principal_point_x, param.principal_point_y], dtype=np.float64
        )
        if param.HasField("linear_cde"):
            c, d, e = param.linear_cde.linear_c, param.linear_cde.linear_d, param.linear_cde.linear_e
        else:
            c, d, e = 1.0, 0.0, 0.0
        self.linear = np.array([[c, d], [e, 1.0]], dtype=np.float64)
        self.max_angle = param.max_angle if param.max_angle > 0 else None
        forward = np.asarray(param.angle_to_pixeldist_poly, dtype=np.float64)
        backward = np.asarray(param.pixeldist_to_angle_poly, dtype=np.float64)
        if forward.size:
            self._theta_to_radius = lambda theta: np.polynomial.polynomial.polyval(theta, forward)
        elif backward.size:
            # invert the pixel-distance-to-angle polynomial over the image diagonal
            radii = np.linspace(0.0, float(np.hypot(*self.native_hw)), 8192)
            thetas = np.polynomial.polynomial.polyval(radii, backward)
            keep = np.concatenate(([True], np.diff(thetas) > 0))
            keep &= np.cumprod(keep).astype(bool)  # monotonic prefix only
            radii, thetas = radii[keep], thetas[keep]
            if self.max_angle is None:
                self.max_angle = float(thetas[-1])
            self._theta_to_radius = lambda theta: np.interp(theta, thetas, radii)
        else:
            raise ValueError(f"{camera.logical_id}: f-theta intrinsics without polynomial")

    def project(self, rays_rig: np.ndarray, delivered_hw: tuple[int, int]):
        """Rig-frame unit rays (N, 3) -> delivered-image pixels (N, 2), theta, valid."""
        rays = rays_rig @ self.camera_to_rig  # = (camera_to_rig.T @ ray) per row
        xy = rays[:, :2]
        z = rays[:, 2]
        xy_norm = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(xy_norm, z)
        radius = self._theta_to_radius(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(xy_norm > 1e-9, radius / np.maximum(xy_norm, 1e-12), 0.0)
        pixels = (xy * scale[:, None]) @ self.linear.T + self.principal_point
        native_h, native_w = self.native_hw
        delivered_h, delivered_w = delivered_hw
        pixels *= np.array([delivered_w / native_w, delivered_h / native_h])
        valid = (
            (z > 0.0)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= delivered_w - 1.0)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= delivered_h - 1.0)
        )
        if self.max_angle is not None:
            valid &= theta <= self.max_angle
        return pixels, theta, valid


def _pinhole_rays_rig(camera_to_rig: np.ndarray) -> np.ndarray:
    xs = (np.arange(NUPLAN_WIDTH, dtype=np.float64) - NUPLAN_PRINCIPAL_POINT[0]) / NUPLAN_FOCAL
    ys = (np.arange(NUPLAN_HEIGHT, dtype=np.float64) - NUPLAN_PRINCIPAL_POINT[1]) / NUPLAN_FOCAL
    grid_x, grid_y = np.meshgrid(xs, ys)
    rays = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=-1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays @ camera_to_rig.T


def _remap_bilinear(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = np.clip(np.floor(map_x).astype(np.int64), 0, w - 2)
    y0 = np.clip(np.floor(map_y).astype(np.int64), 0, h - 2)
    fx = (map_x - x0)[..., None]
    fy = (map_y - y0)[..., None]
    img = image.astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return np.clip(top * (1 - fy) + bottom * fy + 0.5, 0, 255).astype(np.uint8)


def pai_source_ids(target_ids: Iterable[str]) -> tuple[str, ...]:
    """The PAI cameras the given nuPlan views are sampled from."""
    return PAI_SOURCE_IDS if "CAM_B0" in target_ids else FRONT_SOURCE_IDS


class NuplanViews:
    """Renders nuPlan views (some of CAM_F0/B0/L0/R0) from one session's PAI images."""

    @classmethod
    def for_rig(cls, available_cameras: Iterable, target_ids: Iterable[str]) -> NuplanViews | None:
        """The views for a PAI rig, or None if the session doesn't have its cameras."""
        available_cameras = list(available_cameras)
        available = {camera.logical_id for camera in available_cameras}
        if not set(pai_source_ids(target_ids)) <= available:
            return None
        return cls(available_cameras, target_ids)

    def __init__(self, available_cameras: Iterable, target_ids: Iterable[str]) -> None:
        target_ids = tuple(target_ids)
        unknown = [c for c in target_ids if c not in NUPLAN_CAMERA_TO_RIG_QUAT]
        if unknown:
            raise ValueError(f"no nuPlan calibration for {unknown}")
        self.source_ids = pai_source_ids(target_ids)
        cameras = {camera.logical_id: camera for camera in available_cameras}
        missing = [c for c in self.source_ids if c not in cameras]
        if missing:
            raise ValueError(f"missing PAI cameras: {missing}")
        self._sources = {c: _FthetaSource(cameras[c]) for c in self.source_ids}
        self._targets = {
            c: _quat_to_matrix(*NUPLAN_CAMERA_TO_RIG_QUAT[c]) for c in target_ids
        }
        # per nuPlan view: [(source id, map_x, map_y, mask)], built for the delivered sizes
        self._maps: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._maps_for_hw: dict[str, tuple[int, int]] | None = None
        self._dumped = False
        LOGGER.info(
            "PAI -> nuPlan views; source yaws %s",
            {c: round(_yaw_deg(s.camera_to_rig), 1) for c, s in self._sources.items()},
        )

    def _build_maps(self, delivered_hw: Mapping[str, tuple[int, int]]) -> None:
        self._maps = {}
        for target_id, camera_to_rig in self._targets.items():
            rays = _pinhole_rays_rig(camera_to_rig)
            best_theta = np.full(rays.shape[0], np.inf)
            best_source = np.full(rays.shape[0], -1, dtype=np.int64)
            projections = []
            for index, source_id in enumerate(self.source_ids):
                pixels, theta, valid = self._sources[source_id].project(
                    rays, delivered_hw[source_id]
                )
                better = valid & (theta < best_theta)
                best_theta[better] = theta[better]
                best_source[better] = index
                projections.append(pixels)
            entries = []
            for index, source_id in enumerate(self.source_ids):
                mask = best_source == index
                if not mask.any():
                    continue
                pixels = projections[index]
                map_x = np.where(mask, pixels[:, 0], -1.0).astype(np.float32)
                map_y = np.where(mask, pixels[:, 1], -1.0).astype(np.float32)
                shape = (NUPLAN_HEIGHT, NUPLAN_WIDTH)
                entries.append(
                    (source_id, map_x.reshape(shape), map_y.reshape(shape), mask.reshape(shape))
                )
            self._maps[target_id] = entries
            uncovered = float((best_source < 0).mean())
            LOGGER.info(
                "%s from %s (%.1f%% uncovered)",
                target_id,
                {s: round(float(m.mean()) * 100, 1) for s, _, _, m in entries},
                uncovered * 100,
            )
        self._maps_for_hw = dict(delivered_hw)

    def render(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """PAI images by logical id -> nuPlan views (1080x1920x3 uint8) by CAM id."""
        delivered_hw = {c: tuple(images[c].shape[:2]) for c in self.source_ids}
        if delivered_hw != self._maps_for_hw:
            self._build_maps(delivered_hw)
        views = {}
        for target_id, entries in self._maps.items():
            out = np.zeros((NUPLAN_HEIGHT, NUPLAN_WIDTH, 3), dtype=np.uint8)
            for source_id, map_x, map_y, mask in entries:
                if cv2 is not None:
                    sampled = cv2.remap(
                        images[source_id], map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                    )
                else:
                    sampled = _remap_bilinear(images[source_id], map_x, map_y)
                out[mask] = sampled[mask]
            out.setflags(write=False)
            views[target_id] = out
        dump_dir = os.environ.get("DRIVOR_PAI_VIEW_DUMP")
        if dump_dir and not self._dumped:  # debugging: the first views of each session
            from PIL import Image

            os.makedirs(dump_dir, exist_ok=True)
            for name, image in list(views.items()) + [(c, images[c]) for c in self.source_ids]:
                Image.fromarray(image).save(os.path.join(dump_dir, f"{id(self):x}_{name}.jpg"))
            self._dumped = True
        return views
