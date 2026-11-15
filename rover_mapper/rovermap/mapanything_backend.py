"""MapAnything (Meta, Apache-2.0 weights) as a drop-in network for the Mapper.

    m = MapAnythingMapper(process_res=336, metric=0)   # same API as Mapper

Only the network changes. Tracking, keyframes, alignment, floor calibration
and masking are Mapper's own code, so a DA3 / MapAnything comparison measures
the network, not the glue.

Network call (`_infer`)
* Each frame is resized (no crop) so its longest side is `process_res`, both
  sides multiples of the ViT patch (14 px). For 640x480 input: 336 -> 336x252,
  the same grid DA3 uses at process_res=336; 504 -> 504x378 (close to the
  518 px MapAnything was trained at).
* Image-only inference by default (no intrinsics / poses / depth given; known
  intrinsics can be passed with `intrinsics=(fx, fy, cx, cy)`), all views of one
  call reconstructed jointly. MapAnything returns cam->world poses in OpenCV
  axes relative to its own world (the first view), Z depth in metres,
  intrinsics at the processed resolution and a confidence, i.e. the same
  contract as Mapper._infer. Pixels MapAnything flags as ambiguous (sky,
  mirrors, ...) get depth 0 and are dropped by Mapper._valid_mask; its own
  edge masking is off because Mapper._valid_mask already removes flying
  pixels for both networks.
* Mixed precision: fp16 autocast on MPS (bf16 is reported unsupported on MPS by
  MapAnything's device helper, which falls back to fp16); `amp="fp32"` for
  full precision. The prediction heads always run in fp32 (MapAnything
  disables autocast there).

Metric scale (`metric`)
  0  Exactly the DA3 pipeline: floor fit fixes the metric scale (cam_height),
     every new frame is aligned to the map by a similarity, so its scale is
     taken from the stored keyframes (scale chains from keyframe to keyframe).
  1  Trust the network's metres for the map scale: the floor fit only levels
     the map (up axis, origin on the floor); no floor rescaling, no
     ground-robot floor re-anchoring. Alignment still a similarity, so after
     the first keyframe the scale chains exactly as in mode 0.
  2  Mode 1 + rigid alignment: the scale in `_align` is fixed to 1, so every
     inference's own metric depth and translation are used as they are and
     scale cannot drift through the keyframe chain. The measured map/network
     depth ratio is still stored in `last_align[2]` for diagnosis.
In modes 1-2 the floor heights fitted on keyframes are still recorded in
`ground_heights` (in metres): a free check of the metric scale against the
known camera height.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from .mapper import Keyframe, Mapper, _rot_z, fit_ground, project_to_so3, rot_angle_deg

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(_ROOT, "third_party", "map-anything-weights")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # "dinov2" normalisation
_STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14


def target_size(h: int, w: int, process_res: int) -> tuple[int, int]:
    """(W, H) with the longest side ~process_res, both multiples of 14."""
    long_ = max(PATCH, int(round(process_res / PATCH)) * PATCH)
    if w >= h:
        return long_, max(PATCH, int(round(h * long_ / w / PATCH)) * PATCH)
    return max(PATCH, int(round(w * long_ / h / PATCH)) * PATCH), long_


class MapAnythingMapper(Mapper):
    def __init__(
        self,
        model_name: str = DEFAULT_WEIGHTS,  # local dir with config.json + model.safetensors
        process_res: int = 336,
        cam_height: float = 0.10,
        n_recent: int = 2,
        n_near: int = 1,
        kf_translation: float = 0.07,
        kf_rotation: float = 12.0,
        point_stride: int = 3,
        max_depth: float = 6.0,
        voxel: float = 0.015,
        conf_percentile: float = 25.0,
        calib_frames: int = 6,
        min_quality: float = 0.3,
        min_overlap: float = 0.3,
        lost_patience: int = 10,
        sensor_range: tuple[float, float] = (0.07, 0.8),
        min_texture: float = 15.0,
        min_kf_texture: float = 50.0,
        mask_moving: bool = True,
        ground_robot: bool = False,
        device: str | None = None,
        metric: int = 0,          # 0 floor scale | 1 network metres | 2 + rigid align
        amp: str = "fp16",        # fp16 | fp32
        model=None,               # an already built MapAnything (tests, reuse)
        # Known pinhole intrinsics (fx, fy, cx, cy) at the input image resolution.
        # None (default): image-only, MapAnything estimates them like DA3 does.
        intrinsics: tuple[float, float, float, float] | None = None,
    ):
        # Mapper.__init__ would load DA3: set the same attributes here instead.
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        if model is None:
            from mapanything.models import MapAnything
            if os.path.isdir(model_name) and not os.path.exists(os.path.join(model_name, "model.safetensors")):
                raise FileNotFoundError(f"{model_name}/model.safetensors missing (weights not downloaded yet?)")
            model = MapAnything.from_pretrained(model_name)
        self.model = model.to(self.device).eval()
        self.amp = str(amp)
        self.intrinsics = None if intrinsics is None else tuple(float(v) for v in intrinsics)
        self.metric = int(metric)
        self.process_res = process_res
        self.cam_height = cam_height
        self.n_recent, self.n_near = n_recent, n_near
        self.kf_translation, self.kf_rotation = kf_translation, kf_rotation
        self.point_stride = point_stride
        self.max_depth = max_depth
        self.voxel = voxel
        self.conf_percentile = conf_percentile
        self.calib_frames = max(1, calib_frames)
        self.min_quality, self.min_overlap = min_quality, min_overlap
        self.lost_patience = lost_patience
        self.sensor_range = tuple(sensor_range)
        self.min_texture, self.min_kf_texture = min_texture, min_kf_texture
        self.ground_robot = ground_robot
        self.masker = None
        if mask_moving:
            from .dynamic import MovingMasker
            self.masker = MovingMasker(self.device)
        self.last_moving_share = 0.0
        self.rng = np.random.default_rng(0)
        self.last_align = (0.0, 0.0, 1.0)
        self.reset()

    # -- network ----------------------------------------------------------
    def _infer(self, images: list[np.ndarray]):
        h0, w0 = images[0].shape[:2]
        W, H = target_size(h0, w0, self.process_res)
        imgs, views = [], []
        for im in images:
            interp = cv2.INTER_AREA if im.shape[1] > W else cv2.INTER_CUBIC
            r = cv2.resize(im, (W, H), interpolation=interp)
            imgs.append(r)
            x = (r.astype(np.float32) / 255.0 - _MEAN) / _STD
            view = {"img": torch.from_numpy(x).permute(2, 0, 1)[None].contiguous(),
                    "data_norm_type": ["dinov2"]}
            if self.intrinsics is not None:
                fx, fy, cx, cy = self.intrinsics
                sx, sy = W / im.shape[1], H / im.shape[0]
                # pixel centres: u' + 0.5 = s (u + 0.5)
                view["intrinsics"] = torch.tensor([[[fx * sx, 0.0, (cx + 0.5) * sx - 0.5],
                                                    [0.0, fy * sy, (cy + 0.5) * sy - 0.5],
                                                    [0.0, 0.0, 1.0]]], dtype=torch.float32)
            views.append(view)
        preds = self.model.infer(
            views, memory_efficient_inference=False, use_amp=self.amp != "fp32",
            amp_dtype=self.amp, apply_mask=False, mask_edges=False)
        f32 = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
        depth = np.stack([f32(p["depth_z"][0, ..., 0]) for p in preds]).astype(np.float32)
        conf = np.stack([f32(p["conf"][0]) for p in preds]).astype(np.float32)
        amb = np.stack([f32(p["non_ambiguous_mask"][0]) for p in preds]) < 0.5
        depth[amb | ~np.isfinite(depth)] = 0.0
        c2w = np.stack([f32(p["camera_poses"][0]) for p in preds]).astype(np.float64)
        K = np.stack([f32(p["intrinsics"][0]) for p in preds]).astype(np.float64)
        self.last_metric_factor = float(f32(preds[0]["metric_scaling_factor"]).ravel()[0])
        return depth, conf, c2w, K, np.stack(imgs)

    # -- metric-scale variants ----------------------------------------------
    def _try_calibrate(self, kf: Keyframe) -> bool:
        if self.metric == 0:
            return super()._try_calibrate(kf)
        # Same as Mapper._try_calibrate with s = 1: level the map, origin on the
        # floor below this keyframe, keep the network's metres.
        g = fit_ground(kf.depth, kf.K, kf.valid, self.rng)
        if g is None:
            return False
        n_cam, h = g
        up = kf.c2w[:3, :3] @ n_cam
        fwd = kf.c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        x = fwd - (fwd @ up) * up
        if np.linalg.norm(x) < 1e-6:
            return False
        x /= np.linalg.norm(x)
        y = np.cross(up, x)
        R = np.stack([x, y, up])
        t = np.array([0.0, 0.0, h]) - R @ kf.c2w[:3, 3]
        self._apply_similarity(1.0, R, t)
        self.scale_calibrated = True
        self.ground_heights = [h]  # measured camera height (m): metric check
        self.calib_kfs = [kf]
        if self.ground_robot:
            Rc = kf.c2w[:3, :3]
            yaw0 = float(np.arctan2(Rc[1, 2], Rc[0, 2]))
            self._mount = _rot_z(-yaw0) @ Rc
            self._cam_z = float(kf.c2w[2, 3])
        return True

    def _refine_scale(self, kf: Keyframe) -> bool:
        if self.metric == 0:
            return super()._refine_scale(kf)
        # Record the floor height (diagnostic only), never rescale.
        g = fit_ground(kf.depth, kf.K, kf.valid, self.rng)
        if g is not None and 0.5 * self.cam_height < g[1] < 2.0 * self.cam_height:
            self.ground_heights.append(g[1])
            self.calib_kfs.append(kf)
        return False

    def _floor_anchor(self, depth, K, valid):
        return super()._floor_anchor(depth, K, valid) if self.metric == 0 else depth

    def _align(self, refs: list[Keyframe], depth: np.ndarray, L: np.ndarray):
        if self.metric < 2:
            return super()._align(refs, depth, L)
        # Mapper._align with the scale fixed to 1 (rigid): trust the network's metres.
        m = len(refs)
        ratios = []
        for i, k in enumerate(refs):
            kd, kv = k.depth, k.valid
            if kd.shape != depth[i].shape:
                hw = (depth[i].shape[1], depth[i].shape[0])
                kd = cv2.resize(kd, hw, interpolation=cv2.INTER_NEAREST)
                kv = cv2.resize(kv.astype(np.uint8), hw, interpolation=cv2.INTER_NEAREST) > 0
            ok = kv & (depth[i] > 1e-6)
            if ok.sum() > 100:
                ratios.append(float(np.median(kd[ok] / depth[i][ok])))
        s_meas = float(np.median(ratios)) if ratios else 1.0
        s = 1.0
        R = project_to_so3(sum(k.c2w[:3, :3] @ L[i, :3, :3].T for i, k in enumerate(refs)))
        ts = np.stack([k.c2w[:3, 3] - s * R @ L[i, :3, 3] for i, k in enumerate(refs)])
        t = np.median(ts, axis=0)
        c2w = np.eye(4)
        c2w[:3, :3] = R @ L[m, :3, :3]
        c2w[:3, 3] = s * R @ L[m, :3, 3] + t
        scene = float(np.median(depth[m])) * s
        pos_err = float(np.max(np.linalg.norm(ts - t, axis=1))) / max(scene, 1e-6)
        rot_err = max(rot_angle_deg(R @ L[i, :3, :3], k.c2w[:3, :3]) for i, k in enumerate(refs))
        quality = float(np.clip(1.0 - max(pos_err / 0.15, rot_err / 15.0), 0.0, 1.0))
        self.last_align = (pos_err, rot_err, s_meas)
        return c2w, s, quality
