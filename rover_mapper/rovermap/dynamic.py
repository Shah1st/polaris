"""Masks of things that move (people, animals, vehicles), kept out of the map.

The mapper assumes a still world. People walking by otherwise get painted into
the map (seen live: legs in the 3D view) and disturb tracking. A small
semantic segmentation network (torchvision DeepLabV3-MobileNetV3, 21 VOC
classes) marks those pixels; the mapper then ignores them.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

# VOC class ids that can move on their own.
MOVING = {
    "person": 15, "bicycle": 2, "bird": 3, "boat": 4, "bus": 6, "car": 7, "cat": 8,
    "cow": 10, "dog": 12, "horse": 13, "motorbike": 14, "sheep": 17, "train": 19,
}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class MovingMasker:
    def __init__(self, device: torch.device, width: int = 384, dilate_px: int = 9):
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)

        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        self.net = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(device)
        self.device = device
        self.width = width
        self.ids = torch.tensor(sorted(MOVING.values()), device=device)
        # Grow the mask a little: segmentation edges are soft, and depth at a
        # person's outline is unreliable anyway.
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        """Boolean mask (True = moving object) at resolution out_hw = (h, w)."""
        h0, w0 = rgb.shape[:2]
        h = max(32, int(round(self.width * h0 / w0 / 8)) * 8)
        x = cv2.resize(rgb, (self.width, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        cls = self.net(t)["out"].argmax(1)[0]
        moving = torch.isin(cls, self.ids).to(torch.uint8).cpu().numpy()
        if moving.any():
            moving = cv2.dilate(moving, self.kernel)
        return cv2.resize(moving, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
