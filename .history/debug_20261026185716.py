"""depth_source.py - the single place that turns an RGB frame into a RAW (uncalibrated) depth map.

resnet : ResNetDepth student (weights/best_model.pth). Input resized to 180x180 (INTER_AREA, as in
         training), output = raw network output (no activation), 180x180.
dino   : precomputed DINOv3 maps dataset/depth_dino/<SESSION>/image_<id>_depth.npy (teacher, reference only).

scale_shift.json is only valid for the source it was fitted on (it stores "source").
Camera-only: no MoCap/pose is used here.
"""
from pathlib import Path

import cv2
import numpy as np

SESSION = "2025_0813_1543-S3"
NET_SIZE = 180
WEIGHTS_PATH = "weights/best_model.pth"
DEPTH_ROOT = Path("dataset/depth_dino") / SESSION


class DepthSource:
    def __init__(self, source="resnet", weights=WEIGHTS_PATH, depth_root=DEPTH_ROOT):
        if source not in ("resnet", "dino"):
            raise ValueError("source must be 'resnet' or 'dino'")
        self.source = source
        self.depth_root = Path(depth_root)
        if source == "resnet":
            import torch
            from model import ResNetDepth
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = ResNetDepth().to(self.device)
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            self.model.eval()

    def depth(self, img_id, bgr):
        """Raw 2D depth (float32) for one frame, or None. Native size: 180x180 (resnet)."""
        if self.source == "dino":
            p = self.depth_root / f"image_{img_id}_depth.npy"
            if not p.exists():
                return None
            d = np.squeeze(np.load(p)).astype(np.float32)
            return d if d.ndim == 2 else None
        rgb = cv2.cvtColor(cv2.resize(bgr, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        t = (self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
             .to(self.device))
        with self.torch.inference_mode():
            out = self.model(t)
        d = out.squeeze().cpu().numpy().astype(np.float32)
        return d if d.ndim == 2 else None