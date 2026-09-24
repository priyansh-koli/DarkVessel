"""A learned ship detector: a small U-Net that predicts a heatmap of ship centres.

Why centres and not boxes: the `Detector` protocol asks for points, fusion matches points, and
most Sentinel-1 ships are a handful of pixels — a box adds parameters without adding anything
the pipeline uses. The network is CenterNet-style: one output channel, a Gaussian bump at each
ship's centre, peaks read off with a max-pool. It runs at full resolution because the smallest
LS-SSDD ships are ~5 px and a stride-4 output would merge neighbours.

Inputs are normalised per window by robust statistics (median and MAD of the in-swath
pixels), not by fixed constants, so the same weights work on 8-bit LS-SSDD chips and on
calibrated backscatter — only contrast against the surrounding sea matters. A second input
channel marks no-data (pixels at exactly zero, outside the swath) so the edge of the swath is
never mistaken for a hull.

Requires the `detector` extra (`pip install -e ".[detector]"`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional name
from torch import nn

from darkvessel.data.tiling import Tiling

_CLIP = (-4.0, 40.0)


class CentreNet(nn.Module):
    """U-Net, four scales, `width` channels at full resolution. ~0.4 M parameters at 16.

    BatchNorm by default, with its statistics recomputed for the final weights before they are
    saved (see `train`): the running averages lag the weights at a high learning rate, and
    made validation AP swing between epochs. `norm="group"` is kept for comparison; it trains
    half as fast on Apple GPUs and plateaued lower.
    """

    def __init__(self, width: int = 16, norm: str = "batch") -> None:
        super().__init__()
        c = [width, width * 2, width * 4, width * 6]
        self.stem = _block(2, c[0], norm)
        self.down = nn.ModuleList([_block(c[i], c[i + 1], norm) for i in range(3)])
        self.up = nn.ModuleList(
            [_block(c[i + 1] + c[i], c[i], norm) for i in reversed(range(3))]
        )
        self.head = nn.Sequential(
            nn.Conv2d(c[0], c[0], 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c[0], 1, 1)
        )
        # Start from "almost nowhere is a ship" (CenterNet's prior), or the first steps are
        # spent unlearning a 50% guess over millions of sea pixels.
        nn.init.constant_(self.head[-1].bias, -4.6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for down in self.down:
            skips.append(down(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for up in self.up:
            skip = skips.pop()
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(torch.cat([y, skip], dim=1))
        return self.head(y)


def _block(cin: int, cout: int, norm: str) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        _norm(cout, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        _norm(cout, norm),
        nn.ReLU(inplace=True),
    )


def _norm(channels: int, kind: str) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(min(8, channels // 4), channels)


def prepare(window: np.ndarray) -> np.ndarray:
    """(2, H, W) float32: the window in robust z-units against its own sea, and a valid mask."""
    image = np.asarray(window, dtype=np.float32)
    valid = np.isfinite(image) & (image != 0)
    if valid.sum() < 16:
        return np.stack([np.zeros_like(image), valid.astype(np.float32)])
    values = image[valid]
    median = float(np.median(values))
    spread = 1.4826 * float(np.median(np.abs(values - median)))
    if spread <= 1e-6:
        # Quantised 8-bit calm sea can have a MAD of zero; fall back to the standard deviation.
        spread = float(values.std()) + 1e-6
    z = np.where(valid, (image - median) / spread, 0.0)
    return np.stack([np.clip(z, *_CLIP) / 10.0, valid.astype(np.float32)]).astype(np.float32)


def peaks(heat: torch.Tensor, valid: torch.Tensor, floor: float, radius: int = 2) -> np.ndarray:
    """(n, 3) (row, col, score) local maxima of a (H, W) heatmap at or above `floor`."""
    pooled = F.max_pool2d(heat[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]
    keep = (heat == pooled) & (heat >= floor) & (valid > 0)
    rows, cols = torch.nonzero(keep, as_tuple=True)
    scores = heat[rows, cols]
    return torch.stack([rows.float(), cols.float(), scores], dim=1).cpu().numpy()


class CNNDetector:
    """Satisfies the `Detector` protocol with trained `CentreNet` weights."""

    def __init__(
        self, weights: str | Path, threshold: float | None = None, device: str | None = None
    ) -> None:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.device = torch.device(device or default_device())
        self.model = CentreNet(
            width=checkpoint.get("width", 16), norm=checkpoint.get("norm", "batch")
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        # The operating point is chosen on validation scenes at training time and shipped with
        # the weights, so the detector behaves as reported unless a caller overrides it.
        self.threshold = float(threshold if threshold is not None else checkpoint["threshold"])
        self.metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}

    # Tiles a run uses when its configuration names none. On a 4800 px mosaic of LS-SSDD scene
    # 12, 512 and 1024 px tiles found the same 14 ships, and 1024 ran 17% faster (M5 GPU).
    # Batching several windows per forward pass was tried and gained nothing: at this size
    # one window already keeps the GPU busy.
    preferred_tiling = Tiling(tile_px=1024, overlap_px=64)

    def __call__(self, window: np.ndarray) -> list[tuple[float, float]]:
        return [(float(r), float(c)) for r, c, _ in self.scored(window, floor=self.threshold)]

    @torch.inference_mode()
    def scored(self, window: np.ndarray, floor: float = 0.05) -> np.ndarray:
        x = torch.from_numpy(prepare(window))[None].to(self.device)
        heat = torch.sigmoid(self.model(x))[0, 0]
        return peaks(heat, x[0, 1], floor)


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
