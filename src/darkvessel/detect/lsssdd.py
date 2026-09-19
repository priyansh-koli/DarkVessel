"""LS-SSDD-v1.0: reading it, and auditing it before anything is trained on it.

LS-SSDD-v1.0 (Zhang et al., Remote Sensing 2020) is 15 Sentinel-1 IW scenes of 24000 x 16000
pixels, cut into 9000 sub-images of 800 x 800 with Pascal VOC boxes around every ship, split
by scene: scenes 01-10 train, 11-15 test. Sentinel-1 IW at this pixel scale is what this
pipeline runs on, which is why it is the training set rather than a higher-resolution one.

Every label is checked before it is trusted, and every rule counts what it changed — the same
discipline as `data.ais.clean`, for the same reason: a model trained on a silently-dropped
label, or evaluated on one, reports a number nobody can audit. Problems a rule can fix without
guessing (a box running past the image edge, an exact duplicate) are fixed; problems it cannot
(an unreadable image) drop the sample; anything merely surprising is flagged and kept.

Get the data from the authors' release: https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN
(Apache-2.0), and unzip it so that `<root>/JPEGImages_sub`, `<root>/Annotations_sub` and
`<root>/ImageSets/Main` exist.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 800
TRAIN_SCENES = [f"{n:02d}" for n in range(1, 11)]
TEST_SCENES = [f"{n:02d}" for n in range(11, 16)]
# Validation is carved out of the training scenes by *scene*, never by sub-image: neighbouring
# sub-images share sea state, incidence and often the same ships, so a random split would leak.
VAL_SCENES = ["05", "10"]

# A ship longer than this many pixels (1.5 km at ~5 m) is almost certainly a labelling slip.
_SUSPICIOUS_BOX_PX = 300
# Pixels at exactly zero are outside the radar swath (no data), not dark sea.
_NODATA_FLAG_FRACTION = 0.5


@dataclass
class Sample:
    """One 800 x 800 sub-image and the ship boxes (x0, y0, x1, y1, in pixels) inside it."""

    name: str
    image_path: Path
    boxes: np.ndarray  # (n, 4) float32, x0 y0 x1 y1
    nodata_fraction: float = 0.0

    @property
    def scene(self) -> str:
        return self.name.split("_")[0]

    @property
    def centres(self) -> np.ndarray:
        """(n, 2) ship centres as (row, col) — the `Detector` protocol's frame."""
        if len(self.boxes) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        rows = (self.boxes[:, 1] + self.boxes[:, 3]) / 2
        cols = (self.boxes[:, 0] + self.boxes[:, 2]) / 2
        return np.stack([rows, cols], axis=1).astype(np.float32)


@dataclass
class Audit:
    """What the audit found, per rule, in the order the rules ran."""

    images: int = 0
    boxes_read: int = 0
    counts: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def note(self, rule: str, where: str, n: int = 1) -> None:
        self.counts[rule] += n
        self.examples.setdefault(rule, [])
        if len(self.examples[rule]) < 5:
            self.examples[rule].append(where)

    def lines(self) -> list[str]:
        out = [f"{self.images} images, {self.boxes_read} boxes read"]
        for rule, count in self.counts.items():
            shown = ", ".join(self.examples.get(rule, []))
            out.append(f"  {rule}: {count}" + (f"  (e.g. {shown})" if shown else ""))
        return out


def split_names(root: Path, split: str) -> list[str]:
    """`train`, `val` or `test` (and `test_inshore` / `test_offshore`) sub-image names.

    `train` here is the official train list minus the validation scenes.
    """
    main = root / "ImageSets" / "Main"
    if split in ("train", "val"):
        names = _read_list(main / "train.txt")
        keep = (lambda s: s in VAL_SCENES) if split == "val" else (lambda s: s not in VAL_SCENES)
        return [n for n in names if keep(n.split("_")[0])]
    return _read_list(main / f"{split}.txt")


def load(root: Path, names: list[str], audit: Audit | None = None) -> list[Sample]:
    """Read and audit `names`. Samples that cannot be read are dropped and counted."""
    audit = audit if audit is not None else Audit()
    samples = []
    for name in names:
        sample = _load_one(root, name, audit)
        if sample is not None:
            samples.append(sample)
    return samples


def audit_splits(root: Path) -> Audit:
    """Checks that span splits: the same sub-image, or the same scene, on both sides."""
    audit = Audit()
    train = set(_read_list(root / "ImageSets" / "Main" / "train.txt"))
    test = set(_read_list(root / "ImageSets" / "Main" / "test.txt"))
    overlap = sorted(train & test)
    if overlap:
        audit.note("sub-image in both train and test", overlap[0], len(overlap))
    shared = sorted({n.split("_")[0] for n in train} & {n.split("_")[0] for n in test})
    if shared:
        audit.note("scene in both train and test", ",".join(shared), len(shared))
    inshore = set(_read_list(root / "ImageSets" / "Main" / "test_inshore.txt"))
    offshore = set(_read_list(root / "ImageSets" / "Main" / "test_offshore.txt"))
    if inshore | offshore != test:
        audit.note("inshore + offshore does not equal test", "", len((inshore | offshore) ^ test))
    listed = train | test
    on_disk = {p.stem for p in (root / "JPEGImages_sub").glob("*.jpg")}
    if on_disk - listed:
        audit.note("image on disk in no split", sorted(on_disk - listed)[0], len(on_disk - listed))
    return audit


def read_image(path: Path) -> np.ndarray:
    """A sub-image as float32 in [0, 1], single band."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _load_one(root: Path, name: str, audit: Audit) -> Sample | None:
    audit.images += 1
    image_path = root / "JPEGImages_sub" / f"{name}.jpg"
    label_path = root / "Annotations_sub" / f"{name}.xml"

    if not image_path.exists():
        audit.note("image missing (dropped)", name)
        return None
    try:
        with Image.open(image_path) as image:
            image.verify()
        pixels = read_image(image_path)
    except Exception:  # noqa: BLE001 - any decoder failure means the same thing here
        audit.note("image unreadable (dropped)", name)
        return None
    if pixels.shape != (SIZE, SIZE):
        audit.note("image not 800 x 800 (dropped)", f"{name} {pixels.shape}")
        return None

    nodata = float((pixels == 0).mean())
    if nodata >= _NODATA_FLAG_FRACTION:
        audit.note("image mostly outside the swath (kept, no-data masked)", name)

    if not label_path.exists():
        audit.note("label file missing (kept as no ships)", name)
        boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        try:
            boxes = _read_boxes(label_path, name, audit)
        except ET.ParseError:
            audit.note("label file unparseable (dropped)", name)
            return None

    audit.boxes_read += len(boxes)
    boxes = _clean_boxes(boxes, name, audit)
    return Sample(name=name, image_path=image_path, boxes=boxes, nodata_fraction=nodata)


def _read_boxes(path: Path, name: str, audit: Audit) -> np.ndarray:
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.iter("object"):
        label = (obj.findtext("name") or "").strip().lower()
        if label != "ship":
            audit.note("object not labelled 'ship' (kept)", f"{name}:{label!r}")
        bnd = obj.find("bndbox")
        if bnd is None:
            audit.note("object without a box (dropped)", name)
            continue
        try:
            boxes.append([float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        except (TypeError, ValueError):
            audit.note("box with non-numeric corners (dropped)", name)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _clean_boxes(boxes: np.ndarray, name: str, audit: Audit) -> np.ndarray:
    if len(boxes) == 0:
        return boxes

    swapped = (boxes[:, 0] > boxes[:, 2]) | (boxes[:, 1] > boxes[:, 3])
    if swapped.any():
        audit.note("box corners swapped (fixed)", name, int(swapped.sum()))
        boxes = np.stack(
            [
                np.minimum(boxes[:, 0], boxes[:, 2]),
                np.minimum(boxes[:, 1], boxes[:, 3]),
                np.maximum(boxes[:, 0], boxes[:, 2]),
                np.maximum(boxes[:, 1], boxes[:, 3]),
            ],
            axis=1,
        )

    clipped = np.clip(boxes, 0, SIZE)
    outside = (clipped != boxes).any(axis=1)
    if outside.any():
        audit.note("box past the image edge (clipped)", name, int(outside.sum()))
    boxes = clipped

    empty = (boxes[:, 2] - boxes[:, 0] < 1) | (boxes[:, 3] - boxes[:, 1] < 1)
    if empty.any():
        audit.note("box under one pixel (dropped)", name, int(empty.sum()))
        boxes = boxes[~empty]

    _, first = np.unique(np.round(boxes, 1), axis=0, return_index=True)
    if len(first) < len(boxes):
        audit.note("exact duplicate box (dropped)", name, len(boxes) - len(first))
        boxes = boxes[np.sort(first)]

    # Two hulls cannot overlap this much at ~5 m pixels; this is one ship boxed twice, and
    # left in, it would count as a miss against every detector.
    keep = _without_near_duplicates(boxes)
    if len(keep) < len(boxes):
        audit.note("same ship boxed twice, IoU > 0.7 (merged)", name, len(boxes) - len(keep))
        boxes = boxes[keep]

    longest = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    suspicious = int((longest > _SUSPICIOUS_BOX_PX).sum())
    if suspicious:
        audit.note(f"box longer than {_SUSPICIOUS_BOX_PX} px (kept)", name, suspicious)
    return boxes.astype(np.float32)


def _without_near_duplicates(boxes: np.ndarray, iou_above: float = 0.7) -> np.ndarray:
    """Indices to keep: the first of every group of boxes overlapping above `iou_above`."""
    if len(boxes) < 2:
        return np.arange(len(boxes))
    x0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter)
    keep: list[int] = []
    for i in range(len(boxes)):
        if all(iou[i, j] <= iou_above for j in keep):
            keep.append(i)
    return np.asarray(keep)


def _read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
