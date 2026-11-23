"""Core data contracts shared by every tool and agent.

Coordinate conventions
----------------------
* A ``BBox`` is ``[x0, y0, x1, y1]`` in pixels, ``x0 < x1`` and ``y0 < y1``.
* Boxes reported in outputs are always in ORIGINAL page coordinates: the pixel grid of the
  page as loaded from the input file (for PDFs: the page rasterised at ``loader.pdf_dpi``).
* Every image operation that changes geometry (rotate, deskew, dewarp, crop, resize) is
  expressed as a ``Transform`` (3x3 homography) mapping coordinates of its input image to
  coordinates of its output image. ``Page.transform`` maps ORIGINAL -> CURRENT image coords,
  so a box found on the processed image goes back with ``page.to_original(box)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (self.x0 <= self.x1 and self.y0 <= self.y1):
            raise ValueError(f"invalid bbox {self.as_list()}: need x0<=x1 and y0<=y1")

    @classmethod
    def from_seq(cls, seq) -> BBox:
        x0, y0, x1, y1 = (float(v) for v in seq)
        return cls(x0, y0, x1, y1)

    @classmethod
    def from_points(cls, pts: np.ndarray) -> BBox:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return cls(float(pts[:, 0].min()), float(pts[:, 1].min()),
                   float(pts[:, 0].max()), float(pts[:, 1].max()))

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def corners(self) -> np.ndarray:
        """4x2 array: top-left, top-right, bottom-right, bottom-left."""
        return np.array([[self.x0, self.y0], [self.x1, self.y0],
                         [self.x1, self.y1], [self.x0, self.y1]], dtype=np.float64)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def clip(self, width: float, height: float) -> BBox:
        return BBox(min(max(self.x0, 0.0), width), min(max(self.y0, 0.0), height),
                    min(max(self.x1, 0.0), width), min(max(self.y1, 0.0), height))

    def iou(self, other: BBox) -> float:
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Transform:
    """A 3x3 homography mapping input-image coords to output-image coords."""

    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError("Transform matrix must be 3x3")
        if abs(np.linalg.det(m)) < 1e-12:
            raise ValueError("Transform matrix is singular")
        object.__setattr__(self, "matrix", m / m[2, 2] if m[2, 2] != 0 else m)

    # ---- constructors -------------------------------------------------------------
    @classmethod
    def identity(cls) -> Transform:
        return cls(np.eye(3))

    @classmethod
    def translation(cls, tx: float, ty: float) -> Transform:
        return cls(np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64))

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> Transform:
        sy = sx if sy is None else sy
        return cls(np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64))

    @classmethod
    def rotation(cls, degrees: float, cx: float = 0.0, cy: float = 0.0) -> Transform:
        """Counter-clockwise rotation (in image coords, y down) about (cx, cy)."""
        t = np.deg2rad(degrees)
        c, s = np.cos(t), np.sin(t)
        rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
        return Transform.translation(-cx, -cy).then(Transform(rot)).then(Transform.translation(cx, cy))

    @classmethod
    def rot90(cls, k: int, width: float, height: float) -> Transform:
        """Exact pixel mapping of ``np.rot90(img, k)`` for an image of size width x height.

        Matches PIL ``Image.rotate(90 * k, expand=True)`` (counter-clockwise).
        """
        k %= 4
        if k == 0:
            return cls.identity()
        if k == 1:  # (x, y) -> (y, W - x)
            m = [[0, 1, 0], [-1, 0, width], [0, 0, 1]]
        elif k == 2:  # (x, y) -> (W - x, H - y)
            m = [[-1, 0, width], [0, -1, height], [0, 0, 1]]
        else:  # (x, y) -> (H - y, x)
            m = [[0, -1, height], [1, 0, 0], [0, 0, 1]]
        return cls(np.array(m, dtype=np.float64))

    # ---- algebra ------------------------------------------------------------------
    def then(self, other: Transform) -> Transform:
        """Apply ``self`` first, then ``other``."""
        return Transform(other.matrix @ self.matrix)

    def compose(self, other: Transform) -> Transform:
        """Alias of :meth:`then`: the returned transform applies ``self`` then ``other``."""
        return self.then(other)

    def inverse(self) -> Transform:
        return Transform(np.linalg.inv(self.matrix))

    def is_identity(self, tol: float = 1e-9) -> bool:
        return bool(np.allclose(self.matrix, np.eye(3), atol=tol))

    # ---- mapping ------------------------------------------------------------------
    def map_points(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homog = np.hstack([pts, np.ones((len(pts), 1))]) @ self.matrix.T
        return homog[:, :2] / homog[:, 2:3]

    def map_quad(self, box: BBox) -> np.ndarray:
        """The box's 4 corners mapped through the transform (a quad, 4x2)."""
        return self.map_points(box.corners())

    def map_box(self, box: BBox | list | tuple) -> BBox:
        """Axis-aligned bounding box of the mapped corners."""
        b = box if isinstance(box, BBox) else BBox.from_seq(box)
        return BBox.from_points(self.map_quad(b))

    def to_list(self) -> list[list[float]]:
        return self.matrix.tolist()

    @classmethod
    def from_list(cls, m) -> Transform:
        return cls(np.asarray(m, dtype=np.float64))


@dataclass
class Word:
    text: str
    bbox: BBox
    confidence: float = 1.0
    page: int = 1  # 1-based


@dataclass
class Page:
    """One page image plus the geometry needed to map back to the original page.

    ``index`` is 1-based. ``original_size`` is (width, height) of the page as loaded.
    ``image`` may be a processed version; ``transform`` maps original -> current coords.
    """

    index: int
    image: np.ndarray
    original_size: tuple[int, int]
    transform: Transform = field(default_factory=Transform.identity)
    source: str = ""  # e.g. "png", "pdf@200dpi", "tiff-frame"
    words: list[Word] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return int(w), int(h)

    def to_original(self, box: BBox | list | tuple) -> BBox:
        w, h = self.original_size
        return self.transform.inverse().map_box(box).clip(w, h)

    def to_current(self, box: BBox | list | tuple) -> BBox:
        return self.transform.map_box(box)

    def with_image(self, image: np.ndarray, step: Transform) -> Page:
        """A new Page for a processed image; ``step`` maps current -> new image coords."""
        return Page(index=self.index, image=image, original_size=self.original_size,
                    transform=self.transform.then(step), source=self.source,
                    words=[], meta=dict(self.meta))
