# -*- coding: utf-8 -*-
"""
R3 - Traffic light color detection.

Public API:
    detect_light(frame) -> "RED" / "GREEN" / None

The detector uses HSV masks because hue separates red/green more reliably than
raw BGR values. Red wraps around the HSV hue boundary, so it is represented by
two ranges and merged into one mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from config import CONFIG
except ImportError:
    CONFIG = {}


@dataclass(frozen=True)
class LightEvidence:
    """Debug information that R4/R6 can log while tuning."""

    color: str | None
    red_area: int
    green_area: int
    red_ratio: float
    green_ratio: float
    roi: tuple[int, int, int, int]


DEFAULT_LIGHT_CONFIG: dict[str, Any] = {
    "roi": (0.35, 0.00, 0.30, 0.45),
    "min_area": 35,
    "min_ratio": 0.0015,
    "winner_margin": 1.25,
    "kernel_size": 5,
    "blur_size": 5,
    "red_ranges": [
        ((0, 80, 80), (10, 255, 255)),
        ((170, 80, 80), (179, 255, 255)),
    ],
    "green_ranges": [
        ((35, 60, 70), (90, 255, 255)),
    ],
}


def detect_light(
    frame: np.ndarray,
    profile: str | None = None,
    light_config: dict[str, Any] | None = None,
    return_evidence: bool = False,
) -> str | None | tuple[str | None, LightEvidence]:
    """
    Detect the dominant traffic light color in a frame.

    Args:
        frame: BGR image from OpenCV.
        profile: Optional CONFIG profile name, e.g. "day" or "night".
        light_config: Optional direct override for light thresholds.
        return_evidence: When True, return (color, LightEvidence).

    Returns:
        "RED", "GREEN", or None. With return_evidence=True, returns a tuple.
    """

    if frame is None or frame.size == 0:
        evidence = LightEvidence(None, 0, 0, 0.0, 0.0, (0, 0, 0, 0))
        return (None, evidence) if return_evidence else None

    cfg = _load_light_config(profile, light_config)
    roi_img, roi_box = _crop_roi(frame, cfg["roi"])

    blur_size = _odd_kernel_size(cfg.get("blur_size", 5))
    if blur_size > 1:
        roi_img = cv2.GaussianBlur(roi_img, (blur_size, blur_size), 0)

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    red_mask = _build_mask(hsv, cfg["red_ranges"], cfg.get("kernel_size", 5))
    green_mask = _build_mask(hsv, cfg["green_ranges"], cfg.get("kernel_size", 5))

    red_area = _largest_contour_area(red_mask)
    green_area = _largest_contour_area(green_mask)
    roi_area = max(1, roi_img.shape[0] * roi_img.shape[1])
    red_ratio = red_area / roi_area
    green_ratio = green_area / roi_area

    color = _choose_color(
        red_area=red_area,
        green_area=green_area,
        red_ratio=red_ratio,
        green_ratio=green_ratio,
        min_area=int(cfg["min_area"]),
        min_ratio=float(cfg["min_ratio"]),
        winner_margin=float(cfg["winner_margin"]),
    )

    evidence = LightEvidence(color, red_area, green_area, red_ratio, green_ratio, roi_box)
    return (color, evidence) if return_evidence else color


def _load_light_config(
    profile: str | None,
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = DEFAULT_LIGHT_CONFIG.copy()

    profile_name = profile or CONFIG.get("default_profile")
    profile_cfg = CONFIG.get("profiles", {}).get(profile_name, {})
    cfg.update(profile_cfg.get("light", {}))

    if override:
        cfg.update(override)

    return cfg


def _crop_roi(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    if not roi:
        return frame, (0, 0, width, height)

    x, y, w, h = roi
    if all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in roi):
        x, y, w, h = int(x * width), int(y * height), int(w * width), int(h * height)
    else:
        x, y, w, h = int(x), int(y), int(w), int(h)

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return frame[y : y + h, x : x + w], (x, y, w, h)


def _build_mask(
    hsv: np.ndarray,
    ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    kernel_size: int,
) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

    kernel_size = _odd_kernel_size(kernel_size)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _largest_contour_area(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    return int(max(cv2.contourArea(contour) for contour in contours))


def _choose_color(
    red_area: int,
    green_area: int,
    red_ratio: float,
    green_ratio: float,
    min_area: int,
    min_ratio: float,
    winner_margin: float,
) -> str | None:
    red_valid = red_area >= min_area and red_ratio >= min_ratio
    green_valid = green_area >= min_area and green_ratio >= min_ratio

    if red_valid and red_area >= green_area * winner_margin:
        return "RED"
    if green_valid and green_area >= red_area * winner_margin:
        return "GREEN"
    return None


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1
