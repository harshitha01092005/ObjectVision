from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQualityConfig:
    blur_good_threshold: float = 80.0
    blur_degraded_threshold: float = 20.0
    brightness_low_threshold: float = 40.0
    brightness_degraded_threshold: float = 75.0
    contrast_low_threshold: float = 10.0
    contrast_degraded_threshold: float = 20.0


@dataclass(frozen=True)
class ImageQualityResult:
    blur_score: float
    brightness: float
    contrast: float
    status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "blur_score": float(self.blur_score),
            "brightness": float(self.brightness),
            "contrast": float(self.contrast),
            "status": self.status,
            "warnings": list(self.warnings),
        }


def _classify_quality(
    blur_score: float,
    brightness: float,
    contrast: float,
    config: ImageQualityConfig,
) -> str:
    if (
        blur_score <= config.blur_degraded_threshold
        or brightness <= config.brightness_low_threshold
        or contrast <= config.contrast_low_threshold
    ):
        return "POOR"
    if (
        blur_score <= config.blur_good_threshold
        or brightness <= config.brightness_degraded_threshold
        or contrast <= config.contrast_degraded_threshold
    ):
        return "DEGRADED"
    return "GOOD"


def analyze_image_quality(image: np.ndarray | None, config: ImageQualityConfig | None = None) -> ImageQualityResult:
    cfg = config or ImageQualityConfig()
    warnings: list[str] = []
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        return ImageQualityResult(
            blur_score=0.0,
            brightness=0.0,
            contrast=0.0,
            status="POOR",
            warnings=("Image is unavailable; quality analysis is not possible.",),
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    status = _classify_quality(blur_score, brightness, contrast, cfg)

    if blur_score < cfg.blur_good_threshold:
        warnings.append("Image is blurry; small-object detection may be less reliable.")
    if brightness < cfg.brightness_degraded_threshold:
        warnings.append("Image brightness is too low.")
    if contrast < cfg.contrast_degraded_threshold:
        warnings.append("Image contrast is low.")
    if status == "POOR":
        warnings.append("Image quality is poor; inspect detections carefully.")

    if not warnings:
        warnings.append("Image quality is acceptable for the current scene.")

    return ImageQualityResult(
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        status=status,
        warnings=tuple(dict.fromkeys(warnings)),
    )
