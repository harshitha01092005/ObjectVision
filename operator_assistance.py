from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperatorAssistanceConfig:
    confidence_threshold: float = 0.35
    blur_threshold: float = 30.0
    low_brightness_threshold: float = 75.0
    low_contrast_threshold: float = 30.0
    moderate_motion_threshold: float = 0.05


@dataclass(frozen=True)
class OperatorAssistanceResult:
    suggestions: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": list(self.suggestions),
            "warnings": list(self.warnings),
            "summary": self.summary,
        }


def _region_label(box: tuple[float, float, float, float], width: float, height: float) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / max(width, 1.0)
    cy = (y1 + y2) / 2.0 / max(height, 1.0)
    if cy < 0.33:
        if cx < 0.33:
            return "upper-left"
        if cx < 0.66:
            return "upper-center"
        return "upper-right"
    if cy < 0.66:
        if cx < 0.33:
            return "center-left"
        if cx < 0.66:
            return "center"
        return "center-right"
    if cx < 0.33:
        return "lower-left"
    if cx < 0.66:
        return "lower-center"
    return "lower-right"


def generate_operator_assistance(
    detections: list[Any],
    image_quality: Any | None = None,
    motion: Any | None = None,
    visual_memory: Any | None = None,
    *,
    frame_fresh: bool = True,
    camera_connected: bool = True,
    config: OperatorAssistanceConfig | None = None,
) -> OperatorAssistanceResult:
    cfg = config or OperatorAssistanceConfig()
    warnings: list[str] = []
    suggestions: list[str] = []

    if not frame_fresh:
        warnings.append("Insufficient visual evidence; treat this observation as uncertain.")
    if not camera_connected:
        warnings.append("Camera connection is unavailable.")

    if image_quality is not None:
        if image_quality.status == "POOR":
            warnings.append("Image quality is poor; inspect detections carefully.")
        if image_quality.blur_score < cfg.blur_threshold:
            warnings.append("Image is blurry; small-object detection may be less reliable.")
        if image_quality.brightness < cfg.low_brightness_threshold:
            warnings.append("Image brightness is too low.")
        if image_quality.contrast < cfg.low_contrast_threshold:
            warnings.append("Image contrast is low.")

    if motion is not None and getattr(motion, "status", "UNKNOWN") == "HIGH":
        warnings.append("Camera movement is high; tracking may be unstable.")
    elif motion is not None and getattr(motion, "status", "UNKNOWN") == "MODERATE":
        warnings.append("Camera movement is moderate.")
    elif motion is not None and getattr(motion, "status", "UNKNOWN") == "UNKNOWN":
        warnings.append("Camera-motion estimation is unavailable.")

    if detections:
        low_confidence = [d for d in detections if getattr(d, "confidence", 0.0) < cfg.confidence_threshold]
        if low_confidence:
            warnings.append("Detection confidence is low; inspect this observation carefully.")

    if visual_memory is not None:
        tracks = getattr(visual_memory, "tracks", {})
        lost_tracks = [track for track in tracks.values() if getattr(track, "status", "ACTIVE") == "LOST"]
        if lost_tracks:
            warnings.append("An object track has been lost and may need to be reacquired.")
        active_tracks = [track for track in tracks.values() if getattr(track, "status", "ACTIVE") == "ACTIVE"]
        if active_tracks and any(len(track.confidence_history) >= 3 for track in active_tracks):
            suggestions.append("The same object has been observed consistently across multiple frames.")

    if detections:
        for detection in detections:
            box = (
                float(getattr(detection, "x1", 0.0)),
                float(getattr(detection, "y1", 0.0)),
                float(getattr(detection, "x2", 0.0)),
                float(getattr(detection, "y2", 0.0)),
            )
            if getattr(detection, "confidence", 0.0) >= cfg.confidence_threshold:
                region = _region_label(box, 1.0, 1.0)
                if region in {"upper-left", "upper-center", "upper-right", "lower-left", "lower-center", "lower-right"}:
                    suggestions.append(f"A small object has been detected in the {region} image region.")
                    break

    if not warnings and not suggestions:
        suggestions.append("The observation appears stable and the object should be reviewed in context.")

    summary = "; ".join(dict.fromkeys(warnings + suggestions))
    return OperatorAssistanceResult(
        suggestions=tuple(dict.fromkeys(suggestions)),
        warnings=tuple(dict.fromkeys(warnings)),
        summary=summary,
    )


def build_operator_assistance(
    detections: list[Any],
    image_quality: Any | None = None,
    motion: Any | None = None,
    visual_memory: Any | None = None,
    *,
    frame_fresh: bool = True,
    camera_connected: bool = True,
    config: OperatorAssistanceConfig | None = None,
) -> OperatorAssistanceResult:
    return generate_operator_assistance(
        detections,
        image_quality,
        motion,
        visual_memory,
        frame_fresh=frame_fresh,
        camera_connected=camera_connected,
        config=config,
    )
