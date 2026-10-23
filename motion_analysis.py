from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    low_motion_threshold: float = 0.015
    moderate_motion_threshold: float = 0.05
    high_motion_threshold: float = 0.12


@dataclass(frozen=True)
class MotionAnalysisResult:
    status: str
    magnitude: float
    translation: tuple[float, float]
    rotation: float
    reliable: bool
    warning: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "magnitude": float(self.magnitude),
            "translation": [float(value) for value in self.translation],
            "rotation": float(self.rotation),
            "reliable": bool(self.reliable),
            "warning": self.warning,
        }


def estimate_camera_motion(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray | None,
    config: MotionConfig | None = None,
) -> MotionAnalysisResult:
    cfg = config or MotionConfig()
    if previous_frame is None or current_frame is None:
        return MotionAnalysisResult(
            "UNKNOWN",
            0.0,
            (0.0, 0.0),
            0.0,
            False,
            "Camera-motion estimation is unavailable.",
        )

    if (
        previous_frame.size == 0
        or current_frame.size == 0
        or previous_frame.ndim != 3
        or current_frame.ndim != 3
        or previous_frame.shape != current_frame.shape
    ):
        return MotionAnalysisResult(
            "UNKNOWN",
            0.0,
            (0.0, 0.0),
            0.0,
            False,
            "Motion estimation is unreliable for this frame.",
        )

    if np.array_equal(previous_frame, current_frame):
        return MotionAnalysisResult(
            "LOW",
            0.0,
            (0.0, 0.0),
            0.0,
            True,
            "Camera movement is low; tracking should remain stable.",
        )

    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8,
    )
    if corners is None or len(corners) < 5:
        return MotionAnalysisResult(
            "UNKNOWN",
            0.0,
            (0.0, 0.0),
            0.0,
            False,
            "Motion estimation is unreliable for this frame.",
        )

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return MotionAnalysisResult(
            "UNKNOWN",
            0.0,
            (0.0, 0.0),
            0.0,
            False,
            "Motion estimation is unreliable for this frame.",
        )

    valid = status.ravel() == 1
    if valid.sum() < 5:
        return MotionAnalysisResult(
            "UNKNOWN",
            0.0,
            (0.0, 0.0),
            0.0,
            False,
            "Motion estimation is unreliable for this frame.",
        )

    good_prev = corners[valid].reshape(-1, 2)
    good_next = next_points[valid].reshape(-1, 2)
    motion_vectors = good_next - good_prev
    dx = float(motion_vectors[:, 0].mean())
    dy = float(motion_vectors[:, 1].mean())
    magnitude = float(math.hypot(dx, dy) / max(1.0, max(previous_frame.shape[:2])))
    rotation = float(math.degrees(math.atan2(dy, dx)))

    if magnitude < cfg.low_motion_threshold:
        status_name = "LOW"
        warning = "Camera movement is low; tracking should remain stable."
    elif magnitude < cfg.moderate_motion_threshold:
        status_name = "MODERATE"
        warning = "Camera movement is moderate."
    elif magnitude < cfg.high_motion_threshold:
        status_name = "HIGH"
        warning = "Camera movement is high; tracking may be unstable."
    else:
        status_name = "HIGH"
        warning = "Camera movement is high; tracking may be unstable."

    return MotionAnalysisResult(
        status_name,
        magnitude,
        (dx, dy),
        rotation,
        True,
        warning,
    )
