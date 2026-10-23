from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DetectionStabilityConfig:
    confidence_alpha: float = 0.25
    max_missing_frames: int = 8
    max_age_seconds: float = 3.0
    max_tracks: int = 128
    tracker_config: str = "bytetrack.yaml"

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_alpha) or not 0 < self.confidence_alpha <= 1:
            raise ValueError("confidence_alpha must be in (0, 1].")
        if (
            self.max_missing_frames < 0
            or not math.isfinite(self.max_age_seconds)
            or self.max_age_seconds <= 0
        ):
            raise ValueError("Track-retention limits must be positive.")
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be positive.")
        if not self.tracker_config:
            raise ValueError("tracker_config must not be empty.")


@dataclass(frozen=True)
class StableDetection:
    track_id: int | None
    label: str
    raw_confidence: float
    smoothed_confidence: float
    raw_box: tuple[float, float, float, float]
    display_box: tuple[float, float, float, float]


@dataclass
class _TrackSmoothingState:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    last_seen: float
    missing_frames: int = 0


class DetectionStabilizer:
    """Smooth current tracked detections without generating predicted detections."""

    def __init__(self, config: DetectionStabilityConfig | None = None) -> None:
        self.config = config or DetectionStabilityConfig()
        self._tracks: dict[int, _TrackSmoothingState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        detections: Iterable[
            tuple[
                int | None,
                str,
                float,
                tuple[float, float, float, float],
            ]
        ],
        *,
        timestamp: float,
    ) -> list[StableDetection]:
        current = list(detections)
        visible_track_ids = {
            track_id for track_id, _, _, _ in current if track_id is not None
        }
        for track_id, state in list(self._tracks.items()):
            if track_id not in visible_track_ids:
                state.missing_frames += 1
                if (
                    state.missing_frames > self.config.max_missing_frames
                    or timestamp - state.last_seen > self.config.max_age_seconds
                ):
                    del self._tracks[track_id]

        stabilized: list[StableDetection] = []
        for track_id, label, raw_confidence, raw_box in current:
            box = tuple(float(value) for value in raw_box)
            confidence = float(raw_confidence)
            if (
                track_id is None
                or not math.isfinite(confidence)
                or not all(math.isfinite(value) for value in box)
            ):
                stabilized.append(
                    StableDetection(
                        track_id,
                        label,
                        confidence,
                        confidence,
                        box,
                        box,
                    )
                )
                continue

            state = self._tracks.get(track_id)
            if state is None or state.label != label:
                state = _TrackSmoothingState(label, confidence, box, timestamp)
                self._tracks[track_id] = state
            else:
                state.confidence = (
                    self.config.confidence_alpha * confidence
                    + (1 - self.config.confidence_alpha) * state.confidence
                )
                state.box = box
                state.last_seen = timestamp
                state.missing_frames = 0
            stabilized.append(
                StableDetection(
                    track_id,
                    label,
                    confidence,
                    state.confidence,
                    box,
                    state.box,
                )
            )

        if len(self._tracks) > self.config.max_tracks:
            oldest = sorted(
                self._tracks.items(), key=lambda item: item[1].last_seen
            )
            for track_id, _ in oldest[: len(self._tracks) - self.config.max_tracks]:
                del self._tracks[track_id]
        return stabilized
