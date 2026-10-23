from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrackedObject:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    observations: int = 1
    confidence_history: list[float] = field(default_factory=list)
    box_history: list[tuple[float, float, float, float]] = field(default_factory=list)
    status: str = "ACTIVE"
    quality_history: list[str] = field(default_factory=list)
    motion_history: list[str] = field(default_factory=list)
    missing_frames: int = 0

    def update(
        self,
        confidence: float,
        box: tuple[float, float, float, float],
        *,
        timestamp: float,
        quality_status: str | None = None,
        motion_status: str | None = None,
    ) -> None:
        self.confidence_history.append(float(confidence))
        self.box_history.append(box)
        self.last_seen = timestamp
        self.observations += 1
        if quality_status:
            self.quality_history.append(quality_status)
        if motion_status:
            self.motion_history.append(motion_status)
        self.status = "ACTIVE"
        self.missing_frames = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observations": self.observations,
            "confidence_history": self.confidence_history,
            "box_history": self.box_history,
            "status": self.status,
            "quality_history": self.quality_history,
            "motion_history": self.motion_history,
        }


@dataclass
class VisualMemoryConfig:
    max_history: int = 20
    max_events: int = 30
    max_tracks: int = 128
    temp_missing_limit: int = 3
    lost_after: int = 8
    iou_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.max_history < 1:
            raise ValueError("max_history must be positive.")
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be positive.")


class VisualMemory:
    def __init__(self, config: VisualMemoryConfig | None = None) -> None:
        self.config = config or VisualMemoryConfig()
        self.tracks: dict[int, TrackedObject] = {}
        self.events: list[dict[str, Any]] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.events.clear()
        self._next_id = 1

    def _record_event(self, event_type: str, *, track_id: int | None = None, label: str | None = None, details: str | None = None) -> None:
        entry = {
            "time": time.time(),
            "event": event_type,
            "track_id": track_id,
            "label": label,
            "details": details,
        }
        self.events.append(entry)
        del self.events[:-self.config.max_events]

    def _iou(self, box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def update(
        self,
        detections: list[Any],
        *,
        timestamp: float | None = None,
        image_quality: Any | None = None,
        motion: Any | None = None,
    ) -> dict[str, Any]:
        ts = time.time() if timestamp is None else float(timestamp)
        active_track_ids = set()

        for detection in detections:
            label = getattr(detection, "name", str(detection))
            confidence = float(getattr(detection, "confidence", 0.0))
            box = (
                float(getattr(detection, "x1", 0.0)),
                float(getattr(detection, "y1", 0.0)),
                float(getattr(detection, "x2", 0.0)),
                float(getattr(detection, "y2", 0.0)),
            )
            matched_track_id = None
            matched_iou = 0.0
            for track_id, track in self.tracks.items():
                if track.label != label:
                    continue
                if track.status == "LOST":
                    candidate_threshold = max(self.config.iou_threshold * 0.5, 0.12)
                else:
                    candidate_threshold = self.config.iou_threshold
                score = self._iou(track.box_history[-1], box)
                if score > matched_iou and score >= candidate_threshold:
                    matched_iou = score
                    matched_track_id = track_id

            if matched_track_id is not None:
                track = self.tracks[matched_track_id]
                if track.status in {"TEMPORARILY MISSING", "LOST"}:
                    self._record_event(
                        "object reacquired",
                        track_id=matched_track_id,
                        label=track.label,
                        details=f"Matched a previously lost or missing track at {ts}.",
                    )
                track.update(
                    confidence,
                    box,
                    timestamp=ts,
                    quality_status=getattr(image_quality, "status", None),
                    motion_status=getattr(motion, "status", None),
                )
                track.status = "ACTIVE"
                track.missing_frames = 0
                active_track_ids.add(matched_track_id)
                if len(track.confidence_history) >= 2:
                    self._record_event(
                        "object observed repeatedly",
                        track_id=matched_track_id,
                        label=track.label,
                        details=f"Observed {track.observations} times.",
                    )
            else:
                new_track = TrackedObject(
                    track_id=self._next_id,
                    label=label,
                    first_seen=ts,
                    last_seen=ts,
                    observations=1,
                    confidence_history=[confidence],
                    box_history=[box],
                    status="ACTIVE",
                    quality_history=[getattr(image_quality, "status", "UNKNOWN")],
                    motion_history=[getattr(motion, "status", "UNKNOWN")],
                )
                self._next_id += 1
                self.tracks[new_track.track_id] = new_track
                active_track_ids.add(new_track.track_id)
                self._record_event(
                    "new object detected",
                    track_id=new_track.track_id,
                    label=new_track.label,
                    details=f"First seen at {ts}.",
                )

        for track_id, track in list(self.tracks.items()):
            if track_id in active_track_ids:
                continue
            track.missing_frames += 1
            if track.missing_frames >= self.config.temp_missing_limit and track.status != "TEMPORARILY MISSING":
                track.status = "TEMPORARILY MISSING"
                self._record_event(
                    "object temporarily missing",
                    track_id=track_id,
                    label=track.label,
                    details=f"Missing for {track.missing_frames} frame(s).",
                )
            if track.missing_frames >= self.config.lost_after:
                track.status = "LOST"
                self._record_event(
                    "object track lost",
                    track_id=track_id,
                    label=track.label,
                    details=f"Track exceeded the loss threshold after {track.missing_frames} missed frames.",
                )

        for track_id, track in list(self.tracks.items()):
            if len(track.box_history) > self.config.max_history:
                track.box_history = track.box_history[-self.config.max_history:]
            if len(track.confidence_history) > self.config.max_history:
                track.confidence_history = track.confidence_history[-self.config.max_history:]
            if len(track.quality_history) > self.config.max_history:
                track.quality_history = track.quality_history[-self.config.max_history:]
            if len(track.motion_history) > self.config.max_history:
                track.motion_history = track.motion_history[-self.config.max_history:]

        eviction_priority = {
            "LOST": 0,
            "TEMPORARILY MISSING": 1,
            "ACTIVE": 2,
        }
        while len(self.tracks) > self.config.max_tracks:
            track_id = min(
                self.tracks,
                key=lambda candidate_id: (
                    eviction_priority.get(self.tracks[candidate_id].status, 2),
                    self.tracks[candidate_id].last_seen,
                    candidate_id,
                ),
            )
            del self.tracks[track_id]

        return self.as_dict()

    def register_detections(
        self,
        detections: list[Any],
        *,
        timestamp: float | None = None,
        image_quality: Any | None = None,
        motion: Any | None = None,
    ) -> dict[str, Any]:
        return self.update(detections, timestamp=timestamp, image_quality=image_quality, motion=motion)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tracks": [track.to_dict() for track in self.tracks.values()],
            "events": list(self.events),
            "active_tracks": [track.to_dict() for track in self.tracks.values() if track.status == "ACTIVE"],
        }

    def export_json(self, path: str | Path) -> None:
        payload = self.as_dict()
        Path(path).write_text(json.dumps(payload, indent=2, default=str))

    def export_csv(self, path: str | Path) -> None:
        fieldnames = [
            "track_id",
            "label",
            "first_seen",
            "last_seen",
            "observations",
            "status",
        ]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for track in self.tracks.values():
                writer.writerow({
                    "track_id": track.track_id,
                    "label": track.label,
                    "first_seen": track.first_seen,
                    "last_seen": track.last_seen,
                    "observations": track.observations,
                    "status": track.status,
                })
