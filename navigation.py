from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Iterable


REGIONS = ("LEFT", "CENTER", "RIGHT")
RELEVANT_CLASSES = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "bench",
        "chair",
        "potted plant",
        "dog",
        "cat",
        "suitcase",
        "backpack",
    }
)

STOP = "STOP — OBSTACLE VERY CLOSE"
GO_LEFT = "GO LEFT — LEFT PATH APPEARS CLEARER"
GO_RIGHT = "GO RIGHT — RIGHT PATH APPEARS CLEARER"
MOVE_FORWARD = "MOVE FORWARD CAUTIOUSLY — CENTER PATH APPEARS CLEAR"
SLOW_DOWN = "SLOW DOWN — POSSIBLE OBSTACLE AHEAD"
UNCERTAIN = "UNCERTAIN — CHECK SURROUNDINGS"


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class NavigationConfig:
    confidence_threshold: float = 0.35
    medium_area_fraction: float = 0.08
    high_area_fraction: float = 0.20
    medium_height_fraction: float = 0.35
    high_height_fraction: float = 0.65
    high_bottom_fraction: float = 0.94
    medium_growth_ratio: float = 1.35
    high_growth_ratio: float = 1.8
    blocked_region_score: float = 0.65
    side_clearance_margin: float = 0.25
    max_frame_age_seconds: float = 2.5
    relevant_classes: frozenset[str] = RELEVANT_CLASSES

    def __post_init__(self) -> None:
        fractions = (
            self.confidence_threshold,
            self.medium_area_fraction,
            self.high_area_fraction,
            self.medium_height_fraction,
            self.high_height_fraction,
            self.high_bottom_fraction,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
            raise ValueError("Navigation confidence and image-fraction thresholds must be between 0 and 1.")
        if (
            self.high_area_fraction <= self.medium_area_fraction
            or self.high_height_fraction <= self.medium_height_fraction
        ):
            raise ValueError("High apparent thresholds must exceed medium thresholds.")
        if (
            not math.isfinite(self.medium_growth_ratio)
            or self.medium_growth_ratio <= 1
            or not math.isfinite(self.high_growth_ratio)
            or self.high_growth_ratio <= self.medium_growth_ratio
        ):
            raise ValueError("Bounding-box growth thresholds must increase above 1.")
        if (
            not math.isfinite(self.blocked_region_score)
            or self.blocked_region_score < 0
            or not math.isfinite(self.side_clearance_margin)
            or self.side_clearance_margin < 0
            or not math.isfinite(self.max_frame_age_seconds)
            or self.max_frame_age_seconds <= 0
        ):
            raise ValueError("Navigation score and frame-age thresholds must be non-negative and finite.")
        if not all(isinstance(name, str) for name in self.relevant_classes):
            raise ValueError("Navigation class names must be strings.")


@dataclass(frozen=True)
class RegionStatus:
    obstructed: bool
    score: float
    objects: tuple[str, ...]


@dataclass(frozen=True)
class EvaluatedDetection:
    name: str
    confidence: float
    box: tuple[float, float, float, float]
    regions: tuple[str, ...]
    proximity: str
    area_fraction: float
    height_fraction: float
    growth_ratio: float | None


@dataclass(frozen=True)
class NavigationResult:
    recommendation: str
    explanation: str
    detections: tuple[EvaluatedDetection, ...]
    regions: dict[str, RegionStatus]
    frame_age_seconds: float | None
    frame_fresh: bool
    valid_evidence: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "recommendation": self.recommendation,
            "explanation": self.explanation,
            "detections": [
                {
                    **asdict(detection),
                    "box": list(detection.box),
                    "regions": list(detection.regions),
                }
                for detection in self.detections
            ],
            "regions": {name: asdict(status) for name, status in self.regions.items()},
            "frame_age_seconds": self.frame_age_seconds,
            "frame_fresh": self.frame_fresh,
            "valid_evidence": self.valid_evidence,
        }


class NavigationEngine:
    def __init__(self, config: NavigationConfig | None = None) -> None:
        self.config = config or NavigationConfig()
        self._previous_boxes: list[tuple[str, float, float, float, float]] = []
        self._previous_frame_time: float | None = None
        self._pending_direction: str | None = None
        self._pending_direction_frames = 0
        self._confirmed_direction: str | None = None

    def reset(self) -> None:
        self._previous_boxes.clear()
        self._previous_frame_time = None
        self._pending_direction = None
        self._pending_direction_frames = 0
        self._confirmed_direction = None

    def evaluate(
        self,
        detections: Iterable[Detection] | None,
        frame_width: int | None,
        frame_height: int | None,
        frame_timestamp: float | None,
        *,
        now: float | None = None,
    ) -> NavigationResult:
        current_time = time.time() if now is None else now
        frame_age = (
            None if frame_timestamp is None else current_time - frame_timestamp
        )
        empty_regions = {
            region: RegionStatus(False, 0.0, ()) for region in REGIONS
        }

        if (
            detections is None
            or frame_width is None
            or frame_height is None
            or frame_width <= 0
            or frame_height <= 0
            or frame_age is None
            or not math.isfinite(frame_age)
            or frame_age < -0.25
            or frame_age > self.config.max_frame_age_seconds
        ):
            self._clear_tracking()
            return NavigationResult(
                UNCERTAIN,
                "A current, valid camera frame is not available; check the camera connection.",
                (),
                empty_regions,
                frame_age,
                False,
                False,
            )

        if not self.config.relevant_classes:
            self._clear_tracking()
            return NavigationResult(
                UNCERTAIN,
                "No navigation object classes are enabled; choose relevant classes in the navigation controls.",
                (),
                empty_regions,
                frame_age,
                True,
                False,
            )

        valid_detections: list[Detection] = []
        invalid_count = 0
        low_confidence_count = 0
        for detection in detections:
            if not self._valid_detection(detection, frame_width, frame_height):
                invalid_count += 1
                continue
            if detection.name not in self.config.relevant_classes:
                continue
            if detection.confidence < self.config.confidence_threshold:
                low_confidence_count += 1
                continue
            valid_detections.append(detection)

        if self._previous_frame_time is None or current_time - self._previous_frame_time > 2:
            self._previous_boxes.clear()

        evaluated: list[EvaluatedDetection] = []
        current_boxes: list[tuple[str, float, float, float, float]] = []
        region_scores = {region: 0.0 for region in REGIONS}
        region_objects: dict[str, list[str]] = {region: [] for region in REGIONS}
        region_has_high: dict[str, bool] = {region: False for region in REGIONS}

        for detection in valid_detections:
            x1 = min(frame_width, max(0.0, detection.x1))
            y1 = min(frame_height, max(0.0, detection.y1))
            x2 = min(frame_width, max(0.0, detection.x2))
            y2 = min(frame_height, max(0.0, detection.y2))
            if x2 <= x1 or y2 <= y1:
                invalid_count += 1
                continue

            box_width = x2 - x1
            box_height = y2 - y1
            area_fraction = box_width * box_height / (frame_width * frame_height)
            height_fraction = box_height / frame_height
            center_x = (x1 + x2) / 2 / frame_width
            center_y = (y1 + y2) / 2 / frame_height
            growth_ratio = self._growth_ratio(
                detection.name, center_x, center_y, area_fraction
            )
            proximity = self._proximity(
                area_fraction,
                height_fraction,
                y2 / frame_height,
                growth_ratio,
            )

            region_overlap: list[tuple[str, float]] = []
            for index, region in enumerate(REGIONS):
                left = index / 3
                right = (index + 1) / 3
                overlap = max(
                    0.0,
                    min(x2 / frame_width, right) - max(x1 / frame_width, left),
                )
                if overlap > 0:
                    region_overlap.append((region, overlap / (box_width / frame_width)))

            regions = tuple(region for region, _ in region_overlap)
            if not regions:
                invalid_count += 1
                continue

            evaluated.append(
                EvaluatedDetection(
                    detection.name,
                    detection.confidence,
                    (x1, y1, x2, y2),
                    regions,
                    proximity,
                    area_fraction,
                    height_fraction,
                    growth_ratio,
                )
            )
            current_boxes.append(
                (detection.name, center_x, center_y, area_fraction)
            )
            weight = {"LOW": 0.2, "MEDIUM": 1.0, "HIGH": 2.0}[proximity]
            for region, overlap_fraction in region_overlap:
                region_scores[region] += (
                    detection.confidence * weight * min(1.0, overlap_fraction)
                )
                region_objects[region].append(detection.name)
                if proximity == "HIGH":
                    region_has_high[region] = True

        self._previous_boxes = current_boxes[:60]
        self._previous_frame_time = current_time
        regions_status = {
            region: RegionStatus(
                region_has_high[region]
                or region_scores[region] >= self.config.blocked_region_score,
                round(region_scores[region], 3),
                tuple(region_objects[region]),
            )
            for region in REGIONS
        }
        result = self._decide(
            evaluated,
            regions_status,
            invalid_count,
            low_confidence_count,
            frame_age,
        )
        return result

    def _decide(
        self,
        detections: list[EvaluatedDetection],
        regions: dict[str, RegionStatus],
        invalid_count: int,
        low_confidence_count: int,
        frame_age: float,
    ) -> NavigationResult:
        left = regions["LEFT"]
        center = regions["CENTER"]
        right = regions["RIGHT"]
        high_center = any(
            detection.proximity == "HIGH" and "CENTER" in detection.regions
            for detection in detections
        )

        if high_center:
            recommendation = STOP
            explanation = "A high-apparent-proximity relevant object overlaps the center region."
        elif left.obstructed and right.obstructed:
            recommendation = STOP
            explanation = "Relevant objects make both side regions appear obstructed."
        elif center.obstructed:
            difference = abs(left.score - right.score)
            if difference <= self.config.side_clearance_margin:
                recommendation = UNCERTAIN
                explanation = "The center appears obstructed, but left and right do not differ enough to prefer a direction."
            elif left.score < right.score:
                recommendation = GO_LEFT
                explanation = "The center appears obstructed and the left region has less detected obstruction."
            else:
                recommendation = GO_RIGHT
                explanation = "The center appears obstructed and the right region has less detected obstruction."
        elif not detections and (invalid_count or low_confidence_count):
            recommendation = UNCERTAIN
            explanation = "Available detections were invalid or below the configured confidence threshold."
        elif not detections:
            recommendation = MOVE_FORWARD
            explanation = "No relevant objects met the configured confidence threshold; this does not prove the path is safe."
        elif any(detection.proximity in {"MEDIUM", "HIGH"} for detection in detections):
            recommendation = SLOW_DOWN
            explanation = "A relevant object is visible at medium or high apparent proximity."
        else:
            recommendation = SLOW_DOWN
            explanation = "A relevant object is present; continue cautiously and check the surroundings."

        if recommendation in {GO_LEFT, GO_RIGHT}:
            recommendation, explanation = self._smooth_direction(
                recommendation, explanation
            )
        else:
            self._pending_direction = None
            self._pending_direction_frames = 0
            self._confirmed_direction = None

        if invalid_count:
            explanation += f" {invalid_count} invalid detection box(es) were ignored."
        return NavigationResult(
            recommendation,
            explanation,
            tuple(detections),
            regions,
            frame_age,
            True,
            bool(detections) or (invalid_count == 0 and low_confidence_count == 0),
        )

    def _smooth_direction(self, candidate: str, explanation: str) -> tuple[str, str]:
        if candidate == self._confirmed_direction:
            return candidate, explanation
        if candidate == self._pending_direction:
            self._pending_direction_frames += 1
        else:
            self._pending_direction = candidate
            self._pending_direction_frames = 1
        if self._pending_direction_frames >= 2:
            self._confirmed_direction = candidate
            self._pending_direction = None
            self._pending_direction_frames = 0
            return candidate, explanation
        return (
            UNCERTAIN,
            "A possible direction changed; waiting for a consistent frame before suggesting it.",
        )

    def _growth_ratio(
        self, name: str, center_x: float, center_y: float, area_fraction: float
    ) -> float | None:
        matches = [
            old
            for old in self._previous_boxes
            if old[0] == name
            and abs(old[1] - center_x) < 0.2
            and abs(old[2] - center_y) < 0.2
        ]
        if not matches:
            return None
        previous = min(matches, key=lambda old: abs(old[1] - center_x) + abs(old[2] - center_y))
        if previous[3] <= 0:
            return None
        return area_fraction / previous[3]

    def _proximity(
        self,
        area_fraction: float,
        height_fraction: float,
        bottom_fraction: float,
        growth_ratio: float | None,
    ) -> str:
        if (
            area_fraction >= self.config.high_area_fraction
            or height_fraction >= self.config.high_height_fraction
            or (
                bottom_fraction >= self.config.high_bottom_fraction
                and area_fraction >= self.config.medium_area_fraction * 1.5
            )
            or (
                growth_ratio is not None
                and growth_ratio >= self.config.high_growth_ratio
            )
        ):
            return "HIGH"
        if (
            area_fraction >= self.config.medium_area_fraction
            or height_fraction >= self.config.medium_height_fraction
            or (
                growth_ratio is not None
                and growth_ratio >= self.config.medium_growth_ratio
            )
        ):
            return "MEDIUM"
        return "LOW"

    def _valid_detection(
        self, detection: Detection, frame_width: int, frame_height: int
    ) -> bool:
        values = (
            detection.confidence,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            return False
        if (
            not isinstance(detection.name, str)
            or not detection.name
            or detection.confidence < 0
            or detection.confidence > 1
            or detection.x2 <= detection.x1
            or detection.y2 <= detection.y1
        ):
            return False
        return (
            detection.x2 > 0
            and detection.y2 > 0
            and detection.x1 < frame_width
            and detection.y1 < frame_height
        )

    def _clear_tracking(self) -> None:
        self._previous_boxes.clear()
        self._previous_frame_time = None
        self._pending_direction = None
        self._pending_direction_frames = 0
        self._confirmed_direction = None
