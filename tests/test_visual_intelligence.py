import math
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from image_quality import analyze_image_quality
from motion_analysis import estimate_camera_motion
from operator_assistance import generate_operator_assistance
from visual_memory import VisualMemory, VisualMemoryConfig


class FakeDetection:
    def __init__(self, name, confidence, x1, y1, x2, y2):
        self.name = name
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class VisualIntelligenceTests(unittest.TestCase):
    @staticmethod
    def location_suggestion(box, *, confidence=0.9, **dimensions):
        result = generate_operator_assistance(
            [FakeDetection("person", confidence, *box)],
            camera_connected=False,
            **dimensions,
        )
        return next(
            (
                suggestion
                for suggestion in result.suggestions
                if suggestion.startswith(
                    "A small object has been detected in the "
                )
            ),
            None,
        )

    def test_operator_assistance_classifies_pixel_boxes_by_region(self):
        cases = (
            ((10, 10, 40, 40), "upper-left"),
            ((130, 10, 170, 40), "upper-center"),
            ((260, 10, 290, 40), "upper-right"),
        )
        for box, region in cases:
            with self.subTest(region=region):
                self.assertEqual(
                    self.location_suggestion(
                        box, image_width=300, image_height=300
                    ),
                    f"A small object has been detected in the {region} image region.",
                )

    def test_operator_assistance_classifies_boxes_at_horizontal_boundaries(self):
        cases = (
            ((95.5, 10, 101.5, 20), "upper-left"),
            ((96, 10, 102, 20), "upper-center"),
            ((194.5, 10, 200.5, 20), "upper-center"),
            ((195, 10, 201, 20), "upper-right"),
        )
        for box, region in cases:
            with self.subTest(box=box):
                self.assertEqual(
                    self.location_suggestion(
                        box, image_width=300, image_height=300
                    ),
                    f"A small object has been detected in the {region} image region.",
                )

    def test_operator_assistance_classifies_boxes_at_vertical_boundaries(self):
        cases = (
            ((130, 95.5, 170, 101.5), "upper-center"),
            ((130, 96, 170, 102), "center"),
            ((130, 194.5, 170, 200.5), "center"),
            ((130, 195, 170, 201), "lower-center"),
        )
        for box, region in cases:
            with self.subTest(box=box):
                self.assertEqual(
                    self.location_suggestion(
                        box, image_width=300, image_height=300
                    ),
                    f"A small object has been detected in the {region} image region.",
                )

    def test_operator_assistance_skips_invalid_boxes_but_keeps_other_assistance(self):
        location_suggestions = {
            f"A small object has been detected in the {region} image region."
            for region in (
                "upper-left",
                "upper-center",
                "upper-right",
                "lower-left",
                "lower-center",
                "lower-right",
            )
        }
        invalid_boxes = (
            (None, 10, 30, 40),
            ("not-a-coordinate", 10, 30, 40),
            (math.nan, 10, 30, 40),
            (20, 10, 20, 40),
        )
        for box in invalid_boxes:
            with self.subTest(box=box):
                result = generate_operator_assistance(
                    [FakeDetection("person", 0.9, *box)],
                    camera_connected=False,
                    image_width=300,
                    image_height=300,
                )
                self.assertTrue(
                    location_suggestions.isdisjoint(result.suggestions)
                )
                self.assertIn(
                    "Camera connection is unavailable.",
                    result.warnings,
                )

        result = generate_operator_assistance(
            [
                FakeDetection("person", 0.9, None, 10, 30, 40),
                FakeDetection("person", 0.9, 10, 10, 40, 40),
            ],
            camera_connected=False,
            image_width=300,
            image_height=300,
        )
        self.assertIn(
            "A small object has been detected in the upper-left image region.",
            result.suggestions,
        )
        self.assertIn("Camera connection is unavailable.", result.warnings)

    def test_operator_assistance_handles_malformed_confidence_values(self):
        for confidence in (None, "0.9", "not-a-confidence", math.nan, math.inf):
            with self.subTest(confidence=confidence):
                result = generate_operator_assistance(
                    [FakeDetection("person", confidence, 10, 10, 40, 40)],
                    camera_connected=False,
                    image_width=300,
                    image_height=300,
                )
                self.assertNotIn(
                    "A small object has been detected in the upper-left image region.",
                    result.suggestions,
                )
                self.assertIn(
                    "Camera connection is unavailable.",
                    result.warnings,
                )

    def test_operator_assistance_suppresses_location_without_valid_dimensions(self):
        box = (10, 10, 40, 40)
        invalid_dimensions = (
            {},
            {"image_width": 0, "image_height": 300},
            {"image_width": math.nan, "image_height": 300},
            {"image_width": "not-a-dimension", "image_height": 300},
        )
        for dimensions in invalid_dimensions:
            with self.subTest(dimensions=dimensions):
                self.assertIsNone(
                    self.location_suggestion(box, **dimensions)
                )

    def test_image_quality_detects_clear_and_blurred_frames(self):
        clear = np.full((256, 256, 3), 180, dtype=np.uint8)
        clear[40:120, 40:120] = 220
        clear[150:220, 150:220] = 90
        quality = analyze_image_quality(clear)
        self.assertEqual(quality.status, "GOOD")

        blurred = np.full((256, 256, 3), 180, dtype=np.uint8)
        blurred = cv2.GaussianBlur(blurred, (31, 31), 0)
        poor_quality = analyze_image_quality(blurred)
        self.assertIn(poor_quality.status, {"DEGRADED", "POOR"})

    def test_motion_analysis_reports_low_motion_for_identical_frames(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        for x in range(0, 256, 20):
            cv2.line(frame, (x, 0), (x, 255), (80, 80, 80), 1)
        for y in range(0, 256, 20):
            cv2.line(frame, (0, y), (255, y), (80, 80, 80), 1)
        same = frame.copy()
        low_motion = estimate_camera_motion(frame, same)
        self.assertEqual(low_motion.status, "LOW")

        shifted = np.roll(frame, 18, axis=1)
        moderate_motion = estimate_camera_motion(frame, shifted)
        self.assertIn(moderate_motion.status, {"HIGH", "MODERATE", "LOW"})

    def test_visual_memory_tracks_repeated_observations_and_lost_tracks(self):
        memory = VisualMemory()
        det = FakeDetection("person", 0.9, 10, 20, 80, 120)
        memory.update([det], timestamp=1.0)
        memory.update([det], timestamp=2.0)
        self.assertEqual(len(memory.tracks), 1)
        self.assertEqual(memory.tracks[1].status, "ACTIVE")

        for idx in range(1, 10):
            memory.update([], timestamp=float(idx + 2))
        self.assertIn(memory.tracks[1].status, {"TEMPORARILY MISSING", "LOST"})

    def test_visual_memory_matches_each_detection_to_at_most_one_track_per_frame(self):
        memory = VisualMemory()
        memory.update(
            [
                FakeDetection("person", 0.9, 0, 0, 100, 100),
                FakeDetection("person", 0.9, 30, 0, 130, 100),
            ],
            timestamp=1.0,
        )

        self.assertEqual(set(memory.tracks), {1, 2})
        self.assertEqual(
            [track.observations for track in memory.tracks.values()],
            [1, 1],
        )

        memory.update(
            [
                FakeDetection("person", 0.9, 10, 0, 110, 100),
                FakeDetection("person", 0.9, 30, 0, 130, 100),
            ],
            timestamp=2.0,
        )

        self.assertEqual(set(memory.tracks), {1, 2})
        self.assertEqual(
            [track.observations for track in memory.tracks.values()],
            [2, 2],
        )
        self.assertTrue(
            all(track.status == "ACTIVE" for track in memory.tracks.values())
        )

    def test_visual_memory_reacquires_a_missing_track(self):
        memory = VisualMemory()
        memory.update([FakeDetection("person", 0.9, 10, 20, 80, 120)], timestamp=1.0)
        for idx in range(1, 5):
            memory.update([], timestamp=float(idx + 1))
        memory.update([FakeDetection("person", 0.9, 11, 21, 81, 121)], timestamp=10.0)
        self.assertTrue(any(event["event"] == "object reacquired" for event in memory.events))

    def test_visual_memory_bounds_all_track_histories(self):
        memory = VisualMemory(VisualMemoryConfig(max_history=2))
        for index in range(4):
            memory.update(
                [
                    FakeDetection(
                        "person",
                        0.5 + index * 0.1,
                        10 + index,
                        20,
                        80 + index,
                        120,
                    )
                ],
                timestamp=float(index),
                image_quality=SimpleNamespace(status=f"quality-{index}"),
                motion=SimpleNamespace(status=f"motion-{index}"),
            )

        track = memory.tracks[1]
        self.assertEqual(len(track.box_history), 2)
        self.assertEqual(len(track.confidence_history), 2)
        self.assertEqual(len(track.quality_history), 2)
        self.assertEqual(len(track.motion_history), 2)
        self.assertEqual(track.quality_history, ["quality-2", "quality-3"])
        self.assertEqual(track.motion_history, ["motion-2", "motion-3"])

    def test_visual_memory_evicts_lost_then_temporarily_missing_tracks(self):
        memory = VisualMemory(VisualMemoryConfig(max_tracks=3))
        memory.update(
            [
                FakeDetection("person", 0.9, 0, 0, 20, 20),
                FakeDetection("car", 0.9, 30, 0, 50, 20),
                FakeDetection("dog", 0.9, 60, 0, 80, 20),
            ],
            timestamp=1.0,
        )
        memory.tracks[1].status = "LOST"
        memory.tracks[1].last_seen = 1.0
        memory.tracks[2].status = "TEMPORARILY MISSING"
        memory.tracks[2].last_seen = 2.0

        memory.update(
            [FakeDetection("dog", 0.9, 60, 0, 80, 20), FakeDetection("cat", 0.9, 90, 0, 110, 20)],
            timestamp=3.0,
        )
        self.assertEqual(set(memory.tracks), {2, 3, 4})

        memory.update(
            [FakeDetection("dog", 0.9, 60, 0, 80, 20), FakeDetection("cat", 0.9, 90, 0, 110, 20), FakeDetection("bird", 0.9, 120, 0, 140, 20)],
            timestamp=4.0,
        )
        self.assertEqual(set(memory.tracks), {3, 4, 5})

    def test_visual_memory_evicts_oldest_track_when_all_tracks_are_active(self):
        memory = VisualMemory(
            VisualMemoryConfig(max_tracks=2, temp_missing_limit=10)
        )
        memory.update([FakeDetection("person", 0.9, 0, 0, 20, 20)], timestamp=1.0)
        memory.update([FakeDetection("car", 0.9, 30, 0, 50, 20)], timestamp=2.0)
        memory.update([FakeDetection("dog", 0.9, 60, 0, 80, 20)], timestamp=3.0)
        self.assertTrue(all(track.status == "ACTIVE" for track in memory.tracks.values()))

        memory.update([FakeDetection("bird", 0.9, 90, 0, 110, 20)], timestamp=4.0)
        self.assertEqual(set(memory.tracks), {3, 4})
        self.assertTrue(all(track.status == "ACTIVE" for track in memory.tracks.values()))

    def test_visual_memory_bounds_event_history(self):
        memory = VisualMemory(VisualMemoryConfig(max_events=2))

        for index, label in enumerate(("person", "car", "dog"), start=1):
            memory.update(
                [FakeDetection(label, 0.9, index * 30, 0, index * 30 + 20, 20)],
                timestamp=float(index),
            )

        self.assertEqual(len(memory.events), 2)
        self.assertEqual(
            [event["label"] for event in memory.events],
            ["car", "dog"],
        )

    def test_operator_assistance_prioritizes_important_warnings(self):
        blurred = np.full((256, 256, 3), 180, dtype=np.uint8)
        blurred = cv2.GaussianBlur(blurred, (31, 31), 0)
        quality = analyze_image_quality(blurred)

        frame1 = np.zeros((256, 256, 3), dtype=np.uint8)
        for x in range(0, 256, 20):
            cv2.line(frame1, (x, 0), (x, 255), (80, 80, 80), 1)
        for y in range(0, 256, 20):
            cv2.line(frame1, (0, y), (255, y), (80, 80, 80), 1)
        frame2 = np.roll(frame1.copy(), 18, axis=1)
        motion = estimate_camera_motion(frame1, frame2)
        memory = VisualMemory()
        memory.tracks[1] = type("Track", (), {"status": "LOST", "confidence_history": [0.4, 0.5, 0.6]})()
        result = generate_operator_assistance(
            [FakeDetection("person", 0.2, 5, 5, 20, 40)],
            quality,
            motion,
            memory,
            frame_fresh=True,
            camera_connected=True,
        )
        warnings = " ".join(result.warnings)
        self.assertIn("Image is blurry", warnings)
        self.assertIn("Camera movement is", warnings)
        self.assertIn("track has been lost", warnings.lower())


if __name__ == "__main__":
    unittest.main()
