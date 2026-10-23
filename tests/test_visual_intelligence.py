import unittest

import cv2
import numpy as np

from image_quality import analyze_image_quality
from motion_analysis import estimate_camera_motion
from operator_assistance import generate_operator_assistance
from visual_memory import VisualMemory


class FakeDetection:
    def __init__(self, name, confidence, x1, y1, x2, y2):
        self.name = name
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class VisualIntelligenceTests(unittest.TestCase):
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

    def test_visual_memory_reacquires_a_missing_track(self):
        memory = VisualMemory()
        memory.update([FakeDetection("person", 0.9, 10, 20, 80, 120)], timestamp=1.0)
        for idx in range(1, 5):
            memory.update([], timestamp=float(idx + 1))
        memory.update([FakeDetection("person", 0.9, 11, 21, 81, 121)], timestamp=10.0)
        self.assertTrue(any(event["event"] == "object reacquired" for event in memory.events))

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
