import unittest

from detection_stability import (
    DetectionStabilityConfig,
    DetectionStabilizer,
)


class DetectionStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stabilizer = DetectionStabilizer()

    def test_confidence_ema_reduces_fluctuation(self) -> None:
        first = self.stabilizer.update(
            [(7, "person", 0.8, (0, 0, 40, 80))], timestamp=1.0
        )[0]
        second = self.stabilizer.update(
            [(7, "person", 0.4, (0, 0, 40, 80))], timestamp=1.2
        )[0]
        self.assertEqual(first.smoothed_confidence, 0.8)
        self.assertAlmostEqual(second.smoothed_confidence, 0.7)
        self.assertEqual(second.raw_confidence, 0.4)

    def test_small_confidence_changes_remain_stable(self) -> None:
        self.stabilizer.update([(3, "car", 0.8, (0, 0, 60, 40))], timestamp=1.0)
        result = self.stabilizer.update(
            [(3, "car", 0.82, (0, 0, 60, 40))], timestamp=1.2
        )[0]
        self.assertAlmostEqual(result.smoothed_confidence, 0.805)

    def test_each_track_has_separate_confidence_history(self) -> None:
        first, second = self.stabilizer.update(
            [
                (10, "person", 0.9, (0, 0, 40, 80)),
                (11, "person", 0.3, (45, 0, 85, 80)),
            ],
            timestamp=1.0,
        )
        self.stabilizer.update(
            [
                (10, "person", 0.5, (0, 0, 40, 80)),
                (11, "person", 0.7, (45, 0, 85, 80)),
            ],
            timestamp=1.2,
        )
        self.assertAlmostEqual(self.stabilizer._tracks[10].confidence, 0.8)
        self.assertAlmostEqual(self.stabilizer._tracks[11].confidence, 0.4)
        self.assertEqual((first.track_id, second.track_id), (10, 11))

    def test_missing_detections_are_not_drawn_and_reacquisition_keeps_ema(self) -> None:
        self.stabilizer.update([(1, "person", 0.8, (0, 0, 40, 80))], timestamp=1.0)
        self.assertEqual(self.stabilizer.update([], timestamp=1.2), [])
        reacquired = self.stabilizer.update(
            [(1, "person", 0.4, (2, 0, 42, 80))], timestamp=1.4
        )[0]
        self.assertAlmostEqual(reacquired.smoothed_confidence, 0.7)
        self.assertEqual(reacquired.raw_box, (2.0, 0.0, 42.0, 80.0))

    def test_track_expires_after_missing_frame_limit(self) -> None:
        config = DetectionStabilityConfig(max_missing_frames=2)
        stabilizer = DetectionStabilizer(config)
        stabilizer.update([(4, "bicycle", 0.7, (0, 0, 30, 30))], timestamp=1.0)
        stabilizer.update([], timestamp=1.1)
        stabilizer.update([], timestamp=1.2)
        self.assertIn(4, stabilizer._tracks)
        stabilizer.update([], timestamp=1.3)
        self.assertNotIn(4, stabilizer._tracks)

    def test_track_expires_after_time_limit(self) -> None:
        config = DetectionStabilityConfig(max_age_seconds=1.0)
        stabilizer = DetectionStabilizer(config)
        stabilizer.update([(4, "bicycle", 0.7, (0, 0, 30, 30))], timestamp=1.0)
        stabilizer.update([], timestamp=2.1)
        self.assertNotIn(4, stabilizer._tracks)

    def test_tracker_box_coordinates_are_not_smoothed_a_second_time(self) -> None:
        first = self.stabilizer.update(
            [(2, "car", 0.8, (0, 0, 40, 40))], timestamp=1.0
        )[0]
        second = self.stabilizer.update(
            [(2, "car", 0.8, (40, 0, 80, 40))], timestamp=1.2
        )[0]
        self.assertEqual(first.display_box, first.raw_box)
        self.assertEqual(second.display_box, second.raw_box)
        self.assertEqual(second.display_box, (40.0, 0.0, 80.0, 40.0))

    def test_idless_detections_do_not_share_smoothing_state(self) -> None:
        first = self.stabilizer.update(
            [(None, "person", 0.8, (0, 0, 40, 80))], timestamp=1.0
        )[0]
        second = self.stabilizer.update(
            [(None, "person", 0.4, (0, 0, 40, 80))], timestamp=1.2
        )[0]
        self.assertEqual(first.smoothed_confidence, 0.8)
        self.assertEqual(second.smoothed_confidence, 0.4)
        self.assertEqual(self.stabilizer._tracks, {})

    def test_track_storage_is_bounded(self) -> None:
        stabilizer = DetectionStabilizer(DetectionStabilityConfig(max_tracks=2))
        for track_id in (1, 2, 3):
            stabilizer.update(
                [(track_id, "person", 0.8, (0, 0, 20, 40))],
                timestamp=float(track_id),
            )
        self.assertLessEqual(len(stabilizer._tracks), 2)


if __name__ == "__main__":
    unittest.main()
