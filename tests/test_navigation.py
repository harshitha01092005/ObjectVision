import math
import unittest

from navigation import (
    GO_LEFT,
    GO_RIGHT,
    MOVE_FORWARD,
    STOP,
    UNCERTAIN,
    Detection,
    NavigationEngine,
)


class NavigationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = NavigationEngine()
        self.now = 100.0

    def evaluate(self, detections: list[Detection], *, frame_time: float | None = None):
        self.now += 0.1
        return self.engine.evaluate(
            detections,
            300,
            300,
            self.now if frame_time is None else frame_time,
            now=self.now,
        )

    @staticmethod
    def box(
        name: str,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Detection:
        return Detection(name, confidence, x1, y1, x2, y2)

    def test_no_relevant_obstacles_suggests_cautious_forward_movement(self) -> None:
        result = self.evaluate([])
        self.assertEqual(result.recommendation, MOVE_FORWARD)
        self.assertIn("does not prove", result.explanation)

    def test_high_apparent_proximity_in_center_prioritizes_stop(self) -> None:
        result = self.evaluate([self.box("person", 0.95, 110, 30, 190, 270)])
        self.assertEqual(result.recommendation, STOP)
        self.assertTrue(result.regions["CENTER"].obstructed)

    def test_center_blocked_and_left_clear_suggests_left_after_confirmation(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        self.assertEqual(self.evaluate(detections).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(detections).recommendation, GO_LEFT)

    def test_center_blocked_and_right_clear_suggests_right_after_confirmation(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 5, 20, 75, 140),
        ]
        self.assertEqual(self.evaluate(detections).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(detections).recommendation, GO_RIGHT)

    def test_both_sides_obstructed_suggests_stop(self) -> None:
        result = self.evaluate(
            [
                self.box("chair", 0.9, 5, 10, 95, 140),
                self.box("bench", 0.9, 205, 10, 295, 140),
            ]
        )
        self.assertEqual(result.recommendation, STOP)

    def test_similar_side_obstruction_does_not_choose_arbitrarily(self) -> None:
        result = self.evaluate(
            [
                self.box("person", 0.9, 110, 30, 190, 150),
                self.box("chair", 0.6, 5, 30, 25, 55),
                self.box("chair", 0.6, 275, 30, 295, 55),
            ]
        )
        self.assertEqual(result.recommendation, UNCERTAIN)

    def test_low_confidence_only_detections_are_uncertain(self) -> None:
        result = self.evaluate([self.box("person", 0.2, 110, 30, 190, 150)])
        self.assertEqual(result.recommendation, UNCERTAIN)
        self.assertFalse(result.valid_evidence)

    def test_empty_detection_list_is_a_valid_empty_frame(self) -> None:
        result = self.evaluate([])
        self.assertTrue(result.frame_fresh)
        self.assertTrue(result.valid_evidence)

    def test_boundary_crossing_detection_occupies_each_overlapped_region(self) -> None:
        result = self.evaluate([self.box("person", 0.9, 90, 30, 150, 120)])
        self.assertEqual(len(result.detections), 1)
        self.assertIn("LEFT", result.detections[0].regions)
        self.assertIn("CENTER", result.detections[0].regions)
        self.assertTrue(result.regions["LEFT"].objects)
        self.assertTrue(result.regions["CENTER"].objects)

    def test_partially_out_of_frame_detection_is_clipped_to_left_region(self) -> None:
        result = self.evaluate([self.box("person", 0.9, -10, 100, 30, 130)])

        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].box, (0, 100, 30, 130))
        self.assertEqual(result.detections[0].regions, ("LEFT",))
        self.assertEqual(result.regions["LEFT"].objects, ("person",))
        self.assertFalse(result.regions["CENTER"].objects)
        self.assertFalse(result.regions["RIGHT"].objects)

    def test_missing_or_stale_frames_are_uncertain(self) -> None:
        missing = self.engine.evaluate([], None, None, None, now=self.now)
        stale = self.evaluate([], frame_time=self.now - 10)
        self.assertEqual(missing.recommendation, UNCERTAIN)
        self.assertFalse(missing.frame_fresh)
        self.assertEqual(stale.recommendation, UNCERTAIN)
        self.assertFalse(stale.frame_fresh)

    def test_invalid_frame_dimensions_are_uncertain(self) -> None:
        result = self.engine.evaluate([], 0, 300, self.now, now=self.now)
        self.assertEqual(result.recommendation, UNCERTAIN)
        self.assertFalse(result.frame_fresh)

    def test_rapid_direction_change_is_smoothed(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        right_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 5, 20, 75, 140),
        ]
        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)
        self.assertEqual(self.evaluate(right_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(right_clear).recommendation, GO_RIGHT)

    def test_recent_bounding_box_growth_increases_apparent_proximity(self) -> None:
        first = self.evaluate([self.box("person", 0.9, 120, 100, 180, 140)])
        second = self.evaluate([self.box("person", 0.9, 110, 90, 190, 150)])
        self.assertEqual(first.detections[0].proximity, "LOW")
        self.assertEqual(second.detections[0].proximity, "HIGH")
        self.assertGreater(second.detections[0].growth_ratio or 0, 1.8)

    def test_invalid_bounding_boxes_are_ignored_and_make_evidence_insufficient(self) -> None:
        result = self.evaluate(
            [self.box("person", 0.9, 140, 80, 120, 100)]
        )
        self.assertEqual(result.recommendation, UNCERTAIN)
        self.assertFalse(result.valid_evidence)

    def test_non_finite_box_coordinate_is_rejected(self) -> None:
        result = self.evaluate(
            [self.box("person", 0.9, math.nan, 20, 200, 200)]
        )
        self.assertEqual(result.recommendation, UNCERTAIN)
        self.assertFalse(result.valid_evidence)

    def test_result_never_invents_physical_distance(self) -> None:
        result = self.evaluate([self.box("person", 0.9, 110, 30, 190, 270)])
        serialized = result.to_dict()
        self.assertNotIn("distance", serialized)
        self.assertNotRegex(str(serialized), r"\b\d+(?:\.\d+)?\s*m\b")


if __name__ == "__main__":
    unittest.main()
