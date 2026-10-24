import math
import unittest

from navigation import (
    GO_LEFT,
    GO_RIGHT,
    MOVE_FORWARD,
    STOP,
    UNCERTAIN,
    Detection,
    NavigationConfig,
    NavigationEngine,
)


class NavigationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = NavigationEngine()
        self.now = 100.0

    def evaluate(
        self,
        detections: list[Detection],
        *,
        frame_time: float | None = None,
        visual_memory_tracks: list[dict[str, object]] | None = None,
    ):
        self.now += 0.1
        return self.engine.evaluate(
            detections,
            300,
            300,
            self.now if frame_time is None else frame_time,
            now=self.now,
            visual_memory_tracks=visual_memory_tracks,
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

    @staticmethod
    def missing_track(
        box: tuple[object, ...],
        *,
        last_seen: object = 100.0,
        status: str = "TEMPORARILY MISSING",
    ) -> dict[str, object]:
        return {
            "label": "person",
            "status": status,
            "last_seen": last_seen,
            "box_history": [box],
        }

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

    def test_recent_missing_left_obstacle_downgrades_confirmed_go_left(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        tracks = [self.missing_track((10, 30, 70, 120), last_seen=self.now)]

        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            UNCERTAIN,
        )
        result = self.evaluate(detections, visual_memory_tracks=tracks)
        self.assertEqual(result.recommendation, UNCERTAIN)
        self.assertIn("recently missing", result.explanation)

    def test_recent_missing_right_obstacle_downgrades_confirmed_go_right(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 5, 20, 75, 140),
        ]
        tracks = [self.missing_track((230, 30, 290, 120), last_seen=self.now)]

        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            UNCERTAIN,
        )
        result = self.evaluate(detections, visual_memory_tracks=tracks)
        self.assertEqual(result.recommendation, UNCERTAIN)
        self.assertIn("recently missing", result.explanation)

    def test_missing_obstacle_outside_proposed_region_does_not_downgrade(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        tracks = [self.missing_track((140, 30, 180, 100), last_seen=self.now)]

        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            UNCERTAIN,
        )
        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            GO_LEFT,
        )

    def test_expired_or_lost_memory_tracks_do_not_downgrade_direction(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        tracks = [
            self.missing_track((10, 30, 70, 120), last_seen=self.now - 2.0),
            self.missing_track(
                (10, 30, 70, 120),
                last_seen=self.now,
                status="LOST",
            ),
        ]

        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            UNCERTAIN,
        )
        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            GO_LEFT,
        )

    def test_maximum_missing_track_age_is_inclusive(self) -> None:
        engine = NavigationEngine(
            NavigationConfig(max_missing_track_age_seconds=1.5)
        )
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        tracks = [self.missing_track((10, 30, 70, 120), last_seen=98.5)]
        results = [
            engine.evaluate(
                detections,
                300,
                300,
                100.0,
                now=100.0,
                visual_memory_tracks=tracks,
            )
            for _ in range(2)
        ]

        self.assertEqual(results[0].recommendation, UNCERTAIN)
        self.assertEqual(results[1].recommendation, UNCERTAIN)
        self.assertIn("recently missing", results[1].explanation)

    def test_invalid_memory_timestamps_and_boxes_are_ignored(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        tracks = [
            self.missing_track((10, 30, 70, 120), last_seen=math.nan),
            self.missing_track(("bad", 30, 70, 120)),
            self.missing_track((10, 30, 70, 120), last_seen=True),
            self.missing_track((10, 30, 10, 120)),
        ]

        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            UNCERTAIN,
        )
        self.assertEqual(
            self.evaluate(detections, visual_memory_tracks=tracks).recommendation,
            GO_LEFT,
        )

    def test_memory_caution_preserves_stop_and_uncertain_results(self) -> None:
        tracks = [self.missing_track((10, 30, 70, 120), last_seen=self.now)]
        stop = self.evaluate(
            [self.box("person", 0.95, 110, 30, 190, 270)],
            visual_memory_tracks=tracks,
        )
        self.assertEqual(stop.recommendation, STOP)

        uncertain = self.evaluate(
            [self.box("person", 0.2, 110, 30, 190, 150)],
            visual_memory_tracks=tracks,
        )
        self.assertEqual(uncertain.recommendation, UNCERTAIN)

    def test_memory_caution_cannot_create_a_direction_from_empty_detections(self) -> None:
        tracks = [self.missing_track((10, 30, 70, 120), last_seen=self.now)]

        result = self.evaluate([], visual_memory_tracks=tracks)

        self.assertEqual(result.recommendation, MOVE_FORWARD)
        self.assertNotIn(result.recommendation, {GO_LEFT, GO_RIGHT})

    def test_center_blocked_and_right_clear_suggests_right_after_confirmation(self) -> None:
        detections = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 5, 20, 75, 140),
        ]
        self.assertEqual(self.evaluate(detections).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(detections).recommendation, GO_RIGHT)

    def test_stale_frame_resets_confirmed_left_direction(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

        stale = self.evaluate([], frame_time=self.now - 10)
        self.assertEqual(stale.recommendation, UNCERTAIN)
        self.assertFalse(stale.frame_fresh)

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

    def test_stale_interruption_restarts_pending_direction_confirmation(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        stale = self.evaluate([], frame_time=self.now - 10)
        self.assertEqual(stale.recommendation, UNCERTAIN)
        self.assertFalse(stale.frame_fresh)

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

    def test_low_confidence_interruption_restarts_pending_direction_confirmation(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        insufficient = self.evaluate(
            [self.box("person", 0.2, 110, 30, 190, 150)]
        )
        self.assertEqual(insufficient.recommendation, UNCERTAIN)
        self.assertTrue(insufficient.frame_fresh)
        self.assertFalse(insufficient.valid_evidence)

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

    def test_fresh_empty_frame_interrupts_pending_direction_confirmation(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        empty = self.evaluate([])
        self.assertEqual(empty.recommendation, MOVE_FORWARD)
        self.assertTrue(empty.frame_fresh)
        self.assertTrue(empty.valid_evidence)

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

    def test_low_confidence_frame_resets_confirmed_left_direction(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

        insufficient = self.evaluate(
            [self.box("person", 0.2, 110, 30, 190, 150)]
        )
        self.assertEqual(insufficient.recommendation, UNCERTAIN)
        self.assertTrue(insufficient.frame_fresh)
        self.assertFalse(insufficient.valid_evidence)

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

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

    def test_partially_out_of_frame_detections_are_clipped_at_other_edges(self) -> None:
        cases = (
            ((120, -10, 150, 30), (120, 0, 150, 30), ("CENTER",)),
            ((280, 100, 310, 130), (280, 100, 300, 130), ("RIGHT",)),
            ((120, 280, 150, 310), (120, 280, 150, 300), ("CENTER",)),
        )
        for box, clipped_box, regions in cases:
            with self.subTest(box=box):
                result = self.evaluate([self.box("person", 0.9, *box)])

                self.assertEqual(len(result.detections), 1)
                self.assertEqual(result.detections[0].box, clipped_box)
                self.assertEqual(result.detections[0].regions, regions)
                for region in ("LEFT", "CENTER", "RIGHT"):
                    expected_objects = ("person",) if region in regions else ()
                    self.assertEqual(
                        result.regions[region].objects, expected_objects
                    )

    def test_detection_entirely_outside_frame_is_rejected(self) -> None:
        result = self.evaluate([self.box("person", 0.9, -30, 100, -1, 130)])

        self.assertEqual(result.detections, ())
        self.assertEqual(result.recommendation, UNCERTAIN)
        self.assertFalse(result.valid_evidence)

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

    def test_rapid_direction_alternation_never_emits_unconfirmed_direction(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        right_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 5, 20, 75, 140),
        ]
        sequence = (
            (left_clear, UNCERTAIN),
            (right_clear, UNCERTAIN),
            (left_clear, UNCERTAIN),
            (right_clear, UNCERTAIN),
            (left_clear, UNCERTAIN),
            (left_clear, GO_LEFT),
        )

        for index, (detections, expected) in enumerate(sequence):
            with self.subTest(index=index, expected=expected):
                result = self.evaluate(detections)
                self.assertEqual(result.recommendation, expected)
                if expected == UNCERTAIN:
                    self.assertNotIn(result.recommendation, {GO_LEFT, GO_RIGHT})

    def test_memory_caution_downgrade_requires_fresh_direction_confirmation(self) -> None:
        left_clear = [
            self.box("person", 0.9, 110, 30, 190, 150),
            self.box("chair", 0.9, 225, 20, 295, 140),
        ]
        missing_left_track = [
            self.missing_track((10, 30, 70, 120), last_seen=self.now)
        ]

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

        cautioned = self.evaluate(
            left_clear,
            visual_memory_tracks=missing_left_track,
        )
        self.assertEqual(cautioned.recommendation, UNCERTAIN)
        self.assertIn("recently missing", cautioned.explanation)

        self.assertEqual(self.evaluate(left_clear).recommendation, UNCERTAIN)
        self.assertEqual(self.evaluate(left_clear).recommendation, GO_LEFT)

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
