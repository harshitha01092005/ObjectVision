from __future__ import annotations

import ast
import hashlib
import math
import tempfile
import threading
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import av
import numpy as np

import app
from benchmark_latency import build_real_stream_report


class Values:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeBoxes:
    def __init__(
        self,
        *,
        include_person: bool,
        confidence: float = 0.92,
        detections: list[tuple[float, list[float], float]] | None = None,
    ) -> None:
        values = detections or (
            [(confidence, [100, 20, 220, 260], 1)] if include_person else []
        )
        self.cls = Values([0 for _ in values])
        self.conf = Values([item[0] for item in values])
        self.xyxy = Values([item[1] for item in values])
        ids = [item[2] for item in values]
        self.id = Values(ids) if values and all(value is not None for value in ids) else None


class FakeResult:
    def __init__(
        self,
        image: np.ndarray,
        *,
        include_person: bool,
        confidence: float = 0.92,
        detections: list[tuple[float, list[float], float]] | None = None,
    ) -> None:
        self.image = image
        self.boxes = FakeBoxes(
            include_person=include_person,
            confidence=confidence,
            detections=detections,
        )


class FakeModel:
    names = {0: "person"}

    def __init__(
        self,
        *,
        include_person: bool,
        confidence: float = 0.92,
        detections: list[tuple[float, list[float], float]] | None = None,
    ) -> None:
        self.include_person = include_person
        self.confidence = confidence
        self.detections = detections
        self.calls = 0
        self.prediction_options: dict[str, object] = {}
        self.tracking_options: dict[str, object] = {}
        self.predictor = None

    def track(
        self,
        *,
        source: np.ndarray,
        conf: float,
        verbose: bool,
        device: str,
        persist: bool,
        tracker: str,
    ) -> list[FakeResult]:
        self.calls += 1
        self.prediction_options = {
            "conf": conf,
            "verbose": verbose,
            "device": device,
        }
        self.tracking_options = {"persist": persist, "tracker": tracker}
        return [
            FakeResult(
                source,
                include_person=self.include_person,
                confidence=self.confidence,
                detections=self.detections,
            )
        ]


class FakeFrame:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def to_ndarray(self, *, format: str) -> np.ndarray:
        if format != "bgr24":
            raise ValueError(f"Unexpected format: {format}")
        return self.image


class FakeTracker:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class StreamPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((300, 300, 3), dtype=np.uint8)
        self.owner = "streaming-unit-test"
        with app.STREAM.lock:
            app.STREAM.active_owner = self.owner
            app.STREAM.active_since = 0
            app.STREAM.last_activity = 0

    def tearDown(self) -> None:
        with app.STREAM.lock:
            if app.STREAM.active_owner == self.owner:
                app.STREAM.active_owner = None
                app.STREAM.active_since = 0
                app.STREAM.last_activity = 0

    def run_processor(
        self,
        mode: str,
        *,
        include_person: bool,
        confidence: float = 0.35,
        raw_confidence: float = 0.92,
        detections: list[tuple[float, list[float], float]] | None = None,
    ):
        model = FakeModel(
            include_person=include_person,
            confidence=raw_confidence,
            detections=detections,
        )
        snapshots: list[tuple[dict[str, object], np.ndarray]] = []
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "read_active_mode", return_value=mode),
            patch.object(app, "read_confidence", return_value=confidence),
            patch.object(app, "get_mode_model", return_value=model),
            patch.object(app, "write_camera_status"),
        ):
            def save_snapshot(
                snapshot_mode: str, image: np.ndarray, metrics: dict[str, object]
            ) -> None:
                self.assertEqual(snapshot_mode, mode)
                snapshots.append((metrics, image.copy()))

            with patch.object(app, "write_snapshot", side_effect=save_snapshot):
                processor = app.ObjectVideoProcessor(self.owner)
                output = processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]
        return model, snapshots, output

    def test_navigation_config_loads_missing_track_age(self) -> None:
        with (
            patch.object(
                app,
                "read_json",
                return_value={"max_missing_track_age_seconds": 0.8},
            ),
            patch.object(app, "read_confidence", return_value=0.35),
        ):
            config = app.read_navigation_config()

        self.assertEqual(config.max_missing_track_age_seconds, 0.8)

    def test_receiver_runs_model_and_publishes_empty_detection_frame(self) -> None:
        model, snapshots, output = self.run_processor("custom", include_person=False)

        self.assertEqual(model.calls, 1)
        self.assertEqual(
            model.prediction_options,
            {"conf": 0.35, "verbose": False, "device": "cpu"},
        )
        self.assertEqual(
            model.tracking_options,
            {
                "persist": True,
                "tracker": str(app.ROOT / "objectvision_bytetrack.yaml"),
            },
        )
        self.assertEqual(len(snapshots), 1)
        metrics, annotated = snapshots[0]
        self.assertEqual(metrics["detections"], 0)
        self.assertEqual(metrics["inference_status"], "no detections")
        self.assertEqual(metrics["frame_id"], app.STREAM.frames)
        self.assertEqual(annotated.shape, self.image.shape)
        self.assertIsInstance(output, av.VideoFrame)

    def test_coco_navigation_uses_the_same_single_inference_result(self) -> None:
        model, snapshots, _ = self.run_processor("coco", include_person=True)

        self.assertEqual(model.calls, 1)
        metrics, _ = snapshots[0]
        self.assertEqual(metrics["detections"], 1)
        self.assertEqual(metrics["objects"][0]["name"], "person")
        navigation = metrics["navigation"]
        self.assertEqual(len(navigation["detections"]), 1)
        self.assertEqual(navigation["detections"][0]["name"], "person")
        self.assertEqual(navigation["frame_fresh"], True)

    def test_processor_publishes_monotonic_stage_durations(self) -> None:
        clock_values = iter(
            (
                0,
                1_000_000_000,
                2_000_000_000,
                2_125_000_000,
                3_000_000_000,
                3_010_000_000,
                4_000_000_000,
                5_000_000_000,
                5_200_000_000,
            )
        )
        with patch.object(
            app.time,
            "perf_counter_ns",
            side_effect=lambda: next(clock_values),
        ):
            _, snapshots, _ = self.run_processor(
                "navigation",
                include_person=True,
            )

        metrics, _ = snapshots[0]
        measured_durations = (
            metrics["inference_latency_ms"],
            metrics["navigation_evaluation_latency_ms"],
            metrics["frame_processing_latency_ms"],
        )
        for duration in measured_durations:
            with self.subTest(duration=duration):
                self.assertIsInstance(duration, (int, float))
                self.assertTrue(math.isfinite(duration))
                self.assertGreaterEqual(duration, 0)
        self.assertEqual(metrics["inference_latency_ms"], 125.0)
        self.assertEqual(metrics["navigation_evaluation_latency_ms"], 10.0)
        self.assertEqual(metrics["frame_processing_latency_ms"], 3000.0)
        telemetry = app.STREAM_TELEMETRY.export(now_ns=6_000_000_000)
        self.assertEqual(
            telemetry["samples"][0]["callback_to_snapshot_publication_ms"],
            5200.0,
        )
        self.assertEqual(
            telemetry["samples"][0]["snapshot_publication_ms"],
            200.0,
        )
        self.assertEqual(telemetry["samples"][0]["frame_id"], metrics["frame_id"])
        self.assertEqual(telemetry["rates"]["receive_rate_hz"], 1 / 6)
        self.assertEqual(
            telemetry["rates"]["successful_processing_rate_hz"],
            1 / 6,
        )
        self.assertEqual(
            telemetry["counters"]["snapshot_publications_completed"],
            1,
        )

    def test_non_navigation_mode_reports_navigation_duration_as_unavailable(self) -> None:
        with patch.object(app.time, "perf_counter_ns", side_effect=range(8)):
            _, snapshots, _ = self.run_processor(
                "custom",
                include_person=True,
            )

        metrics, _ = snapshots[0]
        self.assertIsNone(metrics["navigation_evaluation_latency_ms"])
        self.assertGreaterEqual(metrics["inference_latency_ms"], 0)
        self.assertGreaterEqual(metrics["frame_processing_latency_ms"], 0)

    def test_navigation_uses_memory_updated_from_the_evaluated_frame(self) -> None:
        memory_snapshots: list[dict[str, object]] = []
        call_order: list[str] = []
        original_memory_update = app.VisualMemory.update
        original_evaluate = app.NavigationEngine.evaluate

        def update_memory(memory, detections, **kwargs):
            snapshot = original_memory_update(memory, detections, **kwargs)
            memory_snapshots.append(snapshot)
            call_order.append("memory")
            return snapshot

        def evaluate_navigation(engine, *args, **kwargs):
            call_order.append("navigation")
            self.assertTrue(memory_snapshots)
            self.assertEqual(
                kwargs["visual_memory_tracks"],
                memory_snapshots[-1]["tracks"],
            )
            self.assertEqual(
                memory_snapshots[-1]["tracks"][0]["last_seen"],
                args[3],
            )
            return original_evaluate(engine, *args, **kwargs)

        with (
            patch.object(app.VisualMemory, "update", new=update_memory),
            patch.object(
                app.NavigationEngine,
                "evaluate",
                new=evaluate_navigation,
            ),
        ):
            _, snapshots, _ = self.run_processor(
                "navigation",
                include_person=True,
            )
        self.assertLess(call_order.index("memory"), call_order.index("navigation"))
        self.assertEqual(len(snapshots[0][0]["visual_memory"]["tracks"]), 1)
        self.assertEqual(
            snapshots[0][0]["visual_memory"]["tracks"][0]["status"],
            "ACTIVE",
        )

    def test_raw_confidence_threshold_filters_custom_and_coco_results(self) -> None:
        for mode in ("custom", "coco"):
            with self.subTest(mode=mode):
                model, snapshots, _ = self.run_processor(
                    mode,
                    include_person=True,
                    confidence=0.45,
                    raw_confidence=0.4,
                )
                self.assertEqual(model.prediction_options["conf"], 0.45)
                metrics, _ = snapshots[0]
                self.assertEqual(metrics["detections"], 0)
                self.assertEqual(metrics["objects"], [])

    def test_navigation_mode_uses_selected_confidence_threshold(self) -> None:
        model, snapshots, _ = self.run_processor(
            "navigation",
            include_person=True,
            confidence=0.45,
            raw_confidence=0.4,
        )
        self.assertEqual(model.prediction_options["conf"], 0.45)
        self.assertEqual(snapshots[0][0]["detections"], 0)

    def test_navigation_uses_current_raw_detection_without_tracker_id(self) -> None:
        _, snapshots, _ = self.run_processor(
            "navigation",
            include_person=False,
            detections=[(0.9, [100, 20, 220, 260], None)],
        )
        metrics, _ = snapshots[0]
        self.assertEqual(metrics["objects"][0]["track_id"], None)
        self.assertEqual(metrics["navigation"]["recommendation"], app.STOP)

    def test_snapshot_from_previous_threshold_is_not_current(self) -> None:
        metrics = {"confidence": 0.35}
        with patch.object(app, "read_confidence", return_value=0.45):
            self.assertFalse(app.snapshot_uses_current_confidence(metrics))
        metrics["confidence"] = 0.45
        with patch.object(app, "read_confidence", return_value=0.45):
            self.assertTrue(app.snapshot_uses_current_confidence(metrics))

    def test_overlapping_detections_are_not_removed_by_extra_nms(self) -> None:
        overlapping = [
            (0.9, [80, 40, 190, 240], 1),
            (0.88, [100, 50, 210, 250], 2),
        ]
        _, snapshots, _ = self.run_processor(
            "custom",
            include_person=False,
            detections=overlapping,
        )
        metrics, annotated = snapshots[0]
        self.assertEqual(metrics["detections"], 2)
        self.assertEqual(len(metrics["objects"]), 2)
        self.assertEqual({item["track_id"] for item in metrics["objects"]}, {1, 2})
        self.assertFalse(np.array_equal(annotated, self.image))

    def test_annotation_draws_each_box_once_with_compact_bounded_labels(self) -> None:
        from detection_stability import StableDetection

        detections = [
            StableDetection(71, "bottle", 0.61, 0.63, (5, 0, 120, 90), (5, 0, 120, 90)),
            StableDetection(72, "cup", 0.77, 0.75, (12, 8, 135, 98), (12, 8, 135, 98)),
        ]
        rectangle_calls = []
        text_calls = []
        with (
            patch.object(app.cv2, "rectangle", side_effect=lambda *args: rectangle_calls.append(args)),
            patch.object(app.cv2, "putText", side_effect=lambda *args: text_calls.append(args)),
            patch.object(app.cv2, "getTextSize", return_value=((75, 12), 3)),
        ):
            annotated = app.draw_detection_annotations(
                np.zeros((100, 140, 3), dtype=np.uint8),
                detections,
            )

        self.assertEqual(annotated.shape, (100, 140, 3))
        self.assertEqual(len(text_calls), len(detections))
        self.assertEqual(
            [call[1] for call in text_calls],
            ["bottle 63%", "cup 75%"],
        )
        self.assertTrue(all("#" not in call[1] and "EMA" not in call[1] for call in text_calls))
        self.assertNotEqual(text_calls[0][2][1], text_calls[1][2][1])
        self.assertEqual(
            sum(call[1] == (5, 0) and call[2] == (120, 90) for call in rectangle_calls),
            1,
        )
        self.assertEqual(
            sum(call[1] == (12, 8) and call[2] == (135, 98) for call in rectangle_calls),
            1,
        )
        for call in text_calls:
            x, y = call[2]
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, 140)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, 100)

    def test_navigation_gate_preserves_stop_and_existing_empty_scene_guidance(self) -> None:
        from navigation import NavigationEngine, Detection, STOP, UNCERTAIN

        engine = NavigationEngine()
        now = 100.0
        stop = engine.evaluate(
            [Detection("person", 0.9, 110, 30, 190, 270)],
            300,
            300,
            now,
            now=now,
        )
        self.assertEqual(app.gate_navigation_evidence(stop, image_quality_status="POOR").recommendation, STOP)

        empty = NavigationEngine().evaluate([], 300, 300, now, now=now)
        gated_empty = app.gate_navigation_evidence(
            empty,
            image_quality_status="GOOD",
        )
        self.assertEqual(gated_empty.recommendation, empty.recommendation)
        self.assertIn("does not prove", gated_empty.explanation)

        with_detection = engine.evaluate(
            [Detection("person", 0.9, 110, 30, 190, 150)],
            300,
            300,
            now + 0.1,
            now=now + 0.1,
        )
        gated_poor = app.gate_navigation_evidence(
            with_detection,
            image_quality_status="POOR",
        )
        self.assertEqual(gated_poor.recommendation, UNCERTAIN)
        self.assertTrue(with_detection.detections)

        valid_detection = NavigationEngine().evaluate(
            [Detection("person", 0.9, 145, 80, 155, 90)],
            300,
            300,
            now,
            now=now,
        )
        gated_valid = app.gate_navigation_evidence(
            valid_detection,
            image_quality_status="GOOD",
        )
        self.assertEqual(gated_valid.recommendation, valid_detection.recommendation)
        self.assertNotEqual(gated_valid.recommendation, UNCERTAIN)

        stale = NavigationEngine().evaluate([], None, None, None, now=now)
        gated_stale = app.gate_navigation_evidence(
            stale,
            image_quality_status="GOOD",
        )
        self.assertEqual(gated_stale.recommendation, UNCERTAIN)
        self.assertFalse(gated_stale.frame_fresh)

    def test_mode_switch_resets_previous_tracker_and_smoothing_state(self) -> None:
        first = FakeModel(include_person=True, confidence=0.4)
        second = FakeModel(include_person=True, confidence=0.92)
        tracker = FakeTracker()
        first.predictor = SimpleNamespace(trackers=[tracker])
        processor_owner = self.owner
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "read_active_mode", side_effect=["custom", "coco"]),
            patch.object(app, "read_confidence", return_value=0.35),
            patch.object(app, "read_navigation_config", return_value=app.NavigationConfig()),
            patch.object(app, "get_mode_model", side_effect=[first, second]),
            patch.object(app, "write_camera_status"),
            patch.object(app, "write_snapshot"),
            patch.object(app, "reset_model_tracking", wraps=app.reset_model_tracking) as reset,
        ):
            processor = app.ObjectVideoProcessor(processor_owner)
            processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]
            processor.last_inference_at = 0
            processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]

        self.assertIn(first, [call.args[0] for call in reset.call_args_list])
        self.assertGreaterEqual(tracker.reset_calls, 2)
        self.assertEqual(processor.detection_stabilizer._tracks[1].confidence, 0.92)

    def test_stream_end_resets_tracker_and_smoothing_state(self) -> None:
        tracker = FakeTracker()
        model = FakeModel(include_person=True)
        model.predictor = SimpleNamespace(trackers=[tracker])
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "release_camera"),
        ):
            processor = app.ObjectVideoProcessor(self.owner)
            processor.active_mode = "custom"
            processor.tracking_model = model
            processor.detection_stabilizer.update(
                [(8, "person", 0.8, (0, 0, 40, 80))],
                timestamp=1.0,
            )
            processor.on_ended()

        self.assertEqual(tracker.reset_calls, 1)
        self.assertEqual(processor.detection_stabilizer._tracks, {})

    def test_video_processor_owns_track_end_callback(self) -> None:
        source = Path(app.__file__).read_text()
        tree = ast.parse(source)
        streamer_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "webrtc_streamer"
        )
        keyword_names = {keyword.arg for keyword in streamer_call.keywords}
        self.assertIn("video_processor_factory", keyword_names)
        self.assertNotIn("on_video_ended", keyword_names)
        self.assertTrue(callable(app.ObjectVideoProcessor.on_ended))

    def test_processor_track_end_releases_its_camera_lease(self) -> None:
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "release_camera") as release,
        ):
            processor = app.ObjectVideoProcessor(self.owner)
            processor.on_ended()

        release.assert_called_once_with(self.owner)

    def test_snapshot_keeps_frame_and_metadata_in_one_atomic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.object(app, "STATE_DIR", Path(temporary_dir)):
                app.write_snapshot(
                    "coco",
                    self.image,
                    {"frame_id": 27, "detections": 0},
                )
                snapshot = app.read_snapshot("coco")

        self.assertIsNotNone(snapshot)
        metrics, image_bytes = snapshot
        self.assertEqual(metrics["frame_id"], 27)
        self.assertEqual(metrics["detections"], 0)
        self.assertTrue(image_bytes.startswith(b"\xff\xd8"))

    def test_processor_rejects_invalid_empty_frames_before_inference(self) -> None:
        model = FakeModel(include_person=True)
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "read_active_mode", return_value="custom"),
            patch.object(app, "get_mode_model", return_value=model),
            patch.object(app, "write_camera_status"),
        ):
            processor = app.ObjectVideoProcessor(self.owner)
            output = processor.recv(FakeFrame(np.empty((0, 0, 3), dtype=np.uint8)))  # type: ignore[arg-type]

        self.assertEqual(model.calls, 0)
        self.assertIsInstance(output, FakeFrame)
        telemetry = app.STREAM_TELEMETRY.export()
        self.assertEqual(telemetry["counters"]["callback_frames_received"], 1)
        self.assertEqual(telemetry["counters"]["invalid_frames"], 1)
        self.assertEqual(telemetry["counters"]["frames_skipped"], 1)
        self.assertEqual(telemetry["samples"], [])

    def test_stream_telemetry_counts_throttle_retry_and_processing_failure(self) -> None:
        model = FakeModel(include_person=True)
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "read_active_mode", return_value="custom"),
            patch.object(app, "read_confidence", return_value=0.35),
            patch.object(app, "get_mode_model", return_value=model),
            patch.object(app, "write_camera_status"),
            patch.object(app, "mode_error"),
            patch.object(app, "write_snapshot"),
        ):
            processor = app.ObjectVideoProcessor(self.owner)
            processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]
            processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]
            processor.last_inference_at = 0
            processor.model_retry_after["custom"] = app.time.time() + 30
            processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]
            processor.model_retry_after.clear()
            processor.last_inference_at = 0
            model.track = Mock(side_effect=RuntimeError("inference failed"))  # type: ignore[method-assign]
            processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]

        counters = app.STREAM_TELEMETRY.export()["counters"]
        self.assertEqual(counters["callback_frames_received"], 4)
        self.assertEqual(counters["valid_decoded_frames"], 4)
        self.assertEqual(counters["frames_received"], 4)
        self.assertEqual(counters["accepted_for_processing"], 2)
        self.assertEqual(counters["inference_throttled_frames"], 1)
        self.assertEqual(counters["retry_suppressed_frames"], 1)
        self.assertEqual(counters["processing_failures"], 1)
        self.assertEqual(counters["successfully_processed_frames"], 1)
        self.assertEqual(counters["snapshot_publications_completed"], 1)

    def test_stream_telemetry_records_decode_failure_separately(self) -> None:
        class BrokenFrame:
            def to_ndarray(self, *, format: str) -> np.ndarray:
                raise RuntimeError("decode failed")

        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "read_active_mode", return_value="custom"),
            patch.object(app, "write_camera_status"),
            patch.object(app, "mode_error"),
        ):
            processor = app.ObjectVideoProcessor(self.owner)
            processor.recv(BrokenFrame())  # type: ignore[arg-type]

        counters = app.STREAM_TELEMETRY.export()["counters"]
        self.assertEqual(counters["callback_frames_received"], 1)
        self.assertEqual(counters["decode_failures"], 1)
        self.assertEqual(counters["processing_failures"], 0)
        self.assertEqual(counters["frames_skipped"], 0)

    def test_snapshot_publication_failure_is_not_counted_as_success(self) -> None:
        model = FakeModel(include_person=True)
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "read_active_mode", return_value="custom"),
            patch.object(app, "read_confidence", return_value=0.35),
            patch.object(app, "get_mode_model", return_value=model),
            patch.object(app, "write_camera_status"),
            patch.object(app, "mode_error"),
            patch.object(
                app,
                "write_snapshot",
                side_effect=OSError("snapshot write failed"),
            ),
        ):
            processor = app.ObjectVideoProcessor(self.owner)
            processor.recv(FakeFrame(self.image))  # type: ignore[arg-type]

        counters = app.STREAM_TELEMETRY.export()["counters"]
        self.assertEqual(counters["accepted_for_processing"], 1)
        self.assertEqual(counters["processing_failures"], 1)
        self.assertEqual(counters["snapshot_publication_failures"], 1)
        self.assertEqual(counters["snapshot_publications_completed"], 0)
        self.assertEqual(counters["successfully_processed_frames"], 0)

    def test_stream_telemetry_counter_updates_are_thread_safe_and_bounded(self) -> None:
        telemetry = app.StreamTelemetry()
        telemetry.start(self.owner)
        threads = [
            threading.Thread(
                target=lambda: [
                    telemetry.increment(self.owner, "valid_decoded_frames")
                    for _ in range(500)
                ]
            )
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for frame_id in range(telemetry.MAX_FRAME_SAMPLES + 10):
            telemetry.record_snapshot(
                self.owner,
                {"frame_id": frame_id},
                callback_started_ns=1,
                snapshot_started_ns=2,
                published_ns=3,
            )

        exported = telemetry.export(now_ns=10)

        self.assertEqual(exported["counters"]["valid_decoded_frames"], 4000)
        self.assertEqual(
            exported["counters"]["successfully_processed_frames"],
            telemetry.MAX_FRAME_SAMPLES + 10,
        )
        self.assertEqual(len(exported["samples"]), telemetry.MAX_FRAME_SAMPLES)
        self.assertTrue(exported["sample_retention"]["frame_samples_truncated"])
        self.assertEqual(exported["samples"][0]["frame_id"], 10)
        for index in range(telemetry.MAX_DASHBOARD_SAMPLES + 5):
            telemetry.record_dashboard(
                view="navigation",
                stage="snapshot_read",
                elapsed_ms=float(index),
                frame_id=index,
            )
        dashboard_export = telemetry.export(now_ns=11)
        self.assertEqual(
            len(dashboard_export["dashboard_samples"]),
            telemetry.MAX_DASHBOARD_SAMPLES,
        )
        self.assertTrue(
            dashboard_export["sample_retention"]["dashboard_samples_truncated"]
        )

    def test_stream_telemetry_reset_discards_inflight_old_generation(self) -> None:
        telemetry = app.StreamTelemetry()
        telemetry.start(self.owner)
        old_generation = telemetry.begin_callback(self.owner, 100)
        telemetry.start(self.owner, warmup_excluded=True)
        telemetry.increment(
            self.owner,
            "valid_decoded_frames",
            generation=old_generation,
        )

        exported = telemetry.export(now_ns=200)

        self.assertEqual(exported["counters"]["callback_frames_received"], 0)
        self.assertEqual(exported["counters"]["valid_decoded_frames"], 0)
        self.assertTrue(exported["warmup_excluded"])
        self.assertIsNone(exported["rates"]["receive_rate_hz"])

    def test_exported_stream_telemetry_is_accepted_by_report_importer(self) -> None:
        telemetry = app.StreamTelemetry()
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path = Path(temporary_dir) / "model.pt"
            model_path.write_bytes(b"test-model-weights")
            telemetry.start(self.owner)
            telemetry.begin_callback(self.owner, 10)
            telemetry.begin_callback(self.owner, 20)
            telemetry.start(self.owner, warmup_excluded=True)
            generation = telemetry.begin_callback(self.owner, 1_000_000_000)
            telemetry.increment(
                self.owner,
                "valid_decoded_frames",
                generation=generation,
            )
            telemetry.increment(
                self.owner,
                "frames_received",
                generation=generation,
            )
            telemetry.increment(
                self.owner,
                "accepted_for_processing",
                generation=generation,
            )
            telemetry.record_snapshot(
                self.owner,
                {
                    "frame_id": 7,
                    "mode": "navigation",
                    "resolution": {"width": 640, "height": 480},
                    "confidence": 0.35,
                    "model_name": "model.pt",
                    "_model_path": str(model_path),
                    "device": "cpu",
                    "tracker": "objectvision_bytetrack.yaml",
                    "persist": True,
                    "imgsz": "Ultralytics default; not explicitly passed",
                    "inference_latency_ms": 10.0,
                    "navigation_evaluation_latency_ms": None,
                    "frame_processing_latency_ms": 20.0,
                },
                callback_started_ns=1_000_000_000,
                snapshot_started_ns=1_100_000_000,
                published_ns=1_200_000_000,
                generation=generation,
            )
            exported = telemetry.export(now_ns=2_000_000_000)

        report = build_real_stream_report(exported)

        self.assertEqual(report["method"]["warmup_frames"], 2)
        self.assertIsNone(
            report["latency"]["navigation_evaluation_latency_ms"]["p95_ms"]
        )
        self.assertEqual(report["stream_telemetry"]["rates"]["receive_rate_hz"], 1.0)
        self.assertEqual(
            report["inference"]["model_sha256"],
            hashlib.sha256(b"test-model-weights").hexdigest(),
        )

    def test_dashboard_server_timing_is_recorded_separately(self) -> None:
        telemetry = app.StreamTelemetry()
        with (
            patch.object(app, "STREAM_TELEMETRY", telemetry),
            patch.object(app, "read_snapshot", return_value=({"frame_id": 18}, b"jpg")),
            patch.object(app.time, "perf_counter_ns", side_effect=(10, 110, 200, 350)),
        ):
            snapshot = app.read_dashboard_snapshot("coco", "coco")
            rendered = app.measure_dashboard_render("coco")(lambda: "rendered")()

        self.assertEqual(snapshot, ({"frame_id": 18}, b"jpg"))
        self.assertEqual(rendered, "rendered")
        samples = telemetry.export()["dashboard_samples"]
        self.assertEqual(samples[0]["stage"], "snapshot_read")
        self.assertEqual(samples[0]["elapsed_ms"], 0.0001)
        self.assertEqual(samples[0]["frame_id"], 18)
        self.assertEqual(samples[1]["stage"], "render_call")
        self.assertEqual(samples[1]["elapsed_ms"], 0.00015)
        self.assertIsNone(samples[1]["frame_id"])

    def test_interleaved_dashboard_renders_do_not_claim_frame_correlation(self) -> None:
        telemetry = app.StreamTelemetry()
        snapshots = iter(
            (
                ({"frame_id": 101}, b"first"),
                ({"frame_id": 202}, b"second"),
            )
        )
        clock_values = iter((0, 10, 20, 30, 40, 50, 60, 70))

        with (
            patch.object(app, "STREAM_TELEMETRY", telemetry),
            patch.object(app, "read_snapshot", side_effect=lambda mode: next(snapshots)),
            patch.object(app.time, "perf_counter_ns", side_effect=lambda: next(clock_values)),
        ):
            first_session_snapshot = app.read_dashboard_snapshot("coco", "coco")
            second_session_snapshot = app.read_dashboard_snapshot("coco", "coco")
            app.measure_dashboard_render("coco")(lambda: first_session_snapshot)()
            app.measure_dashboard_render("coco")(lambda: second_session_snapshot)()

        samples = telemetry.export()["dashboard_samples"]
        self.assertEqual(
            [sample["frame_id"] for sample in samples if sample["stage"] == "snapshot_read"],
            [101, 202],
        )
        self.assertEqual(
            [sample["frame_id"] for sample in samples if sample["stage"] == "render_call"],
            [None, None],
        )


if __name__ == "__main__":
    unittest.main()
