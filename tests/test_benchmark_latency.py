from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark_latency import (
    FrameSequence,
    _cleanup_benchmark,
    _arguments,
    analyze_repeatability,
    add_threshold_validation,
    build_real_stream_report,
    deterministic_frame,
    load_benchmark_reports,
    main,
    parse_positive_finite_float,
    parse_resolution,
    REAL_STREAM_MEASUREMENT_DEFINITIONS,
    summarize_timings,
    validate_latency_thresholds,
    write_report,
)


def _report(
    *,
    mode: str = "navigation",
    resolution: tuple[int, int] = (640, 480),
    source_kind: str = "deterministic_synthetic",
    measured_frames: int = 4,
    inference_p50: float | None = 10.0,
    inference_p95: float | None = 20.0,
    navigation_p50: float | None = None,
    navigation_p95: float | None = None,
) -> dict:
    def metric_summary(p50: float | None, p95: float | None) -> dict:
        sample_count = measured_frames if p50 is not None else 0
        return {
            "total_samples": measured_frames,
            "sample_count": sample_count,
            "unavailable_samples": measured_frames - sample_count,
            "p50_ms": p50,
            "p95_ms": p95,
        }

    return {
        "environment": {
            "os": "test-os",
            "architecture": "test-arch",
            "ultralytics": "test-version",
        },
        "input": {
            "kind": source_kind,
            "live": False,
            "resolution": {"width": resolution[0], "height": resolution[1]},
            "frames_are_saved": False,
            "generator": "coordinate_pattern_v1",
        },
        "inference": {
            "mode": mode,
            "model_name": "test-model.pt",
            "model_file": "test-model.pt",
            "model_sha256": "test-digest",
            "confidence": 0.35,
            "device": "cpu",
            "tracker": "objectvision_bytetrack.yaml",
            "persist": True,
            "imgsz": "Ultralytics default; not explicitly passed",
        },
        "method": {
            "warmup_frames": 2,
            "measured_frames": measured_frames,
            "percentile_method": "nearest rank",
            "frame_processing_interval": "fixed test interval",
        },
        "latency": {
            "inference_latency_ms": metric_summary(
                inference_p50, inference_p95
            ),
            "navigation_evaluation_latency_ms": metric_summary(
                navigation_p50, navigation_p95
            ),
            "frame_processing_latency_ms": metric_summary(30.0, 40.0),
        },
    }


def _change_measured_count(report: dict) -> None:
    report["method"]["measured_frames"] = 5
    for summary in report["latency"].values():
        summary["total_samples"] = 5
        if summary["sample_count"]:
            summary["sample_count"] = 5
        summary["unavailable_samples"] = 5 - summary["sample_count"]


def _stream_telemetry(
    *,
    confidence: float = 0.35,
    inference: float = 10.0,
    include_navigation: bool = True,
    elapsed: float = 2.0,
) -> dict:
    counters = {
        "callback_frames_received": 10,
        "valid_decoded_frames": 9,
        "frames_received": 9,
        "frames_skipped": 1,
        "invalid_frames": 1,
        "decode_failures": 0,
        "accepted_for_processing": 5,
        "inference_throttled_frames": 3,
        "retry_suppressed_frames": 1,
        "processing_failures": 2,
        "successfully_processed_frames": 3,
        "snapshot_publications_completed": 3,
        "snapshot_publication_failures": 0,
        "dashboard_render_calls": 1,
        "dashboard_snapshot_reads": 1,
    }
    samples = []
    for frame_id in range(3):
        samples.append(
            {
                "frame_id": frame_id + 1,
                "mode": "navigation",
                "resolution": {"width": 640, "height": 480},
                "confidence": confidence,
                "model_name": "yolov8n.pt",
                "model_sha256": "a" * 64,
                "device": "cpu",
                "tracker": "objectvision_bytetrack.yaml",
                "persist": True,
                "imgsz": "Ultralytics default; not explicitly passed",
                "inference_latency_ms": inference + frame_id,
                "navigation_evaluation_latency_ms": (
                    2.0 + frame_id if include_navigation else None
                ),
                "frame_processing_latency_ms": 20.0 + frame_id,
                "callback_to_snapshot_publication_ms": 30.0 + frame_id,
                "snapshot_publication_ms": 1.0 + frame_id,
            }
        )
    return {
        "schema_version": 1,
        "source_type": "real_webrtc_phone_stream",
        "transport": "WebRTC",
        "generated_at_utc": "2026-10-24T00:00:00+00:00",
        "environment": {
            "os": "test-os",
            "python": "3.12",
            "architecture": "test-arch",
            "processor": "test-cpu",
            "logical_cpu_count": 8,
            "ultralytics": "test-version",
        },
        "capture_configuration": {
            "facing_mode": "ideal environment",
            "audio": False,
            "signalling": "HTTPS",
        },
        "elapsed_seconds": elapsed,
        "rate_denominator_seconds": elapsed,
        "warmup_excluded": True,
        "warmup_frames_excluded": 10,
        "measurement_definitions": {
            **REAL_STREAM_MEASUREMENT_DEFINITIONS,
        },
        "counter_definitions": {
            "callback_frames_received": "recv callbacks observed for the active owner",
            "valid_decoded_frames": "callbacks yielding a non-empty BGR frame",
            "frames_received": (
                "legacy count of valid frames counted after stream ownership check"
            ),
            "frames_skipped": "legacy count of empty or invalid frames only",
            "invalid_frames": "alias count for empty or invalid frames",
            "decode_failures": "frame.to_ndarray exceptions",
            "accepted_for_processing": "valid frames passing throttle and retry checks",
            "inference_throttled_frames": "valid frames rejected by the 200 ms inference interval",
            "retry_suppressed_frames": "valid frames rejected during model retry backoff",
            "processing_failures": "accepted-frame exceptions before successful snapshot publication",
            "successfully_processed_frames": "accepted frames with a published snapshot",
            "snapshot_publications_completed": "successful atomic latest-snapshot replacements",
            "snapshot_publication_failures": "snapshot publication exceptions",
            "dashboard_render_calls": "server-side result view render function calls",
            "dashboard_snapshot_reads": "server-side latest-snapshot reads",
        },
        "counters": counters,
        "rates": {
            "receive_rate_hz": counters["callback_frames_received"] / elapsed,
            "successful_processing_rate_hz": counters[
                "successfully_processed_frames"
            ]
            / elapsed,
        },
        "samples": samples,
        "dashboard_samples": [
            {
                "view": "navigation",
                "stage": "snapshot_read",
                "elapsed_ms": 1.0,
                "frame_id": 3,
            },
            {
                "view": "navigation",
                "stage": "render_call",
                "elapsed_ms": 4.0,
                "frame_id": None,
            },
        ],
        "sample_retention": {
            "frame_sample_capacity": 1200,
            "dashboard_sample_capacity": 600,
            "frame_samples_truncated": False,
            "dashboard_samples_truncated": False,
        },
    }


class LatencyBenchmarkTests(unittest.TestCase):
    def test_summarize_timings_uses_available_samples_and_nearest_rank(self) -> None:
        summary = summarize_timings(
            [
                {
                    "inference_latency_ms": 1.0,
                    "navigation_evaluation_latency_ms": 10.0,
                    "frame_processing_latency_ms": 20.0,
                },
                {
                    "inference_latency_ms": 2.0,
                    "navigation_evaluation_latency_ms": None,
                    "frame_processing_latency_ms": 30.0,
                },
                {
                    "inference_latency_ms": 3.0,
                    "navigation_evaluation_latency_ms": 30.0,
                    "frame_processing_latency_ms": 40.0,
                },
                {
                    "inference_latency_ms": 4.0,
                    "navigation_evaluation_latency_ms": None,
                    "frame_processing_latency_ms": 50.0,
                },
                {
                    "inference_latency_ms": 5.0,
                    "navigation_evaluation_latency_ms": 50.0,
                    "frame_processing_latency_ms": 60.0,
                },
            ]
        )

        self.assertEqual(summary["inference_latency_ms"]["total_samples"], 5)
        self.assertEqual(summary["inference_latency_ms"]["sample_count"], 5)
        self.assertEqual(summary["inference_latency_ms"]["unavailable_samples"], 0)
        self.assertEqual(summary["inference_latency_ms"]["p50_ms"], 3.0)
        self.assertEqual(summary["inference_latency_ms"]["p95_ms"], 5.0)
        self.assertEqual(
            summary["navigation_evaluation_latency_ms"]["sample_count"], 3
        )
        self.assertEqual(
            summary["navigation_evaluation_latency_ms"]["unavailable_samples"], 2
        )
        self.assertEqual(
            summary["navigation_evaluation_latency_ms"]["p50_ms"], 30.0
        )
        self.assertEqual(
            summary["navigation_evaluation_latency_ms"]["p95_ms"], 50.0
        )

    def test_summarize_timings_reports_all_unavailable_as_null_percentiles(self) -> None:
        summary = summarize_timings(
            [{"navigation_evaluation_latency_ms": None}]
        )["navigation_evaluation_latency_ms"]

        self.assertEqual(summary["sample_count"], 0)
        self.assertEqual(summary["unavailable_samples"], 1)
        self.assertIsNone(summary["p50_ms"])
        self.assertIsNone(summary["p95_ms"])

    def test_summarize_timings_rejects_invalid_elapsed_values(self) -> None:
        for invalid in (-1.0, math.nan, math.inf, True, "12"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    summarize_timings([{"inference_latency_ms": invalid}])

    def test_threshold_parser_accepts_only_finite_positive_numbers(self) -> None:
        self.assertEqual(parse_positive_finite_float("12.5"), 12.5)
        for invalid in ("0", "-1", "nan", "inf", "-inf", "1e999", "fast"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError,
                    "finite positive number",
                ):
                    parse_positive_finite_float(invalid)

    def test_cli_reports_clear_error_for_invalid_threshold(self) -> None:
        stderr = io.StringIO()
        with (
            patch(
                "sys.argv",
                ["benchmark_latency.py", "--max-inference-latency-ms", "nan"],
            ),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            _arguments()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "latency threshold must be a finite positive number of milliseconds",
            stderr.getvalue(),
        )

    def test_cli_collects_each_optional_threshold(self) -> None:
        with patch(
            "sys.argv",
            [
                "benchmark_latency.py",
                "--max-inference-latency-ms",
                "25",
                "--max-navigation-evaluation-latency-ms",
                "4.5",
                "--max-frame-processing-latency-ms",
                "40",
            ],
        ):
            args = _arguments()

        self.assertEqual(
            args.latency_thresholds,
            {
                "inference_latency_ms": 25.0,
                "navigation_evaluation_latency_ms": 4.5,
                "frame_processing_latency_ms": 40.0,
            },
        )

    def test_threshold_validation_passes_and_fails_on_measured_p95(self) -> None:
        validation = validate_latency_thresholds(
            {
                "inference_latency_ms": {"p95_ms": 20.0},
                "frame_processing_latency_ms": {"p95_ms": 45.0},
            },
            {
                "inference_latency_ms": 20.0,
                "frame_processing_latency_ms": 40.0,
            },
        )

        self.assertEqual(validation["inference_latency_ms"]["status"], "pass")
        self.assertEqual(validation["inference_latency_ms"]["threshold_ms"], 20.0)
        self.assertEqual(validation["inference_latency_ms"]["measured_p95_ms"], 20.0)
        self.assertIn("at or below", validation["inference_latency_ms"]["reason"])
        self.assertEqual(validation["frame_processing_latency_ms"]["status"], "fail")
        self.assertEqual(validation["frame_processing_latency_ms"]["measured_p95_ms"], 45.0)
        self.assertIn("exceeds", validation["frame_processing_latency_ms"]["reason"])

    def test_threshold_validation_marks_missing_measurement_not_evaluable(self) -> None:
        validation = validate_latency_thresholds(
            {"navigation_evaluation_latency_ms": {"p95_ms": None}},
            {"navigation_evaluation_latency_ms": 5.0},
        )["navigation_evaluation_latency_ms"]

        self.assertEqual(validation["threshold_ms"], 5.0)
        self.assertIsNone(validation["measured_p95_ms"])
        self.assertEqual(validation["status"], "not_evaluable")
        self.assertIn("No measured p95", validation["reason"])

    def test_threshold_validation_rejects_invalid_values(self) -> None:
        for invalid in (0, -1, math.nan, math.inf, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    validate_latency_thresholds(
                        {"inference_latency_ms": {"p95_ms": 1.0}},
                        {"inference_latency_ms": invalid},
                    )

    def test_no_thresholds_preserve_report_schema(self) -> None:
        report = {
            "latency": {"inference_latency_ms": {"p95_ms": 10.0}},
            "inference": {"model_name": "model.pt"},
        }

        result = add_threshold_validation(report, {})

        self.assertEqual(result, report)
        self.assertNotIn("threshold_validation", result)

    def test_repeatability_analysis_reports_run_percentiles_and_ranges(self) -> None:
        first = _report(
            inference_p50=10.0,
            inference_p95=20.0,
            navigation_p50=2.0,
            navigation_p95=4.0,
        )
        second = _report(
            inference_p50=12.0,
            inference_p95=24.0,
            navigation_p50=3.0,
            navigation_p95=6.0,
        )

        result = analyze_repeatability([first, second])

        inference = result["metrics"]["inference_latency_ms"]
        self.assertEqual(
            [run["p50_ms"] for run in inference["runs"]],
            [10.0, 12.0],
        )
        self.assertEqual(
            [run["p95_ms"] for run in inference["runs"]],
            [20.0, 24.0],
        )
        self.assertEqual(
            inference["across_run_variability"]["p95_range_ms"],
            4.0,
        )
        self.assertEqual(
            result["metrics"]["navigation_evaluation_latency_ms"][
                "across_run_variability"
            ]["p50_range_ms"],
            1.0,
        )
        self.assertIn("not frame-to-frame variability", result["variability_definition"])

    def test_repeatability_analysis_rejects_configuration_mismatches(self) -> None:
        cases = (
            ("mode", lambda report: report["inference"].update(mode="coco")),
            (
                "resolution",
                lambda report: report["input"]["resolution"].update(width=1280),
            ),
            (
                "inference settings",
                lambda report: report["inference"].update(confidence=0.5),
            ),
            (
                "source type",
                lambda report: report["input"].update(kind="prerecorded_video"),
            ),
            (
                "sample count",
                _change_measured_count,
            ),
        )
        for label, change in cases:
            with self.subTest(configuration=label):
                first = _report()
                second = copy.deepcopy(first)
                change(second)
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    analyze_repeatability([first, second])

    def test_repeatability_keeps_missing_navigation_metrics_unavailable(self) -> None:
        first = _report(navigation_p50=2.0, navigation_p95=4.0)
        second = copy.deepcopy(first)
        second["latency"].pop("navigation_evaluation_latency_ms")

        metric = analyze_repeatability([first, second])["metrics"][
            "navigation_evaluation_latency_ms"
        ]

        self.assertEqual(
            [run["p50_ms"] for run in metric["runs"]],
            [2.0, None],
        )
        self.assertEqual(
            [run["unavailable_samples"] for run in metric["runs"]],
            [0, 4],
        )
        self.assertIsNone(
            metric["across_run_variability"]["p95_range_ms"]
        )
        self.assertEqual(
            metric["across_run_variability"]["available_runs"],
            1,
        )
        self.assertEqual(
            metric["across_run_variability"]["unavailable_runs"],
            1,
        )

    def test_repeatability_rejects_invalid_report_shapes_and_counts(self) -> None:
        for reports in (
            [{}, _report()],
            [_report(), "not a report"],
        ):
            with self.subTest(reports=reports):
                with self.assertRaises(ValueError):
                    analyze_repeatability(reports)

        invalid = _report()
        invalid["latency"]["inference_latency_ms"]["sample_count"] = 3
        with self.assertRaisesRegex(ValueError, "inconsistent sample counts"):
            analyze_repeatability([invalid, _report()])

    def test_repeatability_output_is_deterministic(self) -> None:
        reports = [
            _report(inference_p50=10.0, inference_p95=20.0),
            _report(inference_p50=12.0, inference_p95=24.0),
        ]

        first = json.dumps(analyze_repeatability(reports), sort_keys=True)
        second = json.dumps(analyze_repeatability(reports), sort_keys=True)

        self.assertEqual(first, second)

    def test_repeatability_cli_reads_reports_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            report_paths = [directory / "run-1.json", directory / "run-2.json"]
            for path, report in zip(report_paths, (_report(), _report())):
                path.write_text(json.dumps(report), encoding="utf-8")
            output_path = directory / "repeatability.json"

            with patch(
                "sys.argv",
                [
                    "benchmark_latency.py",
                    "--compare",
                    *(str(path) for path in report_paths),
                    "--output",
                    str(output_path),
                ],
            ):
                main()

            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output["run_count"], 2)
            self.assertIn("metrics", output)

    def test_repeatability_report_loader_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "invalid.json"
            report_path.write_text("{invalid", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not read benchmark report"):
                load_benchmark_reports([report_path])

    def test_real_stream_telemetry_builds_rates_and_latency_summaries(self) -> None:
        report = build_real_stream_report(_stream_telemetry())

        self.assertEqual(report["input"]["kind"], "real_webrtc_phone_stream")
        self.assertEqual(
            report["method"]["measurement_definitions"],
            REAL_STREAM_MEASUREMENT_DEFINITIONS,
        )
        self.assertEqual(report["latency"]["inference_latency_ms"]["sample_count"], 3)
        self.assertEqual(report["latency"]["snapshot_publication_ms"]["p95_ms"], 3.0)
        self.assertEqual(
            report["stream_telemetry"]["rates"]["receive_rate_hz"],
            5.0,
        )
        self.assertEqual(
            report["dashboard_latency"]["dashboard_render_call_ms"]["p50_ms"],
            4.0,
        )

    def test_real_stream_telemetry_rejects_noncanonical_timing_definitions(self) -> None:
        incorrect = _stream_telemetry()
        incorrect["measurement_definitions"][
            "frame_processing_latency_ms"
        ] = "includes snapshot publication"
        with self.assertRaisesRegex(ValueError, "changes measurement definitions"):
            build_real_stream_report(incorrect)

        missing = _stream_telemetry()
        missing["measurement_definitions"].pop("snapshot_publication_ms")
        with self.assertRaisesRegex(ValueError, "changes measurement definitions"):
            build_real_stream_report(missing)

        malformed = _stream_telemetry()
        malformed["measurement_definitions"]["dashboard_render_call_ms"] = 42
        with self.assertRaisesRegex(ValueError, "changes measurement definitions"):
            build_real_stream_report(malformed)

        valid_report = build_real_stream_report(_stream_telemetry())
        mismatched_report = copy.deepcopy(valid_report)
        mismatched_report["method"]["measurement_definitions"][
            "snapshot_publication_ms"
        ] = "excludes JPEG encoding"
        with self.assertRaisesRegex(
            ValueError, "invalid real-stream measurement definitions"
        ):
            analyze_repeatability([valid_report, mismatched_report])

    def test_real_stream_telemetry_keeps_missing_navigation_unavailable(self) -> None:
        report = build_real_stream_report(
            _stream_telemetry(include_navigation=False)
        )
        navigation = report["latency"]["navigation_evaluation_latency_ms"]

        self.assertEqual(navigation["sample_count"], 0)
        self.assertEqual(navigation["unavailable_samples"], 3)
        self.assertIsNone(navigation["p50_ms"])
        self.assertIsNone(navigation["p95_ms"])

    def test_real_stream_telemetry_rejects_invalid_counters_and_denominators(self) -> None:
        invalid_counter = _stream_telemetry()
        invalid_counter["counters"]["frames_skipped"] = 0
        with self.assertRaisesRegex(ValueError, "legacy frames_skipped"):
            build_real_stream_report(invalid_counter)

        malformed_counter = _stream_telemetry()
        malformed_counter["counters"]["successfully_processed_frames"] = "3"
        with self.assertRaisesRegex(ValueError, "successful-frame counter"):
            build_real_stream_report(malformed_counter)

        overcounted_decode_paths = _stream_telemetry()
        overcounted_decode_paths["counters"]["valid_decoded_frames"] = 10
        with self.assertRaisesRegex(ValueError, "inconsistent stream counters"):
            build_real_stream_report(overcounted_decode_paths)

        overcounted_processing_paths = _stream_telemetry()
        overcounted_processing_paths["counters"]["inference_throttled_frames"] = 4
        with self.assertRaisesRegex(ValueError, "inconsistent stream counters"):
            build_real_stream_report(overcounted_processing_paths)

        invalid_denominator = _stream_telemetry()
        invalid_denominator["rate_denominator_seconds"] = 3.0
        with self.assertRaisesRegex(ValueError, "denominator"):
            build_real_stream_report(invalid_denominator)

        invalid_rates = _stream_telemetry()
        invalid_rates["rates"]["receive_rate_hz"] = 0
        with self.assertRaisesRegex(ValueError, "receive_rate_hz"):
            build_real_stream_report(invalid_rates)

    def test_real_stream_telemetry_validates_retention_flags_against_counts(self) -> None:
        invalid_frame_retention = _stream_telemetry()
        invalid_frame_retention["sample_retention"][
            "frame_samples_truncated"
        ] = True
        with self.assertRaisesRegex(ValueError, "sample-retention flags"):
            build_real_stream_report(invalid_frame_retention)

        invalid_dashboard_retention = _stream_telemetry()
        invalid_dashboard_retention["sample_retention"][
            "dashboard_samples_truncated"
        ] = True
        with self.assertRaisesRegex(ValueError, "sample-retention flags"):
            build_real_stream_report(invalid_dashboard_retention)

        truncated_frames = _stream_telemetry()
        counters = truncated_frames["counters"]
        counters.update(
            callback_frames_received=11,
            valid_decoded_frames=10,
            frames_received=10,
            accepted_for_processing=6,
            successfully_processed_frames=4,
            snapshot_publications_completed=4,
        )
        truncated_frames["rates"].update(
            receive_rate_hz=5.5,
            successful_processing_rate_hz=2.0,
        )
        truncated_frames["sample_retention"]["frame_samples_truncated"] = True

        report = build_real_stream_report(truncated_frames)

        self.assertEqual(report["method"]["measured_frames"], 3)
        self.assertTrue(
            report["stream_telemetry"]["sample_retention"][
                "frame_samples_truncated"
            ]
        )

    def test_real_stream_reports_require_valid_telemetry_for_repeatability(self) -> None:
        real_report = build_real_stream_report(_stream_telemetry())
        missing_telemetry = dict(real_report)
        missing_telemetry.pop("stream_telemetry")
        with self.assertRaisesRegex(ValueError, "missing stream telemetry"):
            analyze_repeatability([missing_telemetry, real_report])

        malformed_telemetry = dict(real_report)
        malformed_telemetry["stream_telemetry"] = dict(
            real_report["stream_telemetry"]
        )
        malformed_telemetry["stream_telemetry"]["counters"] = dict(
            real_report["stream_telemetry"]["counters"]
        )
        malformed_telemetry["stream_telemetry"]["counters"].pop("frames_received")
        with self.assertRaisesRegex(ValueError, "missing stream counters"):
            analyze_repeatability([real_report, malformed_telemetry])

    def test_real_stream_telemetry_rejects_mixed_settings_and_invalid_reports(self) -> None:
        mixed = _stream_telemetry()
        mixed["samples"][1]["confidence"] = 0.5
        with self.assertRaisesRegex(ValueError, "mixed frame configurations"):
            build_real_stream_report(mixed)

        invalid_sample = _stream_telemetry()
        invalid_sample["samples"][0]["snapshot_publication_ms"] = math.inf
        with self.assertRaisesRegex(ValueError, "Invalid snapshot_publication_ms"):
            build_real_stream_report(invalid_sample)

        invalid_source = _stream_telemetry()
        invalid_source["source_type"] = "synthetic"
        with self.assertRaisesRegex(ValueError, "source_type"):
            build_real_stream_report(invalid_source)

        not_warmed = _stream_telemetry()
        not_warmed["warmup_excluded"] = False
        with self.assertRaisesRegex(ValueError, "Reset telemetry after stream warm-up"):
            build_real_stream_report(not_warmed)

    def test_real_stream_repeatability_requires_matching_configuration(self) -> None:
        first = build_real_stream_report(_stream_telemetry(inference=10.0))
        second = build_real_stream_report(_stream_telemetry(inference=12.0))
        analysis = analyze_repeatability([first, second])

        self.assertEqual(analysis["run_count"], 2)
        self.assertEqual(
            analysis["throughput"]["receive_rate_hz"]["runs"][0]["rate_hz"],
            5.0,
        )
        self.assertEqual(
            analysis["throughput"]["receive_rate_hz"]["across_run_range_hz"],
            0.0,
        )
        self.assertEqual(
            analysis["metrics"]["snapshot_publication_ms"][
                "across_run_variability"
            ]["p95_range_ms"],
            0.0,
        )
        self.assertEqual(
            analysis["dashboard_metrics"]["dashboard_render_call_ms"]["runs"][0][
                "p50_ms"
            ],
            4.0,
        )

        mismatched = build_real_stream_report(
            _stream_telemetry(confidence=0.5)
        )
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            analyze_repeatability([first, mismatched])
        changed_definition = build_real_stream_report(_stream_telemetry())
        changed_definition["method"]["measurement_definitions"][
            "snapshot_publication_ms"
        ] = "different boundary"
        with self.assertRaisesRegex(
            ValueError, "invalid real-stream measurement definitions"
        ):
            analyze_repeatability([first, changed_definition])
        changed_dashboard = build_real_stream_report(_stream_telemetry())
        changed_dashboard["method"]["dashboard_views_measured"] = ["custom"]
        with self.assertRaisesRegex(ValueError, "configuration differs in: method"):
            analyze_repeatability([first, changed_dashboard])

    def test_real_stream_without_model_digest_is_reportable_but_not_comparable(self) -> None:
        telemetry = _stream_telemetry()
        for sample in telemetry["samples"]:
            sample["model_sha256"] = None

        report = build_real_stream_report(telemetry)

        self.assertIsNone(report["inference"]["model_sha256"])
        with self.assertRaisesRegex(ValueError, "without a model SHA-256 digest"):
            analyze_repeatability([report, report])

    def test_real_stream_telemetry_cli_reads_export_without_running_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            telemetry_path = directory / "telemetry.json"
            output_path = directory / "summary.json"
            telemetry_path.write_text(
                json.dumps(_stream_telemetry()),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "benchmark_latency.py",
                    "--telemetry",
                    str(telemetry_path),
                    "--output",
                    str(output_path),
                ],
            ):
                main()

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["benchmark"],
                "ObjectVision real WebRTC stream performance telemetry",
            )

    def test_resolution_parser_requires_positive_width_and_height(self) -> None:
        self.assertEqual(parse_resolution("640x480"), (640, 480))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_resolution("0x480")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_resolution("-1x480")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_resolution("640")

    def test_deterministic_frame_repeats_at_requested_resolution(self) -> None:
        first = deterministic_frame(32, 24)
        second = deterministic_frame(32, 24)

        self.assertEqual(first.shape, (24, 32, 3))
        self.assertEqual(first.dtype.name, "uint8")
        self.assertTrue((first == second).all())

    def test_default_input_metadata_identifies_synthetic_non_live_source(self) -> None:
        sequence = FrameSequence(None, (32, 24))
        try:
            metadata = sequence.metadata()
        finally:
            sequence.close()

        self.assertEqual(metadata["kind"], "deterministic_synthetic")
        self.assertFalse(metadata["live"])
        self.assertFalse(metadata["frames_are_saved"])

    def test_write_report_refuses_to_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "existing-report.json"
            report_path.write_text("original report")
            with self.assertRaises(FileExistsError):
                write_report(report_path, {"sample_count": 1})

            self.assertEqual(report_path.read_text(), "original report")

    def test_cleanup_restores_stream_and_closes_input_when_model_reset_fails(self) -> None:
        original_state = {"frames": 4, "active_owner": "prior-owner"}
        stream = SimpleNamespace(
            lock=threading.Lock(),
            frames=99,
            active_owner="benchmark-owner",
        )
        sequence = SimpleNamespace(close=Mock())
        app = SimpleNamespace(
            STREAM=stream,
            reset_model_tracking=Mock(side_effect=RuntimeError("reset failed")),
        )
        processor = SimpleNamespace(tracking_model=object(), active_mode="coco")

        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            _cleanup_benchmark(app, processor, sequence, original_state)

        sequence.close.assert_called_once_with()
        self.assertEqual(stream.frames, 4)
        self.assertEqual(stream.active_owner, "prior-owner")

    def test_cleanup_preserves_benchmark_error_when_model_reset_fails(self) -> None:
        stream = SimpleNamespace(
            lock=threading.Lock(),
            frames=99,
            active_owner="benchmark-owner",
        )
        sequence = SimpleNamespace(close=Mock())
        app = SimpleNamespace(
            STREAM=stream,
            reset_model_tracking=Mock(side_effect=RuntimeError("reset failed")),
        )
        processor = SimpleNamespace(tracking_model=object(), active_mode="coco")

        with self.assertRaisesRegex(ValueError, "benchmark failed"):
            try:
                raise ValueError("benchmark failed")
            finally:
                _cleanup_benchmark(
                    app,
                    processor,
                    sequence,
                    {"frames": 4, "active_owner": "prior-owner"},
                    preserve_active_exception=True,
                )

        sequence.close.assert_called_once_with()
        self.assertEqual(stream.frames, 4)
        self.assertEqual(stream.active_owner, "prior-owner")


if __name__ == "__main__":
    unittest.main()
