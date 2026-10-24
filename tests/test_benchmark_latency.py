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
    deterministic_frame,
    load_benchmark_reports,
    main,
    parse_positive_finite_float,
    parse_resolution,
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
