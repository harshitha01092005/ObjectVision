from __future__ import annotations

import argparse
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from benchmark_latency import (
    FrameSequence,
    _cleanup_benchmark,
    deterministic_frame,
    parse_resolution,
    summarize_timings,
    write_report,
)


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
