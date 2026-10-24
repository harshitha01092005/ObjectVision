from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TIMING_FIELDS = (
    "inference_latency_ms",
    "navigation_evaluation_latency_ms",
    "frame_processing_latency_ms",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
THRESHOLD_OPTIONS = {
    "inference_latency_ms": "--max-inference-latency-ms",
    "navigation_evaluation_latency_ms": "--max-navigation-evaluation-latency-ms",
    "frame_processing_latency_ms": "--max-frame-processing-latency-ms",
}


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Resolution must use WIDTHxHEIGHT, for example 640x480."
        ) from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive.")
    return width, height


def parse_positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "latency threshold must be a finite positive number of milliseconds."
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "latency threshold must be a finite positive number of milliseconds."
        )
    return parsed


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def summarize_timings(
    samples: Iterable[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    rows = list(samples)
    summary: dict[str, dict[str, int | float | None]] = {}
    for field in TIMING_FIELDS:
        available: list[float] = []
        unavailable = 0
        for row in rows:
            value = row.get(field)
            if value is None:
                unavailable += 1
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Invalid {field} value in benchmark sample: {value!r}")
            available.append(float(value))

        summary[field] = {
            "total_samples": len(rows),
            "sample_count": len(available),
            "unavailable_samples": unavailable,
            "p50_ms": _nearest_rank(available, 0.50),
            "p95_ms": _nearest_rank(available, 0.95),
        }
    return summary


def _validated_report(report: Any, report_index: int) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError(f"Benchmark report {report_index} must be a JSON object.")
    for section in ("environment", "input", "inference", "method", "latency"):
        if not isinstance(report.get(section), dict):
            raise ValueError(
                f"Benchmark report {report_index} is missing its {section} object."
            )

    method = report["method"]
    measured_frames = method.get("measured_frames")
    if (
        isinstance(measured_frames, bool)
        or not isinstance(measured_frames, int)
        or measured_frames < 1
    ):
        raise ValueError(
            f"Benchmark report {report_index} has an invalid measured_frames value."
        )

    for metric in TIMING_FIELDS:
        summary = report["latency"].get(metric)
        if summary is None:
            continue
        if not isinstance(summary, dict):
            raise ValueError(
                f"Benchmark report {report_index} has an invalid {metric} summary."
            )
        sample_count = summary.get("sample_count")
        unavailable = summary.get("unavailable_samples")
        total = summary.get("total_samples")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in (sample_count, unavailable, total)
        ) or total != measured_frames or sample_count + unavailable != total:
            raise ValueError(
                f"Benchmark report {report_index} has inconsistent sample counts "
                f"for {metric}."
            )
        for percentile in ("p50_ms", "p95_ms"):
            value = summary.get(percentile)
            if value is None:
                if sample_count:
                    raise ValueError(
                        f"Benchmark report {report_index} has samples but no "
                        f"{percentile} for {metric}."
                    )
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Benchmark report {report_index} has an invalid {percentile} "
                    f"for {metric}."
                )
        if not sample_count and any(
            summary.get(percentile) is not None for percentile in ("p50_ms", "p95_ms")
        ):
            raise ValueError(
                f"Benchmark report {report_index} has percentiles without samples "
                f"for {metric}."
            )
    return report


def analyze_repeatability(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("Repeatability analysis requires at least two reports.")
    validated = [
        _validated_report(report, index)
        for index, report in enumerate(reports, start=1)
    ]
    reference = validated[0]
    configuration_sections = ("environment", "input", "inference", "method")
    for index, report in enumerate(validated[1:], start=2):
        differences = [
            section
            for section in configuration_sections
            if report[section] != reference[section]
        ]
        if differences:
            raise ValueError(
                f"Benchmark report {index} configuration differs in: "
                f"{', '.join(differences)}."
            )

    metrics: dict[str, Any] = {}
    measured_frames = reference["method"]["measured_frames"]
    for metric in TIMING_FIELDS:
        run_summaries = []
        p50_values: list[float] = []
        p95_values: list[float] = []
        for index, report in enumerate(validated, start=1):
            summary = report["latency"].get(metric)
            if summary is None:
                sample_count = 0
                unavailable_samples = measured_frames
                p50 = None
                p95 = None
            else:
                sample_count = summary["sample_count"]
                unavailable_samples = summary["unavailable_samples"]
                p50 = summary["p50_ms"]
                p95 = summary["p95_ms"]
            if p50 is not None:
                p50_values.append(float(p50))
            if p95 is not None:
                p95_values.append(float(p95))
            run_summaries.append(
                {
                    "run": index,
                    "sample_count": sample_count,
                    "unavailable_samples": unavailable_samples,
                    "p50_ms": p50,
                    "p95_ms": p95,
                }
            )

        metrics[metric] = {
            "runs": run_summaries,
            "across_run_variability": {
                "measure": "range of per-run percentiles",
                "p50_range_ms": (
                    max(p50_values) - min(p50_values)
                    if len(p50_values) > 1
                    else None
                ),
                "p95_range_ms": (
                    max(p95_values) - min(p95_values)
                    if len(p95_values) > 1
                    else None
                ),
                "available_runs": len(p95_values),
                "unavailable_runs": len(validated) - len(p95_values),
            },
        }

    return {
        "analysis": "ObjectVision benchmark repeatability",
        "configuration": {
            section: reference[section] for section in configuration_sections
        },
        "run_count": len(validated),
        "variability_definition": (
            "Range (maximum minus minimum) of each percentile across benchmark "
            "runs; this is not frame-to-frame variability."
        ),
        "metrics": metrics,
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def load_benchmark_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for index, path in enumerate(paths, start=1):
        try:
            with path.open(encoding="utf-8") as source:
                reports.append(
                    json.load(
                        source,
                        parse_constant=_reject_json_constant,
                    )
                )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Could not read benchmark report {path}: {exc}") from exc
        _validated_report(reports[-1], index)
    return reports


def validate_latency_thresholds(
    latency: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    unknown_metrics = set(thresholds) - set(TIMING_FIELDS)
    if unknown_metrics:
        raise ValueError(
            f"Unsupported latency threshold metric(s): {', '.join(sorted(unknown_metrics))}"
        )

    validation: dict[str, dict[str, Any]] = {}
    for metric in TIMING_FIELDS:
        if metric not in thresholds:
            continue
        threshold = thresholds[metric]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError(
                f"Threshold for {metric} must be a finite positive number of milliseconds."
            )

        measured_p95 = latency.get(metric, {}).get("p95_ms")
        if measured_p95 is None:
            status = "not_evaluable"
            reason = "No measured p95 is available for this metric."
        elif (
            isinstance(measured_p95, bool)
            or not isinstance(measured_p95, (int, float))
            or not math.isfinite(measured_p95)
            or measured_p95 < 0
        ):
            raise ValueError(f"Invalid measured p95 value for {metric}.")
        elif measured_p95 <= threshold:
            status = "pass"
            reason = "Measured p95 is at or below the configured threshold."
        else:
            status = "fail"
            reason = "Measured p95 exceeds the configured threshold."

        validation[metric] = {
            "threshold_ms": threshold,
            "measured_p95_ms": measured_p95,
            "status": status,
            "reason": reason,
        }
    return validation


def add_threshold_validation(
    report: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if thresholds:
        report["threshold_validation"] = validate_latency_thresholds(
            report["latency"],
            thresholds,
        )
    return report


def deterministic_frame(width: int, height: int):
    import numpy as np

    y, x = np.indices((height, width), dtype=np.uint32)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (x * 3 + y * 5) % 256
    frame[:, :, 1] = (x * 7 + y * 2) % 256
    frame[:, :, 2] = (x + y * 11) % 256
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as destination:
        destination.write(rendered)
        destination.write("\n")


def _cleanup_benchmark(
    app: Any,
    processor: Any,
    sequence: FrameSequence,
    saved_stream_state: dict[str, Any],
    *,
    preserve_active_exception: bool = False,
) -> None:
    cleanup_errors: list[BaseException] = []
    if processor is not None:
        try:
            app.reset_model_tracking(processor.tracking_model, processor.active_mode)
        except BaseException as exc:
            cleanup_errors.append(exc)
    try:
        sequence.close()
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        with app.STREAM.lock:
            for name, value in saved_stream_state.items():
                setattr(app.STREAM, name, value)
    except BaseException as exc:
        cleanup_errors.append(exc)
    if cleanup_errors and not preserve_active_exception:
        raise cleanup_errors[0]


class FrameSequence:
    def __init__(self, input_path: Path | None, resolution: tuple[int, int]) -> None:
        import cv2

        self._cv2 = cv2
        self.width, self.height = resolution
        self.input_path = input_path
        self._image = None
        self._capture = None
        if input_path is None:
            self.kind = "deterministic_synthetic"
            return
        if not input_path.is_file():
            raise FileNotFoundError(f"Benchmark input file does not exist: {input_path}")
        if input_path.suffix.lower() in IMAGE_SUFFIXES:
            self.kind = "prerecorded_image"
            self._image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if self._image is None:
                raise ValueError(f"Could not decode benchmark image: {input_path}")
        else:
            self.kind = "prerecorded_video"
            self._capture = cv2.VideoCapture(str(input_path))
            if not self._capture.isOpened():
                self._capture.release()
                self._capture = None
                raise ValueError(f"Could not open benchmark video: {input_path}")

    def next_frame(self):
        if self.kind == "deterministic_synthetic":
            frame = deterministic_frame(self.width, self.height)
        elif self._image is not None:
            frame = self._image
        else:
            if self._capture is None:
                raise RuntimeError("Benchmark video capture is not open.")
            ok, frame = self._capture.read()
            if not ok:
                self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Benchmark video contains no decodable frames: {self.input_path}"
                    )
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = self._cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=self._cv2.INTER_AREA,
            )
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def rewind(self) -> None:
        if self._capture is not None:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)

    def metadata(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "kind": self.kind,
            "live": False,
            "resolution": {"width": self.width, "height": self.height},
        }
        if self.input_path is not None:
            details.update(
                {
                    "file_name": self.input_path.name,
                    "sha256": _sha256(self.input_path),
                    "frames_are_saved": False,
                }
            )
        else:
            details["frames_are_saved"] = False
        if self.kind == "deterministic_synthetic":
            details["generator"] = "coordinate_pattern_v1"
        return details


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure ObjectVision's published inference, navigation, and "
            "processed-frame latency fields without a live camera."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("custom", "coco", "navigation"),
        default="navigation",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional prerecorded image/video. Defaults to a deterministic synthetic frame.",
    )
    parser.add_argument("--resolution", type=parse_resolution, default=(640, 480))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="REPORT.json",
        help="Analyze repeatability across two or more existing benchmark reports.",
    )
    for metric, option in THRESHOLD_OPTIONS.items():
        parser.add_argument(
            option,
            dest=f"max_{metric.removesuffix('_latency_ms')}_latency_ms",
            type=parse_positive_finite_float,
            help=f"Optional maximum p95 for {metric} in milliseconds.",
        )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; existing files are not overwritten.",
    )
    args = parser.parse_args()
    if not 0.05 <= args.confidence <= 0.95:
        parser.error("--confidence must be between 0.05 and 0.95.")
    if args.warmup < 0:
        parser.error("--warmup must be zero or greater.")
    if args.samples < 1:
        parser.error("--samples must be positive.")
    args.latency_thresholds = {
        metric: getattr(
            args,
            f"max_{metric.removesuffix('_latency_ms')}_latency_ms",
        )
        for metric in TIMING_FIELDS
        if getattr(
            args,
            f"max_{metric.removesuffix('_latency_ms')}_latency_ms",
        )
        is not None
    }
    return args


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import av
    from unittest.mock import patch

    import app

    model_path = app.MODEL_PATH if args.mode == "custom" else app.COCO_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights are unavailable at {model_path}; benchmark was not run."
        )

    sequence = FrameSequence(args.input, args.resolution)
    owner = f"latency-benchmark-{os.getpid()}"
    captured: list[dict[str, Any]] = []
    errors: list[str] = []
    processor = None
    saved_stream_state: dict[str, Any]
    with app.STREAM.lock:
        saved_stream_state = {
            "frames": app.STREAM.frames,
            "detections": app.STREAM.detections,
            "class_counts": app.STREAM.class_counts,
            "latest_frame": app.STREAM.latest_frame,
            "last_update": app.STREAM.last_update,
            "active_owner": app.STREAM.active_owner,
            "active_since": app.STREAM.active_since,
            "last_activity": app.STREAM.last_activity,
        }
        app.STREAM.active_owner = owner
        app.STREAM.active_since = app.time.time()
        app.STREAM.last_activity = app.STREAM.active_since

    def capture_snapshot(
        mode: str, image: Any, metrics: dict[str, Any]
    ) -> None:
        if mode != args.mode:
            raise RuntimeError(f"Unexpected benchmark snapshot mode: {mode}")
        captured.append(metrics)

    try:
        with (
            patch.object(app, "acquire_camera", return_value=True),
            patch.object(app, "release_camera"),
            patch.object(app, "read_active_mode", return_value=args.mode),
            patch.object(app, "read_confidence", return_value=args.confidence),
            patch.object(
                app,
                "read_navigation_config",
                return_value=app.NavigationConfig(
                    confidence_threshold=args.confidence
                ),
            ),
            patch.object(app, "write_camera_status"),
            patch.object(app, "write_snapshot", side_effect=capture_snapshot),
            patch.object(
                app,
                "mode_error",
                side_effect=lambda mode, owner, message, **kwargs: errors.append(
                    f"{mode}: {message}"
                ),
            ),
        ):
            processor = app.ObjectVideoProcessor(owner)

            def process_one() -> dict[str, Any]:
                snapshot_count = len(captured)
                processor.last_inference_at = 0.0
                image = sequence.next_frame()
                processor.recv(
                    av.VideoFrame.from_ndarray(image, format="bgr24")
                )
                if errors:
                    raise RuntimeError(errors[-1])
                if len(captured) != snapshot_count + 1:
                    raise RuntimeError(
                        "A benchmark frame did not produce exactly one processed snapshot."
                    )
                return captured[-1]

            for _ in range(args.warmup):
                process_one()

            sequence.rewind()
            processor.detection_stabilizer.reset()
            processor.navigation_engine.reset()
            processor.visual_memory.reset()
            processor.previous_frame = None
            app.reset_model_tracking(processor.tracking_model, args.mode)
            captured.clear()
            processor.frames_processed = {}
            processor.stream_started_at = app.time.time()

            samples = [process_one() for _ in range(args.samples)]
            model = processor.tracking_model
            model_name = Path(
                str(getattr(model, "ckpt_path", None) or model_path)
            ).name
            model_names = getattr(model, "names", None)
            model_classes = len(model_names) if model_names is not None else None
    finally:
        benchmark_error_active = sys.exc_info()[0] is not None
        _cleanup_benchmark(
            app,
            processor,
            sequence,
            saved_stream_state,
            preserve_active_exception=benchmark_error_active,
        )

    results = summarize_timings(samples)
    import ultralytics

    report = {
        "benchmark": "ObjectVision processed-frame latency",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "warmup_frames": args.warmup,
            "measured_frames": len(samples),
            "warmup_excluded_from_statistics": True,
            "percentile_method": "nearest rank",
            "inference_throttle_wait_included": False,
            "input_read_decode_and_resize_included": False,
            "frame_decode_quality_and_motion_analysis_included": False,
            "camera_wait_included": False,
            "snapshot_serialization_and_publication_included": False,
            "frame_processing_interval": (
                "from accepted inference processing start after frame decode, "
                "quality/motion analysis, throttling, and retry checks through "
                "operator assistance generation"
            ),
        },
        "environment": {
            "os": platform.platform(),
            "python": sys.version,
            "architecture": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "ultralytics": ultralytics.__version__,
        },
        "input": sequence.metadata(),
        "inference": {
            "mode": args.mode,
            "model_name": model_name,
            "model_file": model_path.name,
            "model_sha256": _sha256(model_path),
            "model_class_count": model_classes,
            "confidence": args.confidence,
            "device": "cpu",
            "tracker": "objectvision_bytetrack.yaml",
            "persist": True,
            "imgsz": "Ultralytics default; not explicitly passed",
        },
        "latency": results,
    }
    return add_threshold_validation(
        report,
        getattr(args, "latency_thresholds", {}),
    )


def main() -> None:
    args = _arguments()
    if args.compare:
        if args.input is not None or args.latency_thresholds:
            raise SystemExit(
                "--compare cannot be combined with --input or latency thresholds."
            )
        try:
            report = analyze_repeatability(load_benchmark_reports(args.compare))
        except ValueError as exc:
            raise SystemExit(f"benchmark comparison failed: {exc}") from exc
        if args.output is None:
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        write_report(args.output, report)
        print(f"Repeatability report written to {args.output}")
        return
    report = run_benchmark(args)
    if args.output is None:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    write_report(args.output, report)
    print(f"Benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
