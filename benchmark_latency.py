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
REAL_STREAM_TIMING_FIELDS = (
    "callback_to_snapshot_publication_ms",
    "snapshot_publication_ms",
)
REAL_STREAM_MEASUREMENT_DEFINITIONS = {
    "inference_latency_ms": (
        "model.track() call duration, excluding model-lock wait"
    ),
    "navigation_evaluation_latency_ms": (
        "NavigationEngine.evaluate() duration; null outside navigation modes"
    ),
    "frame_processing_latency_ms": (
        "accepted processing start through operator-assistance generation; "
        "excludes snapshot publication"
    ),
    "callback_to_snapshot_publication_ms": (
        "recv callback entry through successful atomic snapshot replacement"
    ),
    "snapshot_publication_ms": (
        "snapshot publication function entry through successful atomic "
        "replacement, including JPEG encoding"
    ),
    "dashboard_snapshot_read_ms": (
        "server-side read_snapshot file read and metadata parsing duration"
    ),
    "dashboard_render_call_ms": (
        "server-side Streamlit fragment Python call duration; excludes browser paint"
    ),
}
DASHBOARD_TIMING_FIELDS = (
    "dashboard_snapshot_read_ms",
    "dashboard_render_call_ms",
)
REPEATABILITY_TIMING_FIELDS = TIMING_FIELDS + REAL_STREAM_TIMING_FIELDS
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
    fields: tuple[str, ...] = TIMING_FIELDS,
) -> dict[str, dict[str, int | float | None]]:
    rows = list(samples)
    summary: dict[str, dict[str, int | float | None]] = {}
    for field in fields:
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

    for metric in REPEATABILITY_TIMING_FIELDS:
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
    dashboard_latency = report.get("dashboard_latency", {})
    if not isinstance(dashboard_latency, dict):
        raise ValueError(
            f"Benchmark report {report_index} has an invalid dashboard_latency object."
        )
    for metric in DASHBOARD_TIMING_FIELDS:
        summary = dashboard_latency.get(metric)
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
        ) or sample_count + unavailable != total:
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

    stream_telemetry = report.get("stream_telemetry")
    if (
        report["input"].get("kind") == "real_webrtc_phone_stream"
        and stream_telemetry is None
    ):
        raise ValueError(
            f"Benchmark report {report_index} is missing stream telemetry "
            "required for a real WebRTC report."
        )
    if stream_telemetry is not None:
        if not isinstance(stream_telemetry, dict):
            raise ValueError(
                f"Benchmark report {report_index} has invalid stream telemetry."
            )
        counters = stream_telemetry.get("counters")
        if not isinstance(counters, dict):
            raise ValueError(
                f"Benchmark report {report_index} is missing stream counters."
            )
        for name, count in counters.items():
            if (
                not isinstance(name, str)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    f"Benchmark report {report_index} has an invalid stream counter."
                )
        required_counters = set(STREAM_COUNTERS)
        if not required_counters.issubset(counters):
            raise ValueError(
                f"Benchmark report {report_index} is missing stream counters."
            )
        counter_definitions = stream_telemetry.get("counter_definitions")
        if (
            not isinstance(counter_definitions, dict)
            or set(counter_definitions) != required_counters
            or not all(
                isinstance(value, str) and value
                for value in counter_definitions.values()
            )
            or counter_definitions.get("frames_received")
            != "legacy count of valid frames counted after stream ownership check"
            or counter_definitions.get("frames_skipped")
            != "legacy count of empty or invalid frames only"
        ):
            raise ValueError(
                f"Benchmark report {report_index} has invalid counter definitions."
            )
        retention = stream_telemetry.get("sample_retention")
        if (
            not isinstance(retention, dict)
            or retention.get("frame_sample_capacity") != 1200
            or retention.get("dashboard_sample_capacity") != 600
            or not isinstance(retention.get("frame_samples_truncated"), bool)
            or not isinstance(retention.get("dashboard_samples_truncated"), bool)
        ):
            raise ValueError(
                f"Benchmark report {report_index} has invalid sample-retention metadata."
            )
        if (
            counters["valid_decoded_frames"] > counters["callback_frames_received"]
            or counters["valid_decoded_frames"]
            + counters["invalid_frames"]
            + counters["decode_failures"]
            > counters["callback_frames_received"]
            or counters["frames_received"] > counters["valid_decoded_frames"]
            or counters["accepted_for_processing"] > counters["frames_received"]
            or counters["inference_throttled_frames"]
            + counters["retry_suppressed_frames"]
            + counters["accepted_for_processing"]
            > counters["frames_received"]
            or counters["successfully_processed_frames"]
            + counters["processing_failures"]
            > counters["accepted_for_processing"]
            or counters["snapshot_publications_completed"]
            != counters["successfully_processed_frames"]
            or counters["invalid_frames"] + counters["decode_failures"]
            > counters["callback_frames_received"]
            or counters["snapshot_publication_failures"]
            > counters["processing_failures"]
        ):
            raise ValueError(
                f"Benchmark report {report_index} has inconsistent stream counters."
            )
        if counters["frames_skipped"] != counters["invalid_frames"]:
            raise ValueError(
                f"Benchmark report {report_index} changes the legacy "
                "frames_skipped counter meaning."
            )
        dashboard_sample_count = 0
        for metric in DASHBOARD_TIMING_FIELDS:
            summary = dashboard_latency.get(metric)
            if summary is None:
                raise ValueError(
                    f"Benchmark report {report_index} is missing {metric} "
                    "for stream telemetry."
                )
            dashboard_sample_count += summary["total_samples"]
        dashboard_event_count = (
            counters["dashboard_render_calls"]
            + counters["dashboard_snapshot_reads"]
        )
        if dashboard_sample_count > dashboard_event_count:
            raise ValueError(
                f"Benchmark report {report_index} has more dashboard samples "
                "than recorded dashboard events."
            )
        expected_frame_truncated = (
            counters["successfully_processed_frames"]
            > method["measured_frames"]
        )
        expected_dashboard_truncated = (
            dashboard_event_count > dashboard_sample_count
        )
        if (
            retention["frame_samples_truncated"] != expected_frame_truncated
            or retention["dashboard_samples_truncated"]
            != expected_dashboard_truncated
        ):
            raise ValueError(
                f"Benchmark report {report_index} has sample-retention flags "
                "that disagree with its counters and retained samples."
            )
        elapsed = stream_telemetry.get("rate_denominator_seconds")
        elapsed_seconds = stream_telemetry.get("elapsed_seconds")
        rates = stream_telemetry.get("rates")
        if not isinstance(rates, dict) or set(rates) != {
            "receive_rate_hz",
            "successful_processing_rate_hz",
        }:
            raise ValueError(
                f"Benchmark report {report_index} has invalid stream rates."
            )
        if elapsed_seconds != elapsed:
            raise ValueError(
                f"Benchmark report {report_index} has inconsistent elapsed-time "
                "denominators."
            )
        if elapsed is None:
            if any(value is not None for value in rates.values()):
                raise ValueError(
                    f"Benchmark report {report_index} has rates without a denominator."
                )
        elif (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError(
                f"Benchmark report {report_index} has an invalid rate denominator."
            )
        else:
            rate_counters = {
                "receive_rate_hz": "callback_frames_received",
                "successful_processing_rate_hz": "successfully_processed_frames",
            }
            for rate_name, counter_name in rate_counters.items():
                expected = counters.get(counter_name)
                actual = rates.get(rate_name)
                if expected is None or actual is None:
                    if actual is not None:
                        raise ValueError(
                            f"Benchmark report {report_index} has a rate without "
                            "its counter."
                        )
                elif (
                    isinstance(actual, bool)
                    or not isinstance(actual, (int, float))
                    or not math.isfinite(actual)
                    or actual < 0
                    or not math.isclose(
                        actual,
                        expected / elapsed,
                        rel_tol=1e-9,
                    )
                ):
                    raise ValueError(
                        f"Benchmark report {report_index} has an invalid "
                        f"{rate_name}."
                    )
    if report["input"].get("kind") == "real_webrtc_phone_stream":
        definitions = method.get("measurement_definitions")
        warmup = method.get("warmup_frames")
        if (
            method.get("measurement_definition")
            != "objectvision_real_stream_telemetry_v1"
            or method.get("sample_method")
            != "bounded successful snapshot records"
            or definitions != REAL_STREAM_MEASUREMENT_DEFINITIONS
            or isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or warmup < 1
            or method.get("warmup_excluded_from_statistics") is not True
            or report["input"].get("capture_configuration")
            != {
                "facing_mode": "ideal environment",
                "audio": False,
                "signalling": "HTTPS",
            }
        ):
            raise ValueError(
                f"Benchmark report {report_index} has invalid real-stream "
                "measurement definitions."
            )
    return report


def load_real_stream_telemetry(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Could not read real-stream telemetry {path}: {exc}") from exc
    return build_real_stream_report(payload)


STREAM_COUNTERS = (
    "callback_frames_received",
    "valid_decoded_frames",
    "frames_received",
    "frames_skipped",
    "invalid_frames",
    "decode_failures",
    "accepted_for_processing",
    "inference_throttled_frames",
    "retry_suppressed_frames",
    "processing_failures",
    "successfully_processed_frames",
    "snapshot_publications_completed",
    "snapshot_publication_failures",
    "dashboard_render_calls",
    "dashboard_snapshot_reads",
)


def build_real_stream_report(telemetry: Any) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        raise ValueError("Real-stream telemetry must be a JSON object.")
    if telemetry.get("schema_version") != 1:
        raise ValueError("Unsupported real-stream telemetry schema version.")
    if telemetry.get("source_type") != "real_webrtc_phone_stream":
        raise ValueError("Telemetry source_type is not a real WebRTC phone stream.")
    if telemetry.get("transport") != "WebRTC":
        raise ValueError("Real-stream telemetry must identify WebRTC transport.")
    environment = telemetry.get("environment")
    counters = telemetry.get("counters")
    samples = telemetry.get("samples")
    dashboard_samples = telemetry.get("dashboard_samples")
    if not isinstance(environment, dict) or not isinstance(counters, dict):
        raise ValueError("Telemetry is missing environment or counter metadata.")
    if not all(
        isinstance(environment.get(name), str) and environment.get(name)
        for name in ("os", "python", "architecture")
    ):
        raise ValueError("Telemetry has incomplete environment metadata.")
    cpu_count = environment.get("logical_cpu_count")
    if (
        isinstance(cpu_count, bool)
        or not isinstance(cpu_count, int)
        or cpu_count < 1
        or not isinstance(environment.get("ultralytics"), str)
        or not environment.get("ultralytics")
    ):
        raise ValueError("Telemetry has incomplete laptop environment metadata.")
    capture_configuration = telemetry.get("capture_configuration")
    if capture_configuration != {
        "facing_mode": "ideal environment",
        "audio": False,
        "signalling": "HTTPS",
    }:
        raise ValueError("Telemetry has incompatible phone capture configuration.")
    missing_counters = set(STREAM_COUNTERS) - set(counters)
    if missing_counters:
        raise ValueError(
            "Telemetry is missing counters: " + ", ".join(sorted(missing_counters))
        )
    if not isinstance(samples, list) or not samples:
        raise ValueError("Telemetry contains no successful snapshot samples.")
    if len(samples) > 1200:
        raise ValueError("Telemetry exceeds the bounded frame-sample capacity.")
    if not isinstance(dashboard_samples, list) or len(dashboard_samples) > 600:
        raise ValueError("Telemetry has invalid or excessive dashboard samples.")
    successful_frames = counters.get("successfully_processed_frames")
    if (
        isinstance(successful_frames, bool)
        or not isinstance(successful_frames, int)
        or successful_frames < len(samples)
    ):
        raise ValueError("Telemetry sample count exceeds successful-frame counter.")
    elapsed = telemetry.get("elapsed_seconds")
    denominator = telemetry.get("rate_denominator_seconds")
    if elapsed != denominator:
        raise ValueError("Telemetry elapsed time and rate denominator do not match.")
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError("Telemetry has an invalid monotonic elapsed-time denominator.")
    warmup_excluded = telemetry.get("warmup_excluded")
    warmup_frames_excluded = telemetry.get("warmup_frames_excluded")
    if not isinstance(warmup_excluded, bool):
        raise ValueError("Telemetry must state whether warm-up was excluded.")
    if (
        not warmup_excluded
        or isinstance(warmup_frames_excluded, bool)
        or not isinstance(warmup_frames_excluded, int)
        or warmup_frames_excluded < 1
    ):
        raise ValueError(
            "Warm-up has not been excluded. Reset telemetry after stream warm-up "
            "before exporting a measured run."
        )
    measurement_definitions = telemetry.get("measurement_definitions")
    counter_definitions = telemetry.get("counter_definitions")
    if (
        measurement_definitions != REAL_STREAM_MEASUREMENT_DEFINITIONS
        or not isinstance(counter_definitions, dict)
        or set(counter_definitions) != set(STREAM_COUNTERS)
        or not all(
            isinstance(value, str) and value
            for value in counter_definitions.values()
        )
        or counter_definitions.get("frames_received")
        != "legacy count of valid frames counted after stream ownership check"
        or counter_definitions.get("frames_skipped")
        != "legacy count of empty or invalid frames only"
    ):
        raise ValueError("Telemetry is missing or changes measurement definitions.")

    config: dict[str, Any] | None = None
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Telemetry frame sample {index} must be an object.")
        sample_config = {
            key: sample.get(key)
            for key in (
                "mode",
                "resolution",
                "confidence",
                "model_name",
                "model_sha256",
                "device",
                "tracker",
                "persist",
                "imgsz",
            )
        }
        if (
            sample_config["mode"] not in {"custom", "coco", "navigation"}
            or not isinstance(sample_config["resolution"], dict)
            or isinstance(sample_config["confidence"], bool)
            or not isinstance(sample_config["confidence"], (int, float))
            or not math.isfinite(sample_config["confidence"])
            or not isinstance(sample_config["model_name"], str)
            or not sample_config["model_name"]
            or not isinstance(sample_config["device"], str)
            or not isinstance(sample_config["tracker"], str)
            or not isinstance(sample_config["persist"], bool)
            or not isinstance(sample_config["imgsz"], str)
        ):
            raise ValueError(
                f"Telemetry frame sample {index} has invalid inference settings."
            )
        if config is None:
            config = sample_config
        elif sample_config != config:
            raise ValueError(
                "Telemetry contains mixed frame configurations; export one mode/"
                "model/settings per measurement."
            )

    if config is None:
        raise ValueError("Telemetry contains no inference configuration.")
    model_digest = config["model_sha256"]
    if model_digest is not None and (
        not isinstance(model_digest, str)
        or len(model_digest) != 64
        or any(character not in "0123456789abcdef" for character in model_digest.lower())
    ):
        raise ValueError("Telemetry contains an invalid model SHA-256 digest.")
    resolution = config["resolution"]
    width, height = resolution.get("width"), resolution.get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height < 1
    ):
        raise ValueError("Telemetry has invalid frame resolution metadata.")

    for index, sample in enumerate(samples, start=1):
        frame_id = sample.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(f"Telemetry frame sample {index} has an invalid frame ID.")
    latency = summarize_timings(samples, REPEATABILITY_TIMING_FIELDS)
    dashboard_values: dict[str, list[dict[str, Any]]] = {
        "dashboard_snapshot_read_ms": [],
        "dashboard_render_call_ms": [],
    }
    for index, sample in enumerate(dashboard_samples, start=1):
        if not isinstance(sample, dict) or sample.get("stage") not in {
            "snapshot_read",
            "render_call",
        }:
            raise ValueError(f"Dashboard telemetry sample {index} is invalid.")
        frame_id = sample.get("frame_id")
        if (
            not isinstance(sample.get("view"), str)
            or not sample["view"]
            or (
                frame_id is not None
                and (
                    isinstance(frame_id, bool)
                    or not isinstance(frame_id, int)
                    or frame_id < 0
                )
            )
        ):
            raise ValueError(
                f"Dashboard telemetry sample {index} has invalid correlation metadata."
            )
        elapsed = sample.get("elapsed_ms")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(
                f"Dashboard telemetry sample {index} has an invalid duration."
            )
        metric = (
            "dashboard_snapshot_read_ms"
            if sample["stage"] == "snapshot_read"
            else "dashboard_render_call_ms"
        )
        dashboard_values[metric].append({metric: elapsed})
    dashboard_latency = {}
    for metric, values in dashboard_values.items():
        durations = [float(sample[metric]) for sample in values]
        dashboard_latency[metric] = {
            "total_samples": len(durations),
            "sample_count": len(durations),
            "unavailable_samples": 0,
            "p50_ms": _nearest_rank(durations, 0.50),
            "p95_ms": _nearest_rank(durations, 0.95),
        }
    dashboard_coverage = sorted(
        {
            sample["view"]
            for sample in dashboard_samples
            if isinstance(sample, dict) and isinstance(sample.get("view"), str)
        }
    )
    stream_metadata = {
        "counters": counters,
        "elapsed_seconds": telemetry.get("elapsed_seconds"),
        "rate_denominator_seconds": telemetry.get("rate_denominator_seconds"),
        "rates": telemetry.get("rates"),
        "sample_retention": telemetry.get("sample_retention"),
        "warmup_excluded": warmup_excluded,
        "counter_definitions": counter_definitions,
    }
    report = {
        "benchmark": "ObjectVision real WebRTC stream performance telemetry",
        "generated_at_utc": telemetry.get("generated_at_utc"),
        "method": {
            "warmup_frames": warmup_frames_excluded,
            "measured_frames": len(samples),
            "warmup_excluded_from_statistics": warmup_excluded,
            "percentile_method": "nearest rank",
            "measurement_definition": "objectvision_real_stream_telemetry_v1",
            "sample_method": "bounded successful snapshot records",
            "measurement_definitions": measurement_definitions,
            "counter_definitions": counter_definitions,
            "dashboard_views_measured": dashboard_coverage,
        },
        "environment": environment,
        "input": {
            "kind": "real_webrtc_phone_stream",
            "transport": "WebRTC",
            "live": True,
            "resolution": {"width": width, "height": height},
            "capture_configuration": capture_configuration,
        },
        "inference": config,
        "latency": latency,
        "dashboard_latency": dashboard_latency,
        "stream_telemetry": stream_metadata,
    }
    _validated_report(report, 1)
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
    if (
        reference["input"].get("kind") == "real_webrtc_phone_stream"
        and not reference["inference"].get("model_sha256")
    ):
        raise ValueError(
            "Real-stream reports without a model SHA-256 digest cannot be "
            "compared for repeatability."
        )
    for index, report in enumerate(validated[1:], start=2):
        if (
            report["input"].get("kind") == "real_webrtc_phone_stream"
            and not report["inference"].get("model_sha256")
        ):
            raise ValueError(
                f"Benchmark report {index} has no model SHA-256 digest; "
                "repeatability cannot be verified."
            )
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
    for metric in REPEATABILITY_TIMING_FIELDS:
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

    dashboard_metrics: dict[str, Any] = {}
    for metric in DASHBOARD_TIMING_FIELDS:
        run_summaries = []
        p50_values = []
        p95_values = []
        for index, report in enumerate(validated, start=1):
            summary = report.get("dashboard_latency", {}).get(metric)
            if summary is None:
                sample_count = None
                unavailable_samples = None
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
        dashboard_metrics[metric] = {
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

    throughput: dict[str, Any] = {}
    rate_names = ("receive_rate_hz", "successful_processing_rate_hz")
    for rate_name in rate_names:
        runs = []
        available_values = []
        counter_name = (
            "callback_frames_received"
            if rate_name == "receive_rate_hz"
            else "successfully_processed_frames"
        )
        for index, report in enumerate(validated, start=1):
            stream = report.get("stream_telemetry")
            if stream is None:
                count = None
                denominator = None
                rate = None
            else:
                count = stream["counters"][counter_name]
                denominator = stream["rate_denominator_seconds"]
                rate = stream["rates"].get(rate_name)
            if rate is not None:
                available_values.append(float(rate))
            runs.append(
                {
                    "run": index,
                    "count": count,
                    "denominator_seconds": denominator,
                    "rate_hz": rate,
                }
            )
        throughput[rate_name] = {
            "runs": runs,
            "across_run_range_hz": (
                max(available_values) - min(available_values)
                if len(available_values) > 1
                else None
            ),
            "available_runs": len(available_values),
            "unavailable_runs": len(validated) - len(available_values),
        }

    return {
        "analysis": "ObjectVision benchmark repeatability",
        "configuration": {
            section: reference[section] for section in configuration_sections
        },
        "run_count": len(validated),
        "variability_definition": (
            "Range (maximum minus minimum) of each percentile across benchmark "
            "runs; this is not frame-to-frame variability. Dashboard values "
            "measure server-side calls, not browser paint."
        ),
        "metrics": metrics,
        "dashboard_metrics": dashboard_metrics,
        "throughput": throughput,
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
    parser.add_argument(
        "--telemetry",
        type=Path,
        help=(
            "Validate and summarize JSON telemetry exported by the Android "
            "phone's real WebRTC session."
        ),
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
    if args.telemetry is not None:
        if args.input is not None or args.compare:
            raise SystemExit("--telemetry cannot be combined with --input or --compare.")
        try:
            report = load_real_stream_telemetry(args.telemetry)
            report = add_threshold_validation(report, args.latency_thresholds)
        except ValueError as exc:
            raise SystemExit(f"real-stream telemetry rejected: {exc}") from exc
        if args.output is None:
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        write_report(args.output, report)
        print(f"Real-stream telemetry report written to {args.output}")
        return
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
