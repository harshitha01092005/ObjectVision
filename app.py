from __future__ import annotations

import json
import logging
import math
import os
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

from image_quality import analyze_image_quality
from detection_stability import (
    DetectionStabilityConfig,
    DetectionStabilizer,
    StableDetection,
)
from motion_analysis import estimate_camera_motion
from navigation import (
    Detection,
    NavigationEngine,
    NavigationConfig,
    RELEVANT_CLASSES,
    STOP,
    UNCERTAIN,
)
from operator_assistance import generate_operator_assistance
from visual_memory import VisualMemory


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"
COCO_MODEL_PATH = ROOT / "yolov8n.pt"
CLASS_NAMES = (
    "Battery", "Dice", "ToyCar", "Ants", "Bees", "Beetles", "Caterpillars",
    "Earthworms", "Earwigs", "Grasshoppers", "Moths", "Slugs", "Snails",
    "Wasps", "Weevils",
)
CLASS_LABELS = {name: name for name in CLASS_NAMES}
CLASS_COLORS = dict(
    zip(
        CLASS_NAMES,
        (
            "#38bdf8", "#f59e0b", "#f43f5e", "#22c55e", "#14b8a6",
            "#8b5cf6", "#ec4899", "#84cc16", "#06b6d4", "#eab308",
            "#ef4444", "#10b981", "#6366f1", "#f97316", "#a855f7",
        ),
    )
)
MODES = ("custom", "coco", "navigation")
MODE_LABELS = {
    "custom": "Custom Object Detection",
    "coco": "COCO Object Detection",
    "navigation": "Obstacle Detection & Navigation",
}
DETECTOR_OPTIONS = ("COCO", "Custom")
LOGGER = logging.getLogger("objectvision")
STATE_WRITE_LOCK = threading.Lock()


st.set_page_config(
    page_title="ObjectVision",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


@st.cache_resource
def load_coco_model() -> YOLO:
    if COCO_MODEL_PATH.exists():
        return YOLO(str(COCO_MODEL_PATH))
    return YOLO("yolov8n.pt")


def get_mode_model(mode: str) -> YOLO:
    if mode == "custom":
        return load_model()
    if mode in {"coco", "navigation"}:
        return load_coco_model()
    raise ValueError(f"Unsupported inference mode: {mode}")


@dataclass
class StreamState:
    frames: int = 0
    detections: int = 0
    class_counts: dict[str, int] | None = None
    latest_frame: Any = None
    last_update: float = 0.0
    active_owner: str | None = None
    active_since: float = 0.0
    last_activity: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = {name: 0 for name in CLASS_NAMES}


STREAM = StreamState()
STATE_DIR = ROOT / ".stream_state"
FRAME_PATH = STATE_DIR / "latest.jpg"
METRICS_PATH = STATE_DIR / "metrics.json"
CONFIDENCE_PATH = STATE_DIR / "confidence.json"
ACTIVE_MODE_PATH = STATE_DIR / "active_mode.json"
CAMERA_STATUS_PATH = STATE_DIR / "camera_status.json"
NAVIGATION_CONFIG_PATH = STATE_DIR / "navigation_config.json"
DEFAULT_CONFIDENCE = 0.35
MODEL_LOCKS = {"custom": threading.Lock(), "coco": threading.Lock()}


def mode_snapshot_path(mode: str) -> Path:
    return STATE_DIR / f"{mode}_latest.snapshot"


def write_json(path: Path, value: dict[str, Any]) -> None:
    with STATE_WRITE_LOCK:
        STATE_DIR.mkdir(exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(value))
        os.replace(temporary_path, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Could not read state file %s", path)
        return None
    return value if isinstance(value, dict) else None


def read_active_mode() -> str:
    state = read_json(ACTIVE_MODE_PATH)
    mode = state.get("mode") if state else "custom"
    return mode if mode in MODES else "custom"


def write_active_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Unsupported inference mode: {mode}")
    write_json(ACTIVE_MODE_PATH, {"mode": mode})


def read_navigation_config() -> NavigationConfig:
    state = read_json(NAVIGATION_CONFIG_PATH) or {}
    defaults = NavigationConfig()
    configured_classes = state.get("relevant_classes", sorted(RELEVANT_CLASSES))
    if not isinstance(configured_classes, list) or not all(
        isinstance(name, str) for name in configured_classes
    ):
        raise ValueError("Navigation relevant_classes must be a list of class names.")
    return NavigationConfig(
        confidence_threshold=read_confidence(),
        medium_area_fraction=float(
            state.get("medium_area_fraction", defaults.medium_area_fraction)
        ),
        high_area_fraction=float(
            state.get("high_area_fraction", defaults.high_area_fraction)
        ),
        high_height_fraction=float(
            state.get("high_height_fraction", defaults.high_height_fraction)
        ),
        medium_height_fraction=float(
            state.get("medium_height_fraction", defaults.medium_height_fraction)
        ),
        high_bottom_fraction=float(
            state.get("high_bottom_fraction", defaults.high_bottom_fraction)
        ),
        medium_growth_ratio=float(
            state.get("medium_growth_ratio", defaults.medium_growth_ratio)
        ),
        high_growth_ratio=float(
            state.get("high_growth_ratio", defaults.high_growth_ratio)
        ),
        blocked_region_score=float(
            state.get("blocked_region_score", defaults.blocked_region_score)
        ),
        side_clearance_margin=float(
            state.get("side_clearance_margin", defaults.side_clearance_margin)
        ),
        max_frame_age_seconds=defaults.max_frame_age_seconds,
        relevant_classes=frozenset(configured_classes),
    )


def mode_class_names(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def current_camera_status() -> dict[str, Any]:
    return read_json(CAMERA_STATUS_PATH) or {"status": "Waiting for camera connection"}


def write_camera_status(
    status: str,
    *,
    mode: str | None = None,
    owner: str | None = None,
    **details: Any,
) -> None:
    state: dict[str, Any] = {"status": status, "updated_at": time.time()}
    if mode:
        state["mode"] = mode
    if owner:
        state["owner"] = owner
    state.update(details)
    write_json(CAMERA_STATUS_PATH, state)


def write_snapshot(mode: str, image: Any, metrics: dict[str, Any]) -> None:
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode the annotated frame as JPEG.")
    metadata = json.dumps(metrics, separators=(",", ":")).encode("utf-8")
    snapshot = struct.pack(">Q", len(metadata)) + metadata + encoded.tobytes()
    path = mode_snapshot_path(mode)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    with STATE_WRITE_LOCK:
        STATE_DIR.mkdir(exist_ok=True)
        temporary.write_bytes(snapshot)
        os.replace(temporary, path)


def read_snapshot(mode: str) -> tuple[dict[str, Any], bytes] | None:
    path = mode_snapshot_path(mode)
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        LOGGER.exception("Could not read %s snapshot", mode)
        return None
    if len(payload) < 8:
        LOGGER.error("Snapshot for %s is truncated", mode)
        return None
    metadata_length = struct.unpack(">Q", payload[:8])[0]
    metadata_end = 8 + metadata_length
    if metadata_end >= len(payload):
        LOGGER.error("Snapshot for %s has invalid metadata or no image", mode)
        return None
    try:
        metrics = json.loads(payload[8:metadata_end])
    except (UnicodeDecodeError, json.JSONDecodeError):
        LOGGER.exception("Snapshot metadata for %s is invalid", mode)
        return None
    if not isinstance(metrics, dict):
        LOGGER.error("Snapshot metadata for %s is not an object", mode)
        return None
    return metrics, payload[metadata_end:]


def mode_error(
    mode: str,
    owner: str,
    message: str,
    *,
    frame_id: int = 0,
    frame_received_at: float | None = None,
    frames_received: int | None = None,
) -> None:
    LOGGER.error("%s inference error: %s", mode, message)
    write_camera_status(
        f"{mode} inference failed: {message}",
        mode=mode,
        owner=owner,
        inference_status="error",
        last_error=message,
        last_frame_id=frame_id,
        last_frame_received_at=frame_received_at,
        frames_received=frames_received,
    )


st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#f6f8fb !important; color:#172033 !important; }
    [data-testid="stHeader"] { background:rgba(246,248,251,.96) !important; }
    [data-testid="stSidebar"] { background:#101827 !important; border-right:1px solid #263449; }
    [data-testid="stSidebar"] * { color:#e5edf7 !important; }
    [data-testid="stSidebar"] hr { border-color:#334155 !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#a8b7ca !important; }
    .hero { background:linear-gradient(135deg,#0f172a 0%,#172554 100%); color:#fff !important; padding:28px 32px; border-radius:20px; margin:0 0 22px; }
    .hero h1 { margin:0; color:#fff !important; font-size:2.1rem; letter-spacing:-.04em; }
    .hero p { margin:.5rem 0 0; color:#cbd5e1 !important; font-size:1rem; }
    .section-title { color:#172033 !important; font-size:1.15rem; font-weight:700; margin:1rem 0 .5rem; }
    .stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stCaptionContainer"] p { color:#334155 !important; }
    a { color:#2563eb !important; font-weight:600; }
    .status { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; font-weight:600; font-size:.85rem; }
    .online { color:#166534 !important; background:#dcfce7; }
    .offline { color:#92400e !important; background:#fef3c7; }
    .hint { color:#64748b !important; font-size:.9rem; }
    div[data-testid="stMetric"] { background:#ffffff !important; border:1px solid #d7e0eb; padding:14px 16px; border-radius:14px; box-shadow:0 2px 8px rgba(15,23,42,.06); }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color:#526176 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#0f172a !important; }
    div[data-testid="stAlert"] { color:#334155 !important; }
    div[data-testid="stAlert"] p { color:#334155 !important; }
    .stButton > button, .stDownloadButton > button { color:#ffffff !important; background:#2563eb !important; border:0 !important; }
    .stButton > button:hover, .stDownloadButton > button:hover { background:#1d4ed8 !important; color:#ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_confidence() -> float:
    try:
        value = float(json.loads(CONFIDENCE_PATH.read_text())["confidence"])
        return min(0.95, max(0.05, value))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_CONFIDENCE


def snapshot_uses_current_confidence(metrics: dict[str, Any]) -> bool:
    snapshot_confidence = metrics.get("confidence")
    return (
        isinstance(snapshot_confidence, (int, float))
        and math.isclose(float(snapshot_confidence), read_confidence(), abs_tol=1e-9)
    )


def write_confidence(value: float) -> None:
    write_json(CONFIDENCE_PATH, {"confidence": value})


def acquire_camera(owner: str) -> bool:
    now = time.time()
    with STREAM.lock:
        if STREAM.active_owner and STREAM.active_owner != owner:
            lease_time = max(
                STREAM.last_update, STREAM.active_since, STREAM.last_activity
            )
            if now - lease_time < 15:
                return False
        STREAM.active_owner = owner
        STREAM.active_since = now
        STREAM.last_activity = now
    LOGGER.info("Camera processor initialized; waiting for an incoming video frame")
    write_camera_status(
        "WebRTC processor initialized; waiting for a server-side video frame",
        owner=owner,
        frames_received=0,
        inference_status="waiting for first frame",
    )
    return True


def release_camera(owner: str) -> None:
    released = False
    last_status: dict[str, Any] = {}
    with STREAM.lock:
        if STREAM.active_owner == owner:
            STREAM.active_owner = None
            STREAM.active_since = 0.0
            STREAM.last_activity = 0.0
            released = True
            last_status = current_camera_status()
    if released:
        LOGGER.info("Camera video track ended")
        write_camera_status(
            "Camera stream stopped or disconnected",
            mode=last_status.get("mode"),
            owner=owner,
            inference_status="stopped",
            frames_received=last_status.get("frames_received", 0),
            last_frame_id=last_status.get("last_frame_id"),
            last_frame_received_at=last_status.get("last_frame_received_at"),
        )


def draw_navigation_overlay(
    image: Any,
    recommendation: str,
) -> Any:
    height, width = image.shape[:2]
    overlay = image.copy()
    for x in (width // 3, 2 * width // 3):
        cv2.line(overlay, (x, 0), (x, height), (210, 210, 210), 1, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    for index, label in enumerate(("LEFT", "CENTER", "RIGHT")):
        cv2.putText(
            image,
            label,
            (index * width // 3 + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    short_recommendations = {
        "STOP — OBSTACLE VERY CLOSE": "STOP — OBSTACLE CLOSE",
        "GO LEFT — LEFT PATH APPEARS CLEARER": "GO LEFT",
        "GO RIGHT — RIGHT PATH APPEARS CLEARER": "GO RIGHT",
        "MOVE FORWARD CAUTIOUSLY — CENTER PATH APPEARS CLEAR": "MOVE FORWARD CAUTIOUSLY",
        "UNCERTAIN — CHECK SURROUNDINGS": "UNCERTAIN — CHECK",
    }
    banner_text = short_recommendations.get(recommendation, recommendation)
    font_scale = min(0.55, max(0.25, width / 900))
    (text_width, text_height), baseline = cv2.getTextSize(
        banner_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        1,
    )
    max_text_width = max(1, width - 24)
    while text_width > max_text_width and font_scale > 0.2:
        font_scale -= 0.025
        (text_width, text_height), baseline = cv2.getTextSize(
            banner_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
    banner_height = text_height + baseline + 12
    cv2.rectangle(
        image,
        (0, max(0, height - banner_height)),
        (width - 1, height - 1),
        (15, 23, 42),
        -1,
    )
    cv2.putText(
        image,
        banner_text,
        (12, max(text_height, height - baseline - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def draw_detection_annotations(
    image: Any,
    detections: list[StableDetection],
) -> Any:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    placed_labels: list[tuple[int, int, int, int]] = []
    for detection in detections:
        x1 = min(width - 1, max(0, int(round(detection.display_box[0]))))
        y1 = min(height - 1, max(0, int(round(detection.display_box[1]))))
        x2 = min(width - 1, max(x1 + 1, int(round(detection.display_box[2]))))
        y2 = min(height - 1, max(y1 + 1, int(round(detection.display_box[3]))))
        label = f"{detection.label} {detection.smoothed_confidence:.0%}"
        font_scale = 0.45
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        while text_width > width - 8 and font_scale > 0.25:
            font_scale -= 0.025
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
        if text_width > width - 8:
            label = detection.label[: max(1, len(detection.label) // 2)] + "..."
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

        label_height = text_height + baseline + 6
        candidate_ys = [y1 - label_height, y1 + 2, y2 + 2]
        candidate_ys.extend(
            y1 - label_height - step * label_height
            for step in range(1, max(2, height // max(label_height, 1)))
        )
        label_y = None
        label_x = min(max(2, x1), max(2, width - text_width - 4))
        for candidate_y in candidate_ys:
            top = min(max(0, candidate_y), max(0, height - label_height))
            label_rect = (
                label_x - 2,
                top,
                min(width, label_x + text_width + 4),
                min(height, top + label_height),
            )
            if not any(
                label_rect[0] < other[2]
                and label_rect[2] > other[0]
                and label_rect[1] < other[3]
                and label_rect[3] > other[1]
                for other in placed_labels
            ):
                label_y = top
                break
        if label_y is None:
            label_y = min(max(0, y1 - label_height), max(0, height - label_height))
        label_rect = (
            label_x - 2,
            label_y,
            min(width, label_x + text_width + 4),
            min(height, label_y + label_height),
        )
        placed_labels.append(label_rect)
        color_seed = zlib.crc32(detection.label.encode("utf-8"))
        color = (
            80 + color_seed % 120,
            110 + (color_seed >> 8) % 100,
            150 + (color_seed >> 16) % 90,
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            annotated,
            (label_rect[0], label_rect[1]),
            (label_rect[2], label_rect[3]),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x, min(height - baseline - 1, label_y + text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def reset_model_tracking(model: Any | None, mode: str | None = None) -> None:
    if model is None:
        return

    def reset() -> None:
        predictor = getattr(model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    if mode in MODEL_LOCKS:
        with MODEL_LOCKS[mode]:
            reset()
    else:
        reset()


def gate_navigation_evidence(
    result: Any,
    *,
    image_quality_status: str,
) -> Any:
    if not result.frame_fresh or result.recommendation == STOP:
        return result
    if image_quality_status == "POOR":
        reason = "Image quality is poor; obstacle evidence is not reliable enough for directional guidance."
    else:
        return result
    return replace(
        result,
        recommendation=UNCERTAIN,
        explanation=reason,
        valid_evidence=False,
    )


class ObjectVideoProcessor(VideoProcessorBase):
    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.allowed = acquire_camera(owner)
        if not self.allowed:
            write_camera_status(
                "Another phone camera stream is active. Stop that stream before connecting this one."
            )
        self.active_mode: str | None = None
        self.navigation_config = read_navigation_config()
        self.navigation_engine = NavigationEngine(self.navigation_config)
        self.visual_memory = VisualMemory()
        self.detection_stability_config = DetectionStabilityConfig(
            tracker_config=str(ROOT / "objectvision_bytetrack.yaml")
        )
        self.detection_stabilizer = DetectionStabilizer(
            self.detection_stability_config
        )
        self.tracking_model: Any | None = None
        self.previous_frame: Any | None = None
        self.model_retry_after: dict[str, float] = {}
        self.last_inference_at = 0.0
        self.last_status_at = 0.0
        self.frames_received = 0
        self.frames_processed: dict[str, int] = {}
        self.frames_skipped = 0
        self.stream_started_at = time.time()
        self.history: dict[str, list[dict[str, Any]]] = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_received_at = time.time()
        if not self.allowed:
            return frame
        try:
            image = frame.to_ndarray(format="bgr24")
        except Exception as exc:
            LOGGER.exception("Could not decode an incoming WebRTC video frame")
            mode_error(
                read_active_mode(),
                self.owner,
                f"Could not decode incoming video frame: {exc}",
            )
            return frame
        if (
            image is None
            or image.size == 0
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            self.frames_skipped += 1
            LOGGER.error("Received an empty or invalid WebRTC video frame")
            mode_error(
                read_active_mode(),
                self.owner,
                "Received an empty or invalid video frame.",
            )
            return frame
        quality = analyze_image_quality(image)
        motion = estimate_camera_motion(self.previous_frame, image)

        with STREAM.lock:
            if STREAM.active_owner != self.owner:
                return av.VideoFrame.from_ndarray(image, format="bgr24")
            STREAM.last_activity = frame_received_at
            STREAM.frames += 1
            frame_id = STREAM.frames
        self.frames_received += 1
        if self.frames_received == 1 or self.frames_received % 60 == 0:
            LOGGER.info(
                "Received WebRTC video frame id=%s size=%sx%s",
                frame_id,
                image.shape[1],
                image.shape[0],
            )

        mode = read_active_mode()
        if mode != self.active_mode:
            reset_model_tracking(self.tracking_model, self.active_mode)
            self.tracking_model = None
            self.detection_stabilizer.reset()
            self.active_mode = mode
            self.navigation_engine.reset()
        if frame_received_at - self.last_inference_at < 0.2:
            if frame_received_at - self.last_status_at >= 0.5:
                self.last_status_at = frame_received_at
                write_camera_status(
                    "Phone video frames are arriving; waiting for the next inference",
                    mode=mode,
                    owner=self.owner,
                    frames_received=self.frames_received,
                    last_frame_id=frame_id,
                    last_frame_received_at=frame_received_at,
                    inference_status="frame received",
                )
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        confidence = read_confidence()
        if time.time() < self.model_retry_after.get(mode, 0):
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        self.last_inference_at = frame_received_at
        write_camera_status(
            "Server received a phone video frame; YOLO inference is running",
            mode=mode,
            owner=self.owner,
            frames_received=self.frames_received,
            last_frame_id=frame_id,
            last_frame_received_at=frame_received_at,
            inference_status="running",
        )
        try:
            if mode in {"coco", "navigation"}:
                config = read_navigation_config()
                if config != self.navigation_config:
                    self.navigation_config = config
                    self.navigation_engine = NavigationEngine(config)
            model = get_mode_model(mode)
            if model is not self.tracking_model:
                reset_model_tracking(model, mode)
                self.tracking_model = model
                self.detection_stabilizer.reset()
            inference_started_at = time.time()
            with MODEL_LOCKS["custom" if mode == "custom" else "coco"]:
                result = model.track(
                    source=image,
                    conf=confidence,
                    verbose=False,
                    device="cpu",
                    persist=True,
                    tracker=self.detection_stability_config.tracker_config,
                )[0]
            inference_latency_ms = (time.time() - inference_started_at) * 1000.0
            names = mode_class_names(model)
            boxes = result.boxes
            class_counts: dict[str, int] = {}
            detection_inputs: list[Detection] = []
            tracked_inputs: list[
                tuple[int | None, str, float, tuple[float, float, float, float]]
            ] = []
            if boxes is not None:
                track_ids = boxes.id.tolist() if boxes.id is not None else None
                class_ids = boxes.cls.tolist()
                confidence_values = boxes.conf.tolist()
                coordinates_list = boxes.xyxy.tolist()
                ids = track_ids if track_ids is not None else [None] * len(class_ids)
                for class_id, confidence_value, coordinates, track_id_value in zip(
                    class_ids, confidence_values, coordinates_list, ids
                ):
                    raw_confidence = float(confidence_value)
                    box = tuple(float(value) for value in coordinates)
                    if (
                        not math.isfinite(raw_confidence)
                        or raw_confidence < confidence
                        or len(box) != 4
                        or not all(math.isfinite(value) for value in box)
                        or box[2] <= box[0]
                        or box[3] <= box[1]
                    ):
                        continue
                    name = names.get(int(class_id), str(int(class_id)))
                    class_counts[name] = class_counts.get(name, 0) + 1
                    detection_inputs.append(
                        Detection(
                            name,
                            raw_confidence,
                            *box,
                        )
                    )
                    tracked_inputs.append(
                        (
                            int(track_id_value) if track_id_value is not None else None,
                            name,
                            raw_confidence,
                            box,
                        )
                    )
            stabilized_detections = self.detection_stabilizer.update(
                tracked_inputs,
                timestamp=frame_received_at,
            )
            annotated = draw_detection_annotations(image, stabilized_detections)
            detection_count = sum(class_counts.values())
            navigation_result = None
            if mode in {"coco", "navigation"}:
                height, width = image.shape[:2]
                frame_age = max(0.0, time.time() - frame_received_at)
                fresh_frame = frame_age <= self.navigation_config.max_frame_age_seconds
                guidance_input = detection_inputs if fresh_frame else []
                navigation_result = self.navigation_engine.evaluate(
                    guidance_input,
                    width,
                    height,
                    frame_received_at if fresh_frame else None,
                    now=time.time(),
                )
                if not fresh_frame:
                    navigation_result = self.navigation_engine.evaluate(
                        [],
                        width,
                        height,
                        None,
                        now=time.time(),
                    )
                navigation_result = gate_navigation_evidence(
                    navigation_result,
                    image_quality_status=quality.status,
                )
                annotated = draw_navigation_overlay(
                    annotated,
                    navigation_result.recommendation,
                )
            cv2.putText(
                annotated,
                f"Objects: {detection_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            now = time.time()
            with STREAM.lock:
                STREAM.detections = detection_count
                STREAM.class_counts = class_counts
                STREAM.latest_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                STREAM.last_update = now

            self.frames_processed[mode] = self.frames_processed.get(mode, 0) + 1
            mode_history = self.history.setdefault(mode, [])
            mode_history.append({"time": now, **class_counts})
            del mode_history[:-30]

            memory_snapshot = self.visual_memory.update(
                detection_inputs,
                timestamp=frame_received_at,
                image_quality=quality,
                motion=motion,
            )
            assistance = generate_operator_assistance(
                detection_inputs,
                quality,
                motion,
                self.visual_memory,
                frame_fresh=True,
                camera_connected=True,
            )

            frame_age_seconds = max(0.0, now - frame_received_at)
            fps = (
                self.frames_processed[mode] / max(now - self.stream_started_at, 1.0)
                if self.stream_started_at
                else 0.0
            )
            metrics: dict[str, Any] = {
                "mode": mode,
                "frame_id": frame_id,
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed[mode],
                "frames_skipped": self.frames_skipped,
                "detections": detection_count,
                "objects": [
                    {
                        "name": stable.label,
                        "confidence": stable.raw_confidence,
                        "raw_confidence": stable.raw_confidence,
                        "smoothed_confidence": stable.smoothed_confidence,
                        "track_id": stable.track_id,
                        "box": list(stable.display_box),
                        "raw_box": list(stable.raw_box),
                    }
                    for stable in stabilized_detections
                ],
                "class_counts": class_counts,
                "last_update": now,
                "frame_received_at": frame_received_at,
                "frame_age_seconds": frame_age_seconds,
                "frame_fresh": frame_age_seconds <= self.navigation_config.max_frame_age_seconds,
                "history": mode_history.copy(),
                "confidence": confidence,
                "confidence_smoothing_alpha": self.detection_stability_config.confidence_alpha,
                "tracker": self.detection_stability_config.tracker_config,
                "inference_latency_ms": round(inference_latency_ms, 2),
                "fps": round(fps, 2),
                "inference_status": (
                    "detections found" if detection_count else "no detections"
                ),
                "image_quality": quality.to_dict(),
                "motion": motion.to_dict(),
                "visual_memory": memory_snapshot,
                "operator_assistance": assistance.to_dict(),
            }
            if navigation_result is not None:
                metrics["navigation"] = navigation_result.to_dict()
            self.previous_frame = image.copy()
            write_snapshot(mode, annotated, metrics)
            self.model_retry_after.pop(mode, None)
            write_camera_status(
                (
                    "Inference complete; objects detected"
                    if detection_count
                    else "Inference complete; no objects detected in this frame"
                ),
                mode=mode,
                owner=self.owner,
                frames_received=self.frames_received,
                last_frame_id=frame_id,
                last_frame_received_at=frame_received_at,
                inference_status=metrics["inference_status"],
                detections=detection_count,
                last_inference_at=now,
                frame_age_seconds=frame_age_seconds,
                inference_latency_ms=metrics["inference_latency_ms"],
                fps=metrics["fps"],
            )
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as exc:
            LOGGER.exception(
                "Failed to process WebRTC frame id=%s in mode=%s", frame_id, mode
            )
            self.model_retry_after[mode] = time.time() + 30
            mode_error(
                mode,
                self.owner,
                f"{exc}. Check the selected model, dependencies, and network access for first-run COCO weights.",
                frame_id=frame_id,
                frame_received_at=frame_received_at,
                frames_received=self.frames_received,
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    def on_ended(self) -> None:
        reset_model_tracking(self.tracking_model, self.active_mode)
        self.tracking_model = None
        self.detection_stabilizer.reset()
        release_camera(self.owner)


def render_brand(view: str) -> None:
    label = {
        "phone": "Android camera sender",
        "dashboard": "Operations dashboard",
        "custom": "Custom object detection",
        "coco": "COCO object detection",
        "navigation": "Obstacle detection & navigation",
        "live": "Live detection & navigation",
    }.get(view, "Operations dashboard")
    st.markdown(f'<div class="hero"><h1>🔎 ObjectVision</h1><p>Live object detection · {label}</p></div>', unsafe_allow_html=True)


def clear_stream_state() -> None:
    for path in (
        FRAME_PATH,
        METRICS_PATH,
        CONFIDENCE_PATH,
        CAMERA_STATUS_PATH,
        mode_snapshot_path("custom"),
        mode_snapshot_path("coco"),
        mode_snapshot_path("navigation"),
        STATE_DIR / "coco_latest.jpg",
        STATE_DIR / "coco_metrics.json",
        STATE_DIR / "navigation_latest.jpg",
        STATE_DIR / "navigation_metrics.json",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def render_sidebar(view: str) -> float:
    with st.sidebar:
        st.markdown("## ObjectVision")
        st.caption("AI-assisted object detection")
        st.divider()
        st.markdown("### Navigation")
        st.markdown('[📊 Dashboard](?view=dashboard)')
        st.markdown('[🧪 Custom Object Detection](?view=custom)')
        st.markdown('[🌐 COCO Object Detection](?view=coco)')
        st.markdown('[🧭 Obstacle Detection & Navigation](?view=navigation)')
        st.markdown('[📡 Live Detection & Navigation](?view=live)')
        st.markdown('[📱 Android Camera](?view=phone)')
        st.divider()
        st.markdown("### Active camera inference")
        st.caption(f"Selected on Android Camera: {MODE_LABELS[read_active_mode()]}")
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=read_confidence(),
            step=0.05,
            help="Higher values show fewer but more certain detections.",
            key=f"confidence-{view}",
        )
        write_confidence(confidence)
        st.caption(f"Current threshold: {confidence:.2f}")
        if view != "phone":
            if st.button("Clear session data", use_container_width=True):
                clear_stream_state()
                st.toast("Session data cleared")
                st.rerun()
        st.divider()
        st.caption("Keep one Android Camera page open while viewing results on this laptop.")
    return confidence


@st.fragment(run_every="1s")
def show_camera_status(
    owner: str | None = None,
    frontend_playing: bool = False,
    frontend_signalling: bool = False,
) -> None:
    status = current_camera_status()
    text = status.get("status", "Waiting for camera connection")
    if owner and status.get("owner") not in (None, owner):
        st.warning("Another camera sender owns the active stream. Stop it before starting this sender.")
        return
    last_frame = float(status.get("last_frame_received_at", 0) or 0)
    age = time.time() - last_frame if last_frame else None
    if status.get("inference_status") == "error":
        st.error(f"Server-side inference error: {status.get('last_error', text)}")
    elif status.get("inference_status") == "stopped":
        detail = f" Last server frame was {age:.1f}s ago." if age is not None else ""
        st.warning(f"Camera stream is stopped or disconnected.{detail}")
    elif age is not None and age <= 5:
        st.success(
            f"Frames received by laptop: {status.get('frames_received', 0)} · "
            f"last received {age:.1f}s ago · inference: "
            f"{status.get('inference_status', 'pending')}"
        )
    elif status.get("inference_status") == "error":
        st.error(f"Server-side inference error: {status.get('last_error', text)}")
    elif last_frame:
        st.warning(f"Video frames stopped arriving {age:.0f}s ago: {text}")
    elif frontend_signalling:
        st.warning(
            "WebRTC signaling is in progress, but the laptop has not received a video frame. "
            "The phone's local preview alone does not confirm transmission."
        )
    elif frontend_playing:
        st.warning(
            "The sender is active, but no video frame has reached the laptop yet. "
            "Check ICE/network access and the laptop firewall."
        )
    elif text == "Camera stream stopped or disconnected":
        st.info("Camera stream is stopped. Tap START on the phone to reconnect.")
    else:
        st.info(f"Waiting for phone camera connection. {text}")


def show_phone_view() -> None:
    render_brand("phone")
    st.markdown('<div class="section-title">Android camera uplink</div>', unsafe_allow_html=True)
    selected_mode = st.selectbox(
        "Inference mode for the shared camera stream",
        MODES,
        index=MODES.index(read_active_mode()),
        format_func=lambda mode: MODE_LABELS[mode],
        help="Only one inference mode processes the camera at a time. Change modes here; results remain separate.",
    )
    write_active_mode(selected_mode)
    st.info(
        "Tap START, grant camera permission, and keep this page open. "
        "The laptop runs inference; this phone only streams camera frames."
    )
    owner = st.session_state.setdefault("camera-owner", str(uuid.uuid4()))
    context = webrtc_streamer(
        key="objectvision-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: ObjectVideoProcessor(owner),
        media_stream_constraints={"video": {"facingMode": {"ideal": "environment"}}, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )
    show_camera_status(
        owner,
        context.state.playing,
        context.state.signalling,
    )
    st.caption(
        "WebRTC UI state: "
        f"{'sender active' if context.state.playing else 'stopped'}, "
        f"{'signaling in progress' if context.state.signalling else 'no pending signaling'}."
    )
    st.markdown(
        '<p class="hint">Use Android Chrome and the rear-facing camera. '
        'Stop the current stream before opening another camera sender tab.</p>',
        unsafe_allow_html=True,
    )


def render_stream_health(mode: str, metrics: dict[str, Any] | None) -> None:
    status = current_camera_status()
    if status.get("inference_status") == "error" and status.get("mode") == mode:
        st.error(f"Server-side inference error: {status.get('last_error', status.get('status'))}")
        received_at = float(status.get("last_frame_received_at", 0) or 0)
        if received_at and time.time() - received_at <= 5:
            st.warning("Frames may be arriving, but the most recent inference did not complete.")
        return
    if status.get("inference_status") == "stopped" and status.get("mode") == mode:
        received_at = float(status.get("last_frame_received_at", 0) or 0)
        detail = (
            f" The last frame arrived {time.time() - received_at:.1f}s ago."
            if received_at
            else ""
        )
        st.warning(f"Camera stream is stopped or disconnected.{detail}")
        return
    status_mode = status.get("mode")
    received_at = float(status.get("last_frame_received_at", 0) or 0)
    age = time.time() - received_at if received_at else None
    if status_mode == mode and age is not None and age <= 5:
        st.success(
            f"Phone video received by laptop · {status.get('frames_received', 0)} frames · "
            f"last frame {age:.1f}s ago · inference: "
            f"{status.get('inference_status', 'waiting')}"
        )
    elif status_mode and status_mode != mode and age is not None and age <= 5:
        st.info(
            f"Camera frames are currently being processed as {MODE_LABELS.get(status_mode, status_mode)}; "
            f"this panel is waiting for {MODE_LABELS.get(mode, mode)}."
        )
    elif age is not None and age > 5:
        st.warning(f"Camera frames are stale; the last server frame arrived {age:.0f}s ago.")
    elif metrics:
        st.warning("Showing the last processed frame; no current phone frames are reaching the laptop.")
    else:
        st.info(
            "Waiting for the laptop to receive a phone video frame. A local phone preview is not proof of transmission."
        )


@st.fragment(run_every="1s")
def show_dashboard() -> None:
    snapshot = read_snapshot("custom")
    if snapshot is None:
        render_stream_health("custom", None)
        st.warning("No custom-model camera results yet.")
        st.markdown(
            "1. Open **Android Camera**.  \n"
            "2. Select **Custom Object Detection**, tap **START**, and allow camera access.  \n"
            "3. Keep the phone tab open while viewing the laptop dashboard."
        )
        return
    metrics, frame = snapshot

    render_stream_health("custom", metrics)
    if not snapshot_uses_current_confidence(metrics):
        st.warning(
            f"This frame used confidence threshold {float(metrics['confidence']):.2f}; "
            f"the current threshold is {read_confidence():.2f}. Waiting for a matching inference."
        )
        return

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.image(frame, alt="Latest server-annotated custom detector frame", width="stretch")
        st.caption(
            "Overlay confidence is smoothed for tracked objects; acceptance uses raw confidence."
        )
        st.download_button(
            "Download annotated frame",
            data=frame,
            file_name="objectvision-latest.jpg",
            mime="image/jpeg",
            width="stretch",
        )
    with right:
        class_counts = {name: int(value) for name, value in metrics["class_counts"].items()}
        st.markdown('<div class="section-title">Live snapshot</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Objects detected", int(metrics["detections"]))
        c2.metric("Frames processed", int(metrics.get("frames_processed", 0)))
        st.markdown('<div class="section-title">Visual intelligence</div>', unsafe_allow_html=True)
        quality = metrics.get("image_quality", {})
        motion = metrics.get("motion", {})
        assist = metrics.get("operator_assistance", {})
        st.metric("Blur score", round(float(quality.get("blur_score", 0.0)), 2))
        st.metric("Image quality", quality.get("status", "UNKNOWN"))
        st.metric("Brightness", round(float(quality.get("brightness", 0.0)), 2))
        st.metric("Contrast", round(float(quality.get("contrast", 0.0)), 2))
        st.metric("Camera motion", motion.get("status", "UNKNOWN"))
        st.metric("Motion magnitude", round(float(motion.get("magnitude", 0.0)), 4))
        st.metric("Frame age", round(float(metrics.get("frame_age_seconds", 0.0)), 2))
        st.metric("Inference latency", round(float(metrics.get("inference_latency_ms", 0.0)), 2))
        st.metric("FPS", round(float(metrics.get("fps", 0.0)), 2))
        st.markdown('<div class="section-title">Detection breakdown</div>', unsafe_allow_html=True)
        for name in CLASS_NAMES:
            st.metric(CLASS_LABELS[name], class_counts.get(name, 0))

    st.markdown('<div class="section-title">Operator assistance</div>', unsafe_allow_html=True)
    if assist:
        for entry in assist.get("warnings", []):
            st.warning(entry)
        for entry in assist.get("suggestions", []):
            st.info(entry)
    st.markdown('<div class="section-title">Detection activity</div>', unsafe_allow_html=True)
    history = metrics.get("history", [])
    if history:
        chart_data = [{"time": time.strftime("%H:%M:%S", time.localtime(row["time"])), **{CLASS_LABELS[n]: row.get(n, 0) for n in CLASS_NAMES}} for row in history]
        st.line_chart(chart_data, x="time", y=[CLASS_LABELS[n] for n in CLASS_NAMES], color=[CLASS_COLORS[n] for n in CLASS_NAMES])
    st.caption(
        f"Model: {MODEL_PATH.name} · frame {metrics['frame_id']} · "
        f"frame received {time.strftime('%H:%M:%S', time.localtime(metrics['frame_received_at']))} · "
        f"confidence {metrics['confidence']:.2f} · auto-refreshes every second"
    )


@st.fragment(run_every="1s")
def show_coco_view() -> None:
    snapshot = read_snapshot("coco")
    if snapshot is None:
        render_stream_health("coco", None)
        st.warning(
            "No COCO results yet. On **Android Camera**, select **COCO Object Detection** and start the stream."
        )
        return
    metrics, frame = snapshot
    render_stream_health("coco", metrics)
    if not snapshot_uses_current_confidence(metrics):
        st.warning(
            f"This frame used confidence threshold {float(metrics['confidence']):.2f}; "
            f"the current threshold is {read_confidence():.2f}. Waiting for a matching inference."
        )
        return
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.image(frame, alt="Latest server-annotated COCO frame", width="stretch")
        st.caption(
            "Overlay confidence is smoothed for tracked objects; acceptance uses raw confidence."
        )
        st.download_button(
            "Download annotated COCO frame",
            frame,
            file_name="objectvision-coco-latest.jpg",
            mime="image/jpeg",
            width="stretch",
        )
    with right:
        st.metric("Objects detected", int(metrics.get("detections", 0)))
        st.metric(
            "Last frame received",
            time.strftime("%H:%M:%S", time.localtime(metrics["frame_received_at"])),
        )
        st.markdown('<div class="section-title">Counts by detected class</div>', unsafe_allow_html=True)
        counts = metrics.get("class_counts", {})
        if counts:
            for name, count in sorted(counts.items()):
                st.write(f"**{name}:** {count}")
        else:
            st.write("No objects detected in the latest frame.")
    st.caption(
        f"COCO model: {COCO_MODEL_PATH.name} · frame {metrics['frame_id']} · "
        f"confidence {metrics['confidence']:.2f} · CPU inference · auto-refreshes every second"
    )


@st.fragment(run_every="1s")
def show_live_detection_view(mode: str) -> None:
    snapshot = read_snapshot(mode)
    left_panel, right_panel = st.columns([1.15, 1], gap="large")
    metrics: dict[str, Any] | None = None
    navigation: dict[str, Any] | None = None
    fresh = False

    with left_panel:
        st.subheader("Live Object Detection")
        st.caption(
            "Selected detector: "
            f"{'COCO (yolov8n.pt)' if mode == 'coco' else 'Custom (best.pt)'}"
        )
        if snapshot is None:
            render_stream_health(mode, None)
            st.info("Waiting for a server-processed phone frame.")
        else:
            metrics, frame = snapshot
            age = max(0.0, time.time() - float(metrics["frame_received_at"]))
            status = current_camera_status()
            threshold_matches = snapshot_uses_current_confidence(metrics)
            fresh = (
                age <= 5
                and threshold_matches
                and status.get("mode") == mode
                and status.get("inference_status") not in {"error", "stopped"}
                and status.get("owner") is not None
            )
            render_stream_health(mode, metrics)
            if not threshold_matches:
                st.warning(
                    f"This snapshot used threshold {float(metrics['confidence']):.2f}; "
                    f"the current threshold is {read_confidence():.2f}. "
                    "Waiting for inference at the current threshold."
                )
            else:
                st.image(
                    frame,
                    alt=f"Server-annotated live frame from the {mode} detector",
                    width="stretch",
                )
                number_col, frame_col = st.columns(2)
                number_col.metric("Objects in this frame", int(metrics["detections"]))
                frame_col.metric("Frame sequence", int(metrics["frame_id"]))
                received = float(metrics["frame_received_at"])
                st.caption(
                    "Last frame received: "
                    f"{time.strftime('%H:%M:%S', time.localtime(received))} "
                    f"({age:.1f}s ago)"
                )
                st.caption(
                    f"Acceptance threshold: raw confidence ≥ {float(metrics['confidence']):.2f} · "
                    f"confidence EMA alpha: {float(metrics.get('confidence_smoothing_alpha', 0.25)):.2f}"
                )
                objects = metrics.get("objects", [])
                if objects:
                    st.markdown("**Detections**")
                    for detected in objects:
                        if detected.get("track_id") is None:
                            confidence_text = (
                                f"raw confidence {float(detected['raw_confidence']):.0%}"
                            )
                        else:
                            confidence_text = (
                                f"smoothed confidence "
                                f"{float(detected['smoothed_confidence']):.0%} "
                                f"(raw {float(detected['raw_confidence']):.0%})"
                            )
                        track_text = (
                            f" · track {detected['track_id']}"
                            if detected.get("track_id") is not None
                            else ""
                        )
                        st.write(f"{detected['name']}{track_text} — {confidence_text}")
                else:
                    st.info("No objects detected in this frame; the camera and inference may still be running.")
            if mode == "coco":
                navigation = metrics.get("navigation")

    with right_panel:
        st.subheader("Obstacle Analysis & Navigation")
        if mode != "coco":
            st.warning(
                "UNCERTAIN — CHECK SURROUNDINGS. Navigation uses COCO obstacle classes; "
                "select COCO to analyze the same incoming frame."
            )
        elif snapshot is None:
            st.info("Waiting for a COCO frame and its navigation analysis.")
        elif navigation is None:
            st.warning(
                "UNCERTAIN — navigation analysis is not available for this frame."
            )
        elif not fresh or not navigation.get("frame_fresh", False):
            st.error(
                "UNCERTAIN — CHECK SURROUNDINGS. The camera frame is stale or inference is no longer current."
            )
        else:
            st.subheader(navigation.get("recommendation", UNCERTAIN))
            st.write(navigation.get("explanation", "No explanation is available."))
            region_status = navigation.get("regions", {})
            region_columns = st.columns(3)
            for column, region in zip(
                region_columns, ("LEFT", "CENTER", "RIGHT")
            ):
                region_data = region_status.get(region, {})
                column.metric(
                    region,
                    "OBSTRUCTED" if region_data.get("obstructed") else "No selected object",
                )
            obstacles = navigation.get("detections", [])
            st.metric("Relevant obstacles", len(obstacles))
            if obstacles:
                st.markdown("**Relevant detections from this frame**")
                for obstacle in obstacles:
                    st.write(
                        f"{obstacle['name']} — {float(obstacle['confidence']):.0%} confidence · "
                        f"{obstacle['proximity']} apparent proximity · "
                        f"overlaps {', '.join(obstacle['regions'])}"
                    )
            else:
                st.info(
                    "No configured relevant obstacles met the confidence threshold. "
                    "This does not establish that a path is physically safe."
                )
            analyzed_at = float(metrics["last_update"])
            st.caption(
                f"Analysis completed {time.strftime('%H:%M:%S', time.localtime(analyzed_at))} · "
                f"same frame sequence {metrics['frame_id']}"
            )
        st.caption(
            "Proximity is an image-based estimate, not a distance in metres. "
            "Experimental visual assistance only; not a safety or collision-avoidance system."
        )


def render_navigation_controls() -> None:
    config = read_navigation_config()
    st.markdown("### Navigation rules")
    relevant_classes = st.multiselect(
        "Relevant COCO classes",
        options=sorted(RELEVANT_CLASSES),
        default=sorted(config.relevant_classes),
        help="Only these classes are considered by the visual navigation heuristic.",
    )
    medium_area = st.slider(
        "Medium apparent area threshold",
        min_value=0.01,
        max_value=0.35,
        value=config.medium_area_fraction,
        step=0.01,
        help="Bounding-box area as a fraction of the image; this is not a physical distance.",
    )
    high_area = st.slider(
        "High apparent area threshold",
        min_value=min(0.8, medium_area + 0.01),
        max_value=0.8,
        value=max(config.high_area_fraction, medium_area + 0.01),
        step=0.01,
        help="A box at or above this image-area ratio is classified as high apparent proximity.",
    )
    with st.expander("Advanced apparent-proximity thresholds"):
        medium_height = st.slider(
            "Medium box-height threshold",
            min_value=0.05,
            max_value=0.8,
            value=config.medium_height_fraction,
            step=0.01,
        )
        high_height = st.slider(
            "High box-height threshold",
            min_value=min(1.0, medium_height + 0.01),
            max_value=1.0,
            value=max(config.high_height_fraction, medium_height + 0.01),
            step=0.01,
        )
        high_bottom = st.slider(
            "High bottom-of-frame threshold",
            min_value=0.6,
            max_value=1.0,
            value=config.high_bottom_fraction,
            step=0.01,
            help="A box reaching this far down can raise apparent proximity when its area is also substantial.",
        )
        medium_growth = st.slider(
            "Medium recent box-growth ratio",
            min_value=1.01,
            max_value=3.0,
            value=config.medium_growth_ratio,
            step=0.01,
        )
        high_growth = st.slider(
            "High recent box-growth ratio",
            min_value=medium_growth + 0.01,
            max_value=5.0,
            value=max(config.high_growth_ratio, medium_growth + 0.01),
            step=0.01,
        )
    blocked_score = st.slider(
        "Region obstruction threshold",
        min_value=0.1,
        max_value=1.5,
        value=config.blocked_region_score,
        step=0.05,
    )
    side_margin = st.slider(
        "Minimum left/right difference",
        min_value=0.0,
        max_value=1.0,
        value=config.side_clearance_margin,
        step=0.05,
        help="Similar side scores result in an uncertain recommendation rather than an arbitrary direction.",
    )
    write_json(
        NAVIGATION_CONFIG_PATH,
        {
            "relevant_classes": relevant_classes,
            "medium_area_fraction": medium_area,
            "high_area_fraction": high_area,
            "medium_height_fraction": medium_height,
            "high_height_fraction": high_height,
            "high_bottom_fraction": high_bottom,
            "medium_growth_ratio": medium_growth,
            "high_growth_ratio": high_growth,
            "blocked_region_score": blocked_score,
            "side_clearance_margin": side_margin,
        },
    )


@st.fragment(run_every="1s")
def show_navigation_view() -> None:
    snapshot = read_snapshot("navigation")
    if snapshot is None:
        render_stream_health("navigation", None)
        st.warning(
            "No navigation results yet. On **Android Camera**, select **Obstacle Detection & Navigation** and start the stream."
        )
        return
    metrics, frame = snapshot
    if not snapshot_uses_current_confidence(metrics):
        render_stream_health("navigation", metrics)
        st.warning(
            f"This frame used confidence threshold {float(metrics['confidence']):.2f}; "
            f"the current threshold is {read_confidence():.2f}. "
            "Navigation is uncertain until a matching inference is available."
        )
        st.subheader(UNCERTAIN)
        return
    navigation = metrics.get("navigation", {})
    received_at = float(metrics["frame_received_at"])
    age = max(0.0, time.time() - received_at)
    status = current_camera_status()
    fresh = (
        age <= 5
        and status.get("mode") == "navigation"
        and status.get("inference_status") not in {"error", "stopped"}
        and bool(navigation.get("frame_fresh", False))
    )
    render_stream_health("navigation", metrics)
    recommendation = navigation.get("recommendation", UNCERTAIN) if fresh else UNCERTAIN
    if fresh:
        st.caption(f"Live analysis · frame {metrics['frame_id']} · {age:.1f}s old")
    else:
        st.error(f"UNCERTAIN — camera frame is stale ({age:.0f}s old). Do not use the previous suggestion.")
    st.warning(
        "Experimental visual assistance only. This is not a safety system and does not guarantee collision avoidance."
    )
    st.subheader(recommendation)
    st.write(
        navigation.get(
            "explanation", "Camera evidence is not sufficient for a current suggestion."
        )
    )
    columns = st.columns([1.6, 1])
    with columns[0]:
        st.image(frame, alt="Latest server-annotated navigation frame", width="stretch")
        st.download_button(
            "Download annotated navigation frame",
            frame,
            file_name="objectvision-navigation-latest.jpg",
            mime="image/jpeg",
            width="stretch",
        )
    with columns[1]:
        detections = navigation.get("detections", [])
        st.metric("Relevant obstacle detections", len(detections))
        st.markdown("**Region assessment**")
        regions = navigation.get("regions", {})
        for region in ("LEFT", "CENTER", "RIGHT"):
            status = regions.get(region, {})
            label = "APPEARS OBSTRUCTED" if status.get("obstructed") else "no selected obstruction detected"
            st.write(f"**{region}:** {label}")
        st.markdown("**Relevant objects**")
        if detections:
            for detection in detections:
                touched = ", ".join(detection.get("regions", []))
                st.write(
                    f"{detection['name']} — {float(detection['confidence']):.0%} confidence, "
                    f"{detection['proximity']} apparent proximity; overlaps {touched}"
                )
        else:
            st.write(
                "No configured relevant obstacle met the current confidence threshold. "
                "This does not prove the path is safe."
            )
    st.caption(
        "Proximity is an image-based heuristic from box area, height, vertical position, and recent size change; "
        "it is not a distance measurement."
    )


def main() -> None:
    view = st.query_params.get("view", "dashboard")
    if view not in {"dashboard", "phone", "custom", "coco", "navigation", "live"}:
        st.error(f"Unknown screen: {view}")
        view = "dashboard"
    render_sidebar(view)
    if view == "phone":
        show_phone_view()
    elif view == "live":
        render_brand("live")
        detector = st.selectbox(
            "Detector for the shared live frame",
            DETECTOR_OPTIONS,
            index=0,
            key="live-detector",
            help="COCO enables navigation analysis. Both panels use one laptop-processed camera frame.",
        )
        mode = "coco" if detector == "COCO" else "custom"
        write_active_mode(mode)
        show_live_detection_view(mode)
    elif view == "coco":
        render_brand("coco")
        show_coco_view()
    elif view == "navigation":
        render_brand("navigation")
        render_navigation_controls()
        show_navigation_view()
    elif view == "custom":
        render_brand("custom")
        show_dashboard()
    else:
        render_brand("dashboard")
        show_dashboard()


if __name__ == "__main__":
    main()
