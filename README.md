# ObjectVision

ObjectVision is a local Streamlit application that sends live video from an
Android phone browser to a laptop. Ultralytics YOLO inference runs on the
laptop; the phone is used only as a camera.

## Architecture and screens

```text
Android Chrome rear camera
        │ HTTPS + WebRTC
        ▼
Streamlit / streamlit-webrtc on the laptop
        │ one active camera stream; bounded latest-frame processing
        ├── Custom YOLO model: best.pt
        └── COCO YOLO model: yolov8n.pt
                └── same-frame detections → navigation decision engine
                         └── atomic latest image + metadata snapshot
```

The WebRTC processor selects the model on the laptop. It runs inference once
per processed frame and atomically publishes the annotated image and its
detection/navigation metadata together, so the live panels cannot pair an
image from one frame with analysis from another. Only the latest snapshot for
each mode is retained. One camera sender and one inference session are accepted
at a time.

| Screen | URL | Purpose |
| --- | --- | --- |
| Dashboard | `/?view=dashboard` | Existing custom-model monitoring dashboard |
| Custom Object Detection | `/?view=custom` | Detect the 15 classes from `best.pt` |
| COCO Object Detection | `/?view=coco` | Detect COCO classes with `yolov8n.pt` |
| Obstacle Detection & Navigation | `/?view=navigation` | Filter COCO detections and show experimental visual guidance |
| Live Detection & Navigation | `/?view=live` | Side-by-side annotated detection and same-frame navigation analysis |
| Android Camera | `/?view=phone` | Start/stop the phone camera and choose the active inference mode |

All screens are also linked in the application sidebar. The Dashboard and
Custom Object Detection screens show the same existing custom-model metrics.

## Visual intelligence and operator assistance

The current pipeline now includes four lightweight computer-vision support modules that run alongside the existing YOLO detector without creating an independent camera stream or re-running the model multiple times:

- `image_quality.py` measures blur, brightness, and contrast for each incoming frame and classifies the overall quality as `GOOD`, `DEGRADED`, or `POOR` using configurable thresholds.
- `motion_analysis.py` estimates apparent camera motion from consecutive frames using OpenCV feature tracking and classifies motion as `LOW`, `MODERATE`, `HIGH`, or `UNKNOWN` without claiming exact physical displacement.
- `visual_memory.py` keeps a bounded multi-object memory across frames with track IDs, state transitions, and bounded event history for ACTIVE, TEMPORARILY MISSING, and LOST tracks.
- `operator_assistance.py` combines detection confidence, image quality, motion, track consistency, and frame freshness to emit transparent operator suggestions and warnings without sending commands to any robot or controller.

These modules are integrated directly into the WebRTC `recv()` pipeline, so the same camera frame is evaluated once for quality and motion before the detector result is added to the snapshot and dashboard metrics.

Detection uses Ultralytics ByteTrack (`objectvision_bytetrack.yaml`) on the same single
inference result. The configured confidence threshold filters raw YOLO
confidence before an object is annotated, included in the detection metrics,
or passed to navigation. Annotations show `EMA` confidence for tracker IDs;
the dashboard keeps the raw score alongside it. The confidence EMA uses
`alpha = 0.25`, so 25% of each valid current score and 75% of the previous
smoothed score contribute to the next displayed value. A missing detection
does not count as a zero score, and missing/predicted-only boxes are never
drawn as current detections. ID-less outputs use their raw score and box.

Box coordinates come directly from ByteTrack; the application does not apply a
second box smoother. Smoothing state expires after more than eight missing
processed frames or three seconds and is capped at 128 track IDs; ByteTrack's
own lost-track buffer is also set to eight frames. Smoothing resets when the
inference mode or camera stream changes.
ByteTrack's tracker IDs are useful associations, not guarantees of perfect
identity persistence. No additional NMS is applied; Ultralytics' normal
detection postprocessing/NMS remains responsible for overlapping candidates.
Navigation continues to use current tracker detections and raw confidence,
not the smoothed display values. A strong current STOP result is not delayed
by smoothing; poor-quality frames or a lack of relevant current detections
produce UNCERTAIN rather than a clear-path recommendation.
Result panels also withhold annotated output from an earlier confidence setting
after the slider changes, until a snapshot produced with the new threshold is
available.

Blur and motion thresholds should be calibrated against real camera footage because a single blur score or motion estimate does not guarantee safety or detection quality in every scene. The implementation intentionally avoids claiming exact physical localization or robot position.

## Models

### Custom model

`best.pt` is loaded exclusively for Custom Object Detection. Its 15 classes
are Battery, Dice, ToyCar, Ants, Bees, Beetles, Caterpillars, Earthworms,
Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, and Weevils. It is not
modified by this application.

### COCO model

COCO detection and navigation use a separate Ultralytics `yolov8n.pt` model.
It is initialized once per server process and uses the class-name mapping
provided by the loaded model. If the weights are not already available,
Ultralytics attempts to download them on first use. A download or model-loading
failure is shown in the camera status and on the relevant results screen.

Both models currently run on the laptop CPU. The COCO model is a small model,
but CPU speed depends on laptop hardware and input video resolution.

## Reproducible latency benchmark

Use `benchmark_latency.py` to measure the app's published inference,
navigation-evaluation, and processed-frame latency fields without connecting a
live camera. From the project directory:

```bash
.venv/bin/python benchmark_latency.py \
  --mode navigation \
  --resolution 640x480 \
  --warmup 10 \
  --samples 100
```

By default, the benchmark uses a deterministic generated frame and does not
save image frames. To use a prerecorded image or video instead, pass
`--input /path/to/image-or-video`; the report records its file name and SHA-256
digest, but does not copy or store the source. Video frames are read in order
and looped when the requested sample count exceeds its length. Input reading,
decoding, and resizing occur outside the measured processing interval.

The benchmark warms the model and pipeline for the requested number of frames,
resets tracker/navigation/memory state, then collects the requested measured
samples. It prints a JSON report with environment and inference settings, the
sample count, unavailable-value count, and nearest-rank p50/p95 for each timing
field. An unavailable navigation duration is counted as unavailable, never
treated as zero. Use `--output report.json` to save a new report; it refuses to
overwrite an existing file.

`inference_latency_ms` measures the actual `model.track()` call, excluding
model-lock wait. `navigation_evaluation_latency_ms` measures only
`NavigationEngine.evaluate()` and is unavailable in custom-model mode.
`frame_processing_latency_ms` starts after frame decoding, quality/motion
analysis, throttling, and model-retry checks, and ends after operator
assistance generation. It excludes camera wait, skipped/throttled frames,
snapshot serialization/publication, and input-file reading. These are
processing-stage measurements, not full camera-to-dashboard latency. Compare
reports only when hardware, model weights, mode, resolution, confidence,
tracker, and source type are matched. The tool makes no performance claim by
itself.

### Optional p95 acceptance thresholds

Thresholds are user-selected acceptance criteria, separate from measured
latencies. Supply one or more positive, finite limits in milliseconds:

```bash
.venv/bin/python benchmark_latency.py \
  --mode navigation \
  --resolution 640x480 \
  --warmup 10 \
  --samples 100 \
  --max-inference-latency-ms 35 \
  --max-navigation-evaluation-latency-ms 5 \
  --max-frame-processing-latency-ms 50
```

The optional `threshold_validation` JSON section reports each configured
metric's threshold, measured p95, status (`pass`, `fail`, or `not_evaluable`),
and reason. A metric passes when its measured p95 is at or below the supplied
threshold. If no p95 measurement exists, its status is `not_evaluable`; it is
never substituted with zero. Invalid, non-finite, or non-positive thresholds
are rejected with a CLI error. When no threshold options are provided, the
section is omitted and the existing report schema is unchanged.

Thresholds validate only the selected processing-stage metric for this
benchmark run. Passing one does not establish end-to-end camera-to-dashboard
latency, real-time suitability, or real-world navigation safety. Synthetic
input does not represent a real scene or camera stream.

### Repeatability analysis

To compare two or more saved benchmark reports, run:

```bash
.venv/bin/python benchmark_latency.py \
  --compare run-1.json run-2.json run-3.json \
  --output repeatability.json
```

Reports must match in environment, input source and resolution, inference/model
settings, warm-up count, measured sample count, and measurement method. A
mismatch is rejected with the differing report section; create new runs with
matching settings before comparing. The analysis gives each run's p50 and p95
for every latency metric and reports the across-run range (maximum minus
minimum) of each percentile. Unavailable metrics, including navigation latency
when it was not measured, remain `null` and are counted as unavailable rather
than treated as zero.

The range describes variation between separate benchmark runs, not
frame-to-frame timing variation. Repeatable synthetic-input results do not
establish real-camera performance or end-to-end phone-to-laptop latency, and
repeatability alone does not mean the system is fast or safe. Use the same
prerecorded source and matching environment/settings when reproducibility
against real footage is required; the benchmark does not save input frames.

## Python and installation

The system Python in this workspace reports Python 3.14.6. The existing
project virtual environment is Python 3.12.13 and contains the required
packages; model loading and dependency consistency were checked there.
`requirements.txt` specifies minimum package versions, including historical
minimums that should not be assumed compatible with Python 3.14. Python 3.14
has not been verified for this project. Use Python 3.12 for a reproducible
setup.

Create a new environment only if `.venv` is absent or needs replacement:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For future terminal sessions:

```bash
cd "/path/to/crack_detector 2"
source .venv/bin/activate
```

No dependency changes are required for the added navigation engine. The
existing `requirements.txt` covers Streamlit, streamlit-webrtc, OpenCV,
Ultralytics, PyTorch, and the WebRTC media stack.

## Start the application

From the project directory, activate the existing environment and launch HTTPS:

```bash
cd "/path/to/crack_detector 2"
source .venv/bin/activate
bash run_https.sh
```

The launcher determines the laptop's default-route LAN IP, selects the
configured port (default `8501`, then the next available port), creates a
host-specific certificate under `.stream_state/certs/`, and prints URLs for
all five screens. To choose a starting port:

```bash
OBJECTVISION_PORT=8505 bash run_https.sh
```

The launcher does not overwrite or delete the supplied files in `certs/`.
Its generated certificate and key are local runtime files under
`.stream_state/certs/`.

Run the project unit tests with:

```bash
cd "/path/to/crack_detector 2"
source .venv/bin/activate
python -m unittest discover -s tests -q
```

For desktop-only use without the phone camera, plain HTTP can be used:

```bash
source .venv/bin/activate
streamlit run app.py --server.address 0.0.0.0
```

Phone camera access generally requires a secure HTTPS origin. Do not use the
plain HTTP command for the phone camera workflow.

## Connect an Android phone

1. Connect the laptop and Android phone to the same Wi-Fi network. Avoid guest
   Wi-Fi that isolates devices.
2. Run `bash run_https.sh` and use the **Android camera** URL printed by the
   launcher: `https://<LAPTOP_LAN_IP>:<PORT>/?view=phone`.
3. Open that URL in current Android Chrome. Review the certificate warning
   carefully. This is a self-signed, local-development certificate; proceed
   only when you trust the laptop and network. Some Chrome versions or device
   policies may not allow camera access after a certificate exception. A
   locally trusted certificate may be required for those devices.
4. Select the inference mode on the Android Camera screen, tap **START**, and
   grant camera permission. The capture constraint requests the environment-
   facing camera (`facingMode: ideal environment`). Keep the tab open.
5. On the laptop, open the printed dashboard, COCO, or navigation URL.
6. Tap **STOP** in the camera component before switching phones or opening a
   second sender. A duplicate sender is rejected while the current stream is
   active.

Choose the mode on the phone's Android Camera screen. The stream processes one
mode at a time; changing the selection switches inference on the existing
stream. The laptop result screens are read-only and each mode keeps separate
latest output.

### HTTPS, Wi-Fi, and permission troubleshooting

- **Camera API unavailable or permission denied:** confirm the URL begins
  with `https://`, use Android Chrome, accept/trust the certificate if
  appropriate, then check Android's site camera permission and Chrome's
  camera permission. Reload the page after changing a permission.
- **Certificate warning persists:** the launcher generates a self-signed
  certificate. Android may reject the exception for camera access. Use a
  locally trusted certificate if required by the browser/device. A new LAN IP
  requires a certificate with that IP in its subject alternative name.
- **Phone cannot reach the laptop:** check that both devices are on the same
  non-isolated Wi-Fi network, use the exact current LAN IP and port printed by
  the launcher, and allow incoming connections for Python/Streamlit in the
  laptop firewall.
- **No frames or stale status:** keep the phone page in the foreground, check
  camera permission and Wi-Fi, and stop/restart the WebRTC stream. Mobile
  browsers can pause camera work in background tabs.
- **Duplicate stream warning:** stop the active camera component and wait a
  few seconds before starting another sender. Inactive stream leases expire
  after a short timeout.
- **COCO model error:** confirm internet access on first use so Ultralytics can
  download `yolov8n.pt`, or provide that weights file in the project directory.

## Live Detection & Navigation

1. On the Android Camera page, choose a detector and start the camera stream.
   COCO is recommended when you want navigation analysis; the custom model
   remains available for its 15 project-specific classes.
2. Open `/?view=live` on the laptop and select **COCO** or **Custom**. This
   setting selects the model used by the one incoming stream.
3. The left panel shows the annotated server-processed frame, detections,
   confidence scores, detector, frame sequence, and server receive/inference
   status. An empty detection list means inference completed without a
   threshold-qualified object; it does not mean the camera is disconnected.
4. With COCO selected, the right panel displays navigation analysis made from
   those exact model detections and that exact frame. With the custom model,
   navigation is explicitly uncertain because its class labels are not COCO
   obstacle categories.
5. If frames become stale, the page replaces any directional recommendation
   with an uncertain state. Do not treat the last image or suggestion as live.

The standalone COCO and navigation pages remain available. Adjust the shared
**Confidence threshold** in the sidebar. A lower threshold
   displays more possible objects and can increase false positives.
The COCO page shows the annotated frame, total, counts by detected COCO class,
connection freshness, and download control.

The COCO weights are independent of `best.pt`; selecting COCO never changes the
custom model.

## Obstacle Detection & Navigation

1. Select **Obstacle Detection & Navigation** on the phone's Android Camera
   screen and start the stream.
2. Open `/?view=navigation` on the laptop.
3. Configure the relevant COCO classes, area and height thresholds, bottom-of-
   frame and recent box-growth thresholds, region obstruction threshold, and
   minimum left/right difference on the navigation screen. The shared
   confidence threshold is in the sidebar.
4. Read the marked LEFT, CENTER, and RIGHT regions, relevant detections, the
   apparent-proximity category, and the recommendation. The recommendation
   changes to **UNCERTAIN** when the camera frame is stale.

The decision engine is a separate, unit-tested `navigation.py` module. It
validates boxes, clips boxes to the frame, assigns a detection to every region
it overlaps, and calculates region obstruction scores. A high-apparent-
proximity detection in the center or obstruction on both sides leads to STOP.
If the center appears obstructed, a directional suggestion is made only when
one side is sufficiently clearer, and it must persist across consecutive
frames. Similar side scores and insufficient evidence produce UNCERTAIN.

Apparent proximity is a configurable visual heuristic based on bounding-box
area, height, vertical position, and recent box-size change. Its thresholds
are adjustable in the navigation controls. It is **not a physical distance
measurement**: object dimensions, perspective, camera
angle, and lens characteristics vary. No metres are inferred. This feature is
experimental visual assistance only; it is not automatic control, a safety
system, or guaranteed collision avoidance. Always use independent judgment
and check the surroundings.

## Camera streaming diagnosis

The Android camera component can show its own local preview before the laptop
has received any WebRTC video frames. ObjectVision therefore reports server-
received frame counts, last received time, inference state, and model errors
separately from the browser's `playing`/signaling state.

The server-side processor must be supplied through `video_processor_factory`.
With the installed `streamlit-webrtc` 0.79 API, passing `on_video_ended` at the
same time selects the callback-based processor and takes precedence over
`video_processor_factory`; this prevents the custom processor's `recv()` from
running. ObjectVision now handles track cleanup in the processor's `on_ended()`
method instead.

If the UI reports WebRTC signaling but no frames reach the laptop:

- Confirm the phone can reach the laptop over the same non-isolated Wi-Fi.
- Permit incoming Python/Streamlit network traffic in the laptop firewall.
- WebRTC media uses UDP in addition to the HTTPS signaling connection. A
  firewall, VPN, guest network, or router that blocks peer-to-peer UDP can
  leave a local preview visible while server frames never arrive.
- The component uses Google's public STUN server by default/explicit
  configuration, but STUN alone cannot relay video through every restrictive
  NAT or firewall. Such networks may require an appropriately configured TURN
  service; no TURN credentials are bundled.
- Check the phone status text for signaling, incoming server frames, stale
  video, or inference errors. The laptop's combined screen reports the same
  server state and will not call an empty detection result a disconnection.

## Tests and validation

Run the navigation tests without a camera or downloaded weights:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Run syntax and dependency checks:

```bash
python -m py_compile app.py navigation.py tests/test_navigation.py
python -m pip check
```

The unit tests use synthetic detections and a mocked detector. They do not
verify real Android camera/browser behavior. `best.pt` loaded successfully in the existing
Python 3.12 environment. COCO initialization and live COCO inference require
the separate `yolov8n.pt` weights and first-run network access; they were not
downloaded as part of the unit tests.

## Project files and known limitations

- `app.py` — Streamlit screens, WebRTC processing, model loading, and bounded
  per-mode state.
- `navigation.py` — independent navigation assessment and decision engine.
- `tests/test_navigation.py` — camera-free synthetic unit tests.
- `best.pt` — existing custom model, preserved.
- `requirements.txt` — existing Python dependencies.
- `run_https.sh` — HTTPS launcher and reachable URL output.
- `.stream_state/` — generated certificates, selected mode, confidence,
  latest frames, and bounded metrics; local runtime data.
- `certs/` — supplied certificates, preserved and not modified by the launcher.

Android compatibility has **not** been verified on a physical Android device.
Manual testing is still required for the target Android/Chrome version,
certificate trust behavior, Wi-Fi/firewall access, phone camera permissions,
and real-time CPU performance. COCO model download and live inference also
require first-run validation on the target laptop.
