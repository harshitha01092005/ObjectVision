#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p .stream_state/certs

DEFAULT_INTERFACE="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
HOST_IP=""
if [ -n "$DEFAULT_INTERFACE" ]; then
  HOST_IP="$(ipconfig getifaddr "$DEFAULT_INTERFACE" 2>/dev/null || true)"
fi
if [ -z "$HOST_IP" ]; then
  HOST_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
fi
if [ -z "$HOST_IP" ]; then
  echo "Could not determine the laptop's LAN IP. Connect to a network and retry."
  exit 1
fi

CERT_DIR=".stream_state/certs"
CERT_PATH="$CERT_DIR/cert-$HOST_IP.pem"
KEY_PATH="$CERT_DIR/key-$HOST_IP.pem"
CERT_IP=""
if [ -f "$CERT_PATH" ] && [ -f "$KEY_PATH" ]; then
  CERT_IP="$(openssl x509 -in "$CERT_PATH" -noout -checkip "$HOST_IP" 2>/dev/null || true)"
fi
if [ -z "$CERT_IP" ]; then
  if [ -e "$CERT_PATH" ] || [ -e "$KEY_PATH" ]; then
    CERT_SUFFIX="$(date +%Y%m%d%H%M%S)"
    CERT_PATH="$CERT_DIR/cert-$HOST_IP-$CERT_SUFFIX.pem"
    KEY_PATH="$CERT_DIR/key-$HOST_IP-$CERT_SUFFIX.pem"
  fi
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY_PATH" \
    -out "$CERT_PATH" \
    -days 365 \
    -subj "/CN=$HOST_IP" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$HOST_IP"
fi

BASE_PORT="${OBJECTVISION_PORT:-8501}"
PORT="$BASE_PORT"
until python3 - "$PORT" <<'PY2'
import socket, sys
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        raise SystemExit(1)
    raise SystemExit(0)
PY2
  do
    PORT=$((PORT + 1))
done

echo "Android camera: https://$HOST_IP:$PORT/?view=phone"
echo "Laptop dashboard: https://$HOST_IP:$PORT/?view=dashboard"
echo "Custom detector: https://$HOST_IP:$PORT/?view=custom"
echo "COCO detector: https://$HOST_IP:$PORT/?view=coco"
echo "Obstacle navigation: https://$HOST_IP:$PORT/?view=navigation"

if [ ! -x .venv/bin/streamlit ]; then
  echo "Missing project environment. Create it with Python 3.12 and install requirements.txt."
  exit 1
fi
source .venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.port "$PORT" \
  --server.sslCertFile "$CERT_PATH" \
  --server.sslKeyFile "$KEY_PATH"
