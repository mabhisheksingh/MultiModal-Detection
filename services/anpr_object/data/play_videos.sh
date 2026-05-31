#!/bin/bash
# Video player server launcher - FastAPI version

PORT="${1:-8000}"
HOST="${2:-0.0.0.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    echo "Missing dependencies: fastapi, uvicorn"
    echo "Install with: pip install -r /home/harsha/abhishek/vlm-video-captioning/ANPR/triton_client/requirements.txt"
    exit 1
fi

cd "$SCRIPT_DIR" || exit 1
python3 "$SCRIPT_DIR/fastapi_video_server.py" --port "$PORT" --host "$HOST"
