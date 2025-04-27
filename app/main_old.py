"""
CloudPose Task 2 – FASTAPI web‑service
--------------------------------------
A RESTful service that exposes two endpoints:
1. `/pose/json` – returns detected key‑points and metadata in JSON.
2. `/pose/image` – returns the same JSON *plus* an annotated image encoded as base64.

The service works with MoveNet (TensorFlow‑Lite) provided in *model_path* and
runs on any port between 60000‑61000 (default 60001).  It is designed to be
container‑friendly and thread‑safe.

Run locally:
    uvicorn cloudpose_service:app --host 0.0.0.0 --port 60001 --workers 4

Docker health‑check (once container is up):
    curl -X POST http://localhost:60001/pose/json -d '{"id":"ping","image":"..."}'
"""

import os
import io
import base64
import time
import uuid
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# ---------------------------------------------------------------------------
# Configuration & model bootstrap
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.getenv("CLOUDPOSE_MODEL", "models/movenet-full-256.tflite")
PORT: int = int(os.getenv("CLOUDPOSE_PORT", "60001"))

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not locate MoveNet model at '{MODEL_PATH}'. "
        "Set environment variable CLOUDPOSE_MODEL or mount the model file."
    )

# Initialise interpreter once at startup (re‑used by all requests)
INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH)
INTERPRETER.allocate_tensors()
INPUT_DETAILS = INTERPRETER.get_input_details()
OUTPUT_DETAILS = INTERPRETER.get_output_details()
INPUT_H, INPUT_W = INPUT_DETAILS[0]["shape"][1:3]

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class PoseRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image: str  # base64‑encoded JPEG/PNG

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    probability: float

class PoseResponse(BaseModel):
    id: str
    count: int
    boxes: List[BoundingBox]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float
    # Present only for /pose/image endpoint
    annotated_image: Optional[str] = None  # base64 PNG

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _b64_to_cv2(b64_str: str) -> np.ndarray:
    """Decode base64 to BGR image array."""
    try:
        img_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 imdecode returned None")
        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


def _cv2_to_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode BGR image to base64."""
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("cv2 imencode failed")
    return base64.b64encode(buf).decode()


def _run_movenet(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run MoveNet inference; returns (keypoints, resized_input)"""
    img_height, img_width = img_bgr.shape[:2]

    # Resize with aspect‑ratio distortion (MoveNet expects 256×256)
    resized = cv2.resize(img_bgr, (INPUT_W, INPUT_H))
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    INTERPRETER.set_tensor(INPUT_DETAILS[0]["index"], input_data)
    INTERPRETER.invoke()
    keypoints = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]["index"])[0]  # (17,3)
    return keypoints, resized


def _annotate(img_bgr: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Draw key‑points & skeleton, compute bounding box list."""
    img_out = img_bgr.copy()
    h, w = img_out.shape[:2]

    print(keypoints)

    # Map keypoint coordinates back to original resolution
    pts = [(int(kp[1] * w), int(kp[0] * h), kp[2]) for kp in keypoints]

    # Draw keypoints
    for x, y, conf in pts:
        cv2.circle(img_out, (x, y), 4, (0, 255, 0), -1)

    # Simple skeleton lines (COCO format)
    skeleton = [
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (6, 8), (7, 9), (8, 10),
        (11, 13), (12, 14), (13, 15), (14, 16),
    ]
    for a, b in skeleton:
        x1, y1, c1 = pts[a]
        x2, y2, c2 = pts[b]
        if c1 > 0.3 and c2 > 0.3:
            cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Bounding box (min/max of confident pts)
    xs = [x for x, y, conf in pts if conf > 0.3]
    ys = [y for x, y, conf in pts if conf > 0.3]
    if xs and ys:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        probability = float(np.mean([conf for _, _, conf in pts]))
        boxes = [BoundingBox(x=xmin, y=ymin, width=xmax - xmin, height=ymax - ymin, probability=probability)]
    else:
        boxes = []
    return img_out, boxes

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="CloudPose – Pose Estimation API", version="1.0.0")


@app.post("/pose/json", response_model=PoseResponse)
async def pose_json(req: PoseRequest):
    """Return keypoints & metadata as JSON."""
    t0 = time.perf_counter()
    img_bgr = _b64_to_cv2(req.image)
    t_pre = time.perf_counter()

    keypoints, _ = await run_in_threadpool(_run_movenet, img_bgr)
    t_inf = time.perf_counter()

    _, boxes = _annotate(img_bgr, keypoints)
    t_post = time.perf_counter()

    return PoseResponse(
        id=req.id,
        count=1 if keypoints.size else 0,
        boxes=boxes,
        keypoints=[[keypoints.tolist()]],  # keep outer list for potential multi‑person models
        speed_preprocess=(t_pre - t0) * 1000,
        speed_inference=(t_inf - t_pre) * 1000,
        speed_postprocess=(t_post - t_inf) * 1000,
    )


@app.post("/pose/image", response_model=PoseResponse)
async def pose_image(req: PoseRequest):
    """Return keypoints plus annotated image (base64)."""
    t0 = time.perf_counter()
    img_bgr = _b64_to_cv2(req.image)
    t_pre = time.perf_counter()

    keypoints, _ = await run_in_threadpool(_run_movenet, img_bgr)
    t_inf = time.perf_counter()

    img_annotated, boxes = _annotate(img_bgr, keypoints)
    annotated_b64 = _cv2_to_b64(img_annotated)
    t_post = time.perf_counter()

    return PoseResponse(
        id=req.id,
        count=1 if keypoints.size else 0,
        boxes=boxes,
        keypoints=[[keypoints.tolist()]],
        speed_preprocess=(t_pre - t0) * 1000,
        speed_inference=(t_inf - t_pre) * 1000,
        speed_postprocess=(t_post - t_inf) * 1000,
        annotated_image=annotated_b64,
    )

# ---------------------------------------------------------------------------
# Entrypoint when executed directly (e.g. `python cloudpose_service.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("cloudpose_service:app", host="0.0.0.0", port=PORT, workers=4)