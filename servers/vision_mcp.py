#!/usr/bin/env python3
"""
MCP Server: Vision (webcam still-frame + burst capture)
Tools:
  - list_cameras(max_index?: int=10)
  - vision_start(camera_index?: int=0, width?: int=640, height?: int=480, fps?: int=15, backend?: str="auto")
  - vision_status()
  - vision_capture(save_dir?: str="~/.vision_frames", format?: "jpg"|"png"="jpg")
  - vision_burst(n?: int=8, period_ms?: int=150, save_dir?: str=".", format?: "jpg"|"png"="jpg", warmup?: int=3, duration_ms?: int=0)
  - vision_stop()

Notes:
- No base64 in responses (optimized for @file attachment flow).
- Pure MCP over stdio (FastMCP). Logs to stderr only. No network calls.
"""

import os, sys, time, logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# ----- Logging to stderr only -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("VisionMCP")

# ---------- FastMCP ----------
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from fastmcp import FastMCP  # type: ignore

# ---------- OpenCV ----------
try:
    import cv2
except Exception as e:
    log.error("OpenCV (cv2) not available: %s", e)
    raise

# Global camera handle (single-camera MVP)
_CAM = {
    "cap": None,
    "index": None,
    "props": {},
}

# Backend map for portability
_BACKENDS = {
    "auto": None,  # let OpenCV pick
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", None),  # macOS
    "msmf": getattr(cv2, "CAP_MSMF", None),                  # Windows
    "dshow": getattr(cv2, "CAP_DSHOW", None),                # Windows (alt)
    "v4l2": getattr(cv2, "CAP_V4L2", None),                  # Linux
}

def _open_cam(camera_index: int, width: int, height: int, fps: int, backend: str) -> Tuple[bool, str]:
    """Open camera with optional backend, set properties; populate _CAM."""
    if _CAM["cap"] is not None:
        return True, "Camera already open"

    be = backend.lower().strip() if backend else "auto"
    api_pref = _BACKENDS.get(be, None)

    log.info("Opening camera index=%s backend=%s width=%s height=%s fps=%s", camera_index, be, width, height, fps)
    if api_pref is None:
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index, api_pref)

    if not cap or not cap.isOpened():
        return False, f"Failed to open camera index {camera_index} (backend={be})"

    # Try to set requested properties
    if width > 0:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
    if height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:    cap.set(cv2.CAP_PROP_FPS,          float(fps))

    # Read back actuals
    actual = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps":    float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "backend": be,
    }
    _CAM["cap"] = cap
    _CAM["index"] = camera_index
    _CAM["props"] = actual
    log.info("Camera open with props: %s", actual)
    return True, "Camera opened"

def _close_cam():
    if _CAM["cap"] is not None:
        try:
            _CAM["cap"].release()
        except Exception:
            pass
    _CAM["cap"] = None
    _CAM["index"] = None
    _CAM["props"] = {}

def _grab_frame() -> Tuple[bool, Optional[Any], str]:
    cap = _CAM["cap"]
    if cap is None or not cap.isOpened():
        return False, None, "Camera not open"
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None, "Failed to read frame"
    return True, frame, "ok"

def _encode_image(frame, fmt: str) -> Tuple[bool, bytes, str]:
    ext = ".jpg" if fmt.lower() == "jpg" else ".png"
    ok, buf = cv2.imencode(ext, frame)  # returns ndarray of bytes
    if not ok:
        return False, b"", "cv2.imencode failed"
    return True, buf.tobytes(), ext

def _timestamp_name(prefix="frame", ext=".jpg") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    return f"{prefix}_{ts}_{ms:03d}{ext}"

# ---------- MCP server ----------
mcp = FastMCP("Vision MCP")

@mcp.tool()
def list_cameras(max_index: int = 10) -> Dict[str, Any]:
    """
    Probe camera indexes from 0..max_index-1; return which are openable.
    """
    results = []
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i)
            ok = bool(cap and cap.isOpened())
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                results.append({"index": i, "open": True, "width": w, "height": h, "fps": fps})
            else:
                results.append({"index": i, "open": False})
        except Exception as e:
            results.append({"index": i, "open": False, "error": str(e)})
        finally:
            try:
                if 'cap' in locals() and cap:
                    cap.release()
            except Exception:
                pass
    return {"cameras": results}

@mcp.tool()
def vision_start(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    backend: str = "auto",
) -> Dict[str, Any]:
    """
    Open the camera with optional size/fps/backend.
    backend one of: auto,avfoundation,msmf,dshow,v4l2
    """
    ok, msg = _open_cam(camera_index, width, height, fps, backend)
    return {"ok": ok, "message": msg, "props": _CAM["props"], "index": _CAM["index"]}

@mcp.tool()
def vision_status() -> Dict[str, Any]:
    """
    Report whether camera is open and its properties.
    """
    cap = _CAM["cap"]
    return {
        "open": bool(cap is not None and cap.isOpened()),
        "index": _CAM["index"],
        "props": _CAM["props"],
    }

@mcp.tool()
def vision_capture(
    save_dir: str = "~/.vision_frames",
    format: str = "jpg",
) -> Dict[str, Any]:
    """
    Capture one frame. Saves to save_dir and returns the saved path and metadata.
    (No base64 returned.)
    """
    ok, frame, msg = _grab_frame()
    if not ok:
        return {"ok": False, "error": msg}

    ok2, img_bytes, ext = _encode_image(frame, format)
    if not ok2:
        return {"ok": False, "error": ext}

    out_dir = Path(os.path.expanduser(save_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _timestamp_name("frame", ext)
    fpath = out_dir / fname
    try:
        with open(fpath, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        return {"ok": False, "error": f"Failed to write file: {e}"}

    return {
        "ok": True,
        "path": str(fpath),
        "mime": "image/jpeg" if ext == ".jpg" else "image/png",
        "width": int(_CAM["props"].get("width", 0)),
        "height": int(_CAM["props"].get("height", 0)),
    }

@mcp.tool()
def vision_burst(
    n: int = 8,
    period_ms: int = 150,
    save_dir: str = ".",
    format: str = "jpg",
    warmup: int = 3,
    duration_ms: int = 0,  # optional duration override
) -> Dict[str, Any]:
    """
    Capture N frames spaced by period_ms and return their file paths (chronological).
    If duration_ms > 0, n is computed as round(duration_ms / period_ms).
    (No base64 returned.)
    """
    cap = _CAM["cap"]
    if cap is None or not cap.isOpened():
        return {"ok": False, "error": "Camera not open"}

    # compute n from duration if provided
    if duration_ms and duration_ms > 0:
        n = max(1, int(round(float(duration_ms) / float(period_ms))))

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    for _ in range(max(0, int(warmup))):
        cap.read()

    out_dir = Path(os.path.expanduser(save_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = ".jpg" if format.lower() == "jpg" else ".png"
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    width = int(_CAM["props"].get("width", 0))
    height = int(_CAM["props"].get("height", 0))

    period_s = max(0.0, float(period_ms) / 1000.0)
    t0 = time.perf_counter()

    paths: list[str] = []
    for i in range(max(1, int(n))):
        target = t0 + i * period_s
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)

        ok, frame, msg = _grab_frame()
        if not ok or frame is None:
            return {"ok": False, "error": f"Failed to read frame: {msg}", "paths": paths}

        ok2, img_bytes, _ = _encode_image(frame, format)
        if not ok2:
            return {"ok": False, "error": "cv2.imencode failed", "paths": paths}

        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        fname = f"asl_{ts}_{ms:03d}_{i:02d}{ext}"
        fpath = out_dir / fname
        with open(fpath, "wb") as f:
            f.write(img_bytes)
        paths.append(str(fpath))

        # Optional progress logs (stderr)
        if i == 0 or (i + 1) % 5 == 0 or (i + 1) == n:
            log.info("Burst capture %d/%d saved %s", i + 1, n, fpath.name)

    return {
        "ok": True,
        "paths": paths,
        "mime": mime,
        "width": width,
        "height": height,
        "n": len(paths),
        "period_ms": period_ms,
        "duration_ms": duration_ms,
        "save_dir": str(out_dir),
    }

@mcp.tool()
def vision_stop() -> Dict[str, Any]:
    """
    Release the camera.
    """
    _close_cam()
    return {"ok": True}

if __name__ == "__main__":
    mcp.run()
