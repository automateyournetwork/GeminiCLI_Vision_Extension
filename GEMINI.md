# Vision Extension (Webcam → Frames → Reasoning)

This extension adds a Vision MCP server and a set of `/vision` commands for Gemini-CLI.
Use it to list cameras, open a device, capture frames/bursts, and run ASL mode.

## Commands
- **/vision:help** — quick help and examples.
- **/vision:devices** — list available cameras and basic properties.
- **/vision:start** — open a camera.
  - keys: `camera_index` (int), `width`, `height`, `fps`, `backend` (auto|avfoundation|v4l2|dshow|msmf, etc.)
- **/vision:status** — show camera state + properties.
- **/vision:capture** — single frame; options: `save_dir`, `format` (jpg|png).
- **/vision:burst** *(if present)* — capture N frames spaced by `period_ms`.
- **/vision:stop** — release the camera.
- **/vision:asl** — timed burst → every other frame → ASL interpretation instruction line.

## Typical Flow
1. `/vision:devices`
2. `/vision:start camera_index=0 width=640 height=480 fps=15 backend=auto`
3. `/vision:capture` or `/vision:asl`
4. `/vision:stop`

## Notes & Tips
- **macOS**: if using iPhone/Continuity Camera or USB cams, you may need to grant **Camera** permission to the terminal app.
- **Linux**: ensure your user can access `/dev/video*` (e.g., `video` group) and that `v4l2` is available.
- **Windows**: backends like `msmf` or `dshow` can help if `auto` fails.
- Frame formats default to `jpg` unless your command overrides with `format=png`.

## ASL Mode (High-level)
- Captures a short burst (`duration_ms`, `period_ms`, or `n`) via `vision_burst`.
- Keeps every other frame (`@f0 @f2 @f4 …`) and appends an instruction:
  “You are an ASL interpreter… Output ONLY the transcript text.”
- Designed for quick, inline ASL transcription experiments.

## Safety
- This extension does **not** run shell commands by default.
- Always stop the camera with `/vision:stop` when finished.
