# Vision Extension (Webcam → Frames → Reasoning)

This extension adds a Vision MCP server and a set of `/vision` commands for Gemini-CLI.
Use it to list cameras, open a device, capture frames/bursts, and run ASL mode.

## Commands
- **/vision** — quick help and examples.
- **/vision:devices** — list available cameras and basic properties.
- **/vision:start** — open a camera.
  - keys: `camera_index`, `width`, `height`, `fps`, `backend` (auto|avfoundation|v4l2|dshow|msmf)
- **/vision:status** — show camera state + properties.
- **/vision:capture** — single frame; options: `save_dir`, `format` (jpg|png).
- **/vision:burst** — capture N frames spaced by `period_ms`.
- **/vision:stop** — release the camera.
- **/vision:asl** — timed burst → every other frame → ASL interpretation instruction line.

## Typical Flow
1. `/vision:devices`
2. `/vision:start camera_index=0 width=640 height=480 fps=15 backend=auto`
3. `/vision:capture` or `/vision:asl`
4. `/vision:stop`

## ASL Mode (High-level)
- Captures a short burst (`duration_ms`, `period_ms`, or `n`) via `vision_burst`.
- Keeps every other frame (tokens like `\@f0` `\@f2` `\@f4`) and appends an instruction:
  “You are an ASL interpreter. Analyze ONLY the attached photo sequence (left→right is chronological). Transcribe the user's signing into clear, grammatical English. If unsure about a word, fingerspell it in ALL CAPS in brackets. Output ONLY the transcript text.”

## Notes
- macOS: grant **Camera** permission to your terminal (System Settings → Privacy & Security → Camera).
- Linux: ensure user has access to `/dev/video*` (e.g., add to `video` group).
- Windows: try `msmf`/`dshow` backends if `auto` fails.

## Safety
- No shell execution. Always stop the camera with `/vision:stop`.
