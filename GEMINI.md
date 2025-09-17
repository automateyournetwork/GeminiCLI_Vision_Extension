# Vision Extension (Webcam → Frames → Reasoning)

This extension adds a Vision MCP server and a set of `/vision` commands for Gemini-CLI.
Use it to list cameras, open a device, capture frames/bursts, run ASL mode, generate AI images (Banana), and animate images into videos (Veo).

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
- **/vision:banana** — timed burst → every other frame → ASL interpretation instruction line.
    `Args:`
      `prompt (or quoted string after command)`
      `n (default 1) — number of images`
      `out_dir (default ".")`
      `format (capture format: jpg|png, default jpg)`
      `model (default gemini-2.5-flash-image-preview)`
      `outfile_base (optional; also copy capture to ./<base>.<ext>)`
      `Auto-start overrides (optional): camera_index, width, height, fps, backend`
- **/vision:veo** — capture → Banana image → Veo video. Auto-starts camera if closed.
    `Args:`
      `prompt (one prompt for both steps) or banana_prompt + veo_prompt`
      `negative_prompt (default "")`
      `out_dir (default "."), format (capture jpg|png, default jpg)`
      `banana_model (default gemini-2.5-flash-image-preview)`
      `veo_model (default veo-3.0-generate-001)`
      `n (Banana image count, default 1)`
      `Veo options: aspect_ratio ("16:9" or "9:16"), resolution (e.g. "1080p" for 16:9), seed`


Auto-start overrides (optional): camera_index, width, height, fps, backend
## Typical Flow
1. `/vision:devices`
2. `/vision:start camera_index=0 width=640 height=480 fps=15 backend=auto`
3. `/vision:capture` or `/vision:asl`
4. `/vision:stop`

## ASL Mode (High-level)
- Captures a short burst (`duration_ms`, `period_ms`, or `n`) via `vision_burst`.
- Keeps every other frame (tokens like `\@f0` `\@f2` `\@f4`) and appends an instruction:
  “You are an ASL interpreter. Analyze ONLY the attached photo sequence (left→right is chronological). Transcribe the user's signing into clear, grammatical English. If unsure about a word, fingerspell it in ALL CAPS in brackets. Output ONLY the transcript text.”

## Banana Mode (High-level)
- Goal: Turn a live webcam capture into one or more AI-generated still images.
- Flow:
  `1. Check if the camera is open; auto-start if not.`
  `2. Capture a frame from the webcam.`
  `3. Send the frame + prompt into Gemini Image (banana_generate).`
  `4. Save the generated images to disk (e.g. banana_*.png) and emit @attachments.`
- Prompts: You can pass one free-form prompt (e.g., "Make this look like a Pixar poster") or rely on the default (“Create a refined, cinematic re-imagination of this photo”).
- Use Cases: Quick style transforms of selfies, turning captures into reference art, moodboard exploration, sketch or poster variations.

## Veo Mode (High-level)
- Goal: Animate a captured still (or Banana-styled variant) into an AI-generated video sequence.
- Flow:
  `1. Check/auto-start the camera.`
  `2. Capture a frame.`
  `3. Optionally transform it with Banana mode into a styled still.`
  `4. Pass the Banana still (or raw capture) plus prompt into Veo 3 (neo_generate_video) as an image-conditioned video generation.`
  `5. Poll until ready, then download the MP4(s) to disk and emit @attachments.`

- Prompts: Either supply one unified prompt (used for both Banana + Veo), or split into banana_prompt (for the still) and veo_prompt (for the animation). You can also add negative_prompt, aspect_ratio, resolution, and seed.
- Use Cases: Animate sketches into motion, cinematic selfie sequences, generate vertical shorts (9:16) or widescreen (16:9) clips, prototype storyboard scenes.

## Requirements
- Banana & Veo features require:
  `GEMINI_API_KEY in your environment`
  `google-genai installed (pip install google-genai)`
  `Veo image-conditioning is wired by passing the Banana (or captured) still into the Veo call.`

## Notes
- macOS: grant Camera permission to your terminal (System Settings → Privacy & Security → Camera).
  macOS: if your iPhone or other cameras are connect you *can* use them using device_index (e.g., 1, 2, ...).
- Linux: ensure user has access to /dev/video* (e.g., add to video group). 
- Windows: try msmf/dshow backends if auto fails. Not tested on Windows yet. WSL2 is a real pain to get your webcam working but it does work in WSL2. I assume Windows works too.
- Outputs are saved as real files (no base64). Commands also print convenient @attachments for follow-up prompts.

## Safety
- No shell execution. Always stop the camera with /vision:stop.
- Get consent before capturing/animating real people. Set sensible defaults (e.g., safe prompts).
