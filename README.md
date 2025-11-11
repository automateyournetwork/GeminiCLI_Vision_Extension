# 🎥 Gemini-CLI Vision Extension

> **Webcam + ASL + AI Image + AI Video — all from Gemini-CLI.**  
> Capture frames, interpret American Sign Language, and transform your webcam feed into AI-generated art or animation — entirely by prompt **or slash command**.

---

## 🧠 What It Does

The **Gemini-CLI Vision Extension** brings real-time visual reasoning to your terminal.

It connects your **webcam (or tethered iPhone)** to Gemini’s **Model Context Protocol (MCP)** ecosystem, enabling natural, multimodal interactions such as:

> “Can you take a capture of me using device 0?”  
> “...and use Banana to transform it into a sketch using a fine-point pen.”  
> “...and then take that sketch and turn it into a music video.”  
> “Let’s chat in ASL — I’ll sign my question.”

You can issue these commands two ways:
1. **Natural Language (NL)** — just *ask* Gemini in plain English.  
2. **Slash Commands (structured)** — use `/vision:*` commands for precise control.

You can even skip the live camera and use **any static image** in your working folder.

---

## ⚙️ Setup

### 1. Install

```bash
gemini extensions install https://github.com/automateyournetwork/GeminiCLI_Vision_Extension.git
```

2. Mac Permissions
If prompted, allow Camera access to your terminal (System Settings → Privacy & Security → Camera).
Your iPhone or other tethered cameras may appear as extra devices — and yes, you can use them.

🖥️ Core Commands
Command	Description
/vision:devices	Discover connected cameras (indexes, resolutions, FPS). Start here.
/vision:start	Open a selected device. You can specify width, height, fps, and backend.
/vision:status	Show whether a camera is open and its properties.
/vision:capture	Capture a single frame and optionally send it directly to Gemini.
/vision:burst	Capture a sequence of frames (for ASL or motion analysis).
/vision:stop	Release the camera safely.

You can run these directly, or ask naturally:

“List my available cameras.”
“Open my iPhone camera and take a photo.”
“Stop the camera.”

🎨 Modalities
1️⃣ Devices
Run:

```bash
/vision:devices
```

Lists all available cameras.

✅ macOS: Build in camera usually device 0; iPhone Camera often shows up as device 1.

Example natural language:

“Show me my connected cameras.”

Then:

```bash
/vision:start camera_index=0 width=640 height=480 fps=15
```

2️⃣ Capture
Take a single frame:

```bash
/vision:capture
```

or naturally:

“Can you take a capture of me using device 0?”

You’ll get a saved image and an @attachment you can reuse in a follow-up turn.

3️⃣ Banana Mode 🍌 (AI Image Generation)
Transform your webcam capture into AI-generated artwork:

```bash
/vision:banana "Turn this into a watercolor portrait"
```

Natural language:

“Take a capture and use Banana to transform it into a sketch using a fine-point pen.”

Behind the scenes:

Captures a frame

Sends it to Gemini 2.5 Flash Image

Saves generated images (e.g. banana_001.png)

Emits @attachments for chaining

Use for:

Style transfers

Poster or thumbnail mockups

Cinematic selfies or sketches

4️⃣ Veo Mode 🎬 (AI Video Generation)
Turn stills or Banana images into short AI videos with Veo 3:

```bash
/vision:veo "Animate this sketch into a short music video"
```

or

“...and then take that sketch and turn it into a music video.”

Uses Banana output (or live capture)

Runs Veo 3.0 / 3.1 for image-conditioned generation

Outputs real .mp4 files

Supports aspect_ratio, resolution, seed, and more

5️⃣ ASL Mode 🤟 (American Sign Language)
Chat in ASL directly through your webcam:

```bash
/vision:asl
```

Gemini:

Captures a short burst of frames

Transcribes your signing

Responds naturally in English

Use /vision:asl_veo to go further:

Understands your ASL input

Generates an ASL gloss reply

Animates a generic avatar replying in ASL using Veo

Example:

“Let’s chat in ASL — I’ll sign my question.”

🔄 Typical Flow
```bash
/vision:devices
/vision:start camera_index=0
/vision:capture
/vision:banana "Make this look like a Pixar movie poster"
/vision:veo "Animate the poster into a trailer opening"
/vision:stop
```

Or conversationally:

“Open my main camera, take a selfie, turn it into a Pixar-style poster, and animate it into a short trailer.”

🧩 Architecture
```mermaid
flowchart TD
    A[Camera Device(s)\n/webcam/iPhone/static image] -->|/vision:devices| B[Capture]
    B -->|/vision:capture| C[Banana 🍌 AI Image]
    C -->|/vision:veo| D[Veo 🎬 AI Video]
    B -->|/vision:asl| E[ASL 🤟 Interpreter]
    E -->|/vision:asl_veo| F[Veo Avatar Reply in ASL]
    B -->|Attachments| G[Gemini Context / Multimodal Chain]

    subgraph Gemini CLI
    A
    B
    C
    D
    E
    F
    G
    end
```

Flow Summary:

/vision:devices — detect cameras

/vision:start — open camera

/vision:capture — grab frame

/vision:banana — AI-stylize image

/vision:veo — animate into video

/vision:asl — communicate via ASL

/vision:asl_veo — reply back in sign language

🧠 Under the Hood
Runs as an MCP server using FastMCP (no HTTP)

Uses OpenCV for frame capture

Uses Google Gemini 2.5 for image reasoning (Banana)

Uses Veo 3 for AI video generation

Uses Gemini Flash multimodal for ASL understanding

Saves all outputs as real files (.jpg, .png, .mp4) — no base64 bloat

GEMINI_API_KEY and GOOGLE_API_KEY (for Veo3) in your environment

🛡️ Safety
Always obtain consent before capturing people.

Stop your camera when done (/vision:stop).

Avoid personal likeness prompts in Veo.

Use safe, creative prompt phrasing.

💡 Example Prompts
Natural Language

“Can you take a capture of me using device 0?”
“Make this capture look like a pencil sketch.”
“Animate this into a 10-second video.”
“Let’s chat in ASL.”

Slash Command

```bash
/vision:capture
/vision:banana "Sketch in fine point pen"
/vision:veo "Turn it into a music video"
/vision:asl_veo duration_ms=20000 aspect_ratio="16:9" resolution="1080p"
```

👤 Author
John Capobianco
Head of Developer Relations — Selector AI

📍 Creator of the Gemini-CLI multimodal suite:
/talk, /listen, /vision, /computeruse, /packet_buddy, /subnetcalculator, /file_search

“The CLI is dead — long live the multimodal CLI.”