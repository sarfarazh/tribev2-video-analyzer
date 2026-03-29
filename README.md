# TRIBE Analyzer

A [Gradio](https://www.gradio.app/) web app that analyzes short-form video (for example Instagram Reels or YouTube Shorts) using Meta’s [**TRIBE v2**](https://github.com/facebookresearch/tribev2) brain encoding model. It predicts cortical responses from video, visualizes them on brain surfaces, aggregates activity into functional networks, and sends structured data plus heatmap images to **Claude** (via [OpenRouter](https://openrouter.ai/)) to produce a layperson-friendly report on cognitive engagement.

**Important:** Outputs are *model-based estimates*, not measurements from real brain scans.

---

## What it does

1. **Splits** the uploaded video into **20-second** segments (via ffmpeg).
2. **Runs [TRIBE v2](https://github.com/facebookresearch/tribev2)** on each segment: event extraction, predictions (~20,484 vertices per timestep), and optional transcription-related events.
3. **Aggregates** vertex data through the **Schaefer 400** atlas into **seven Yeo-style networks** (visual, somatomotor, dorsal/ventral attention, limbic, frontoparietal, default mode) — see `networks.py`.
4. **Renders** 5-second interval heatmaps, peak/drop snapshots, and per-segment **GIF** animations.
5. **Builds** a timestamped **JSON** summary (`analysis.py`).
6. **Calls** OpenRouter (OpenAI-compatible API) with **Claude Opus 4.6** and multimodal content (JSON + images) to generate a markdown report, then packages a downloadable **HTML** report (`report.py`).

Maximum video length enforced in the UI: **120 seconds**.

---

## Requirements

| Requirement | Notes |
|-------------|--------|
| **GPU** | Strongly recommended; [TRIBE v2](https://github.com/facebookresearch/tribev2) is GPU-heavy. The project was tested with **~48 GB VRAM** (e.g. RTX 6000 Pro on RunPod). |
| **Python** | **3.12+** recommended. |
| **CUDA** | CUDA 12.x typical for PyTorch stacks. |
| **ffmpeg** | Required for splitting video; often preinstalled on ML images. |
| **Hugging Face** | Access to [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) (token + license acceptance if required). |
| **OpenRouter** | API key for Claude; user enters it in the UI (not stored in the repo). |

---

## Installation

From the `tribe_analyzer` directory:

```bash
pip install -r requirements.txt
```

`requirements.txt` installs [`tribev2[plotting]`](https://github.com/facebookresearch/tribev2) from the official repo, Gradio, the OpenAI client (for OpenRouter), NumPy/Pandas/Matplotlib, and `imageio[ffmpeg]` for GIFs.

Log in to Hugging Face so weights can download:

```bash
huggingface-cli login
```

---

## Run the app

```bash
python app.py
```

By default the server listens on **0.0.0.0:7860** and Gradio **`share=True`** is enabled (public `gradio.live` link). Open the printed URL in a browser, paste your **OpenRouter** API key, upload a video (≤120s), and click **Analyze**.

---

## Project layout

| File | Role |
|------|------|
| `app.py` | Gradio UI, orchestrates the full pipeline, loads the model at startup. |
| `brain.py` | [TRIBE v2](https://github.com/facebookresearch/tribev2) loading, per-segment prediction, atlas aggregation, peaks/drops. |
| `networks.py` | Schaefer region labels → seven functional networks. |
| `video.py` | ffmpeg-based splitting and duration checks. |
| `visuals.py` | Interval heatmaps, peak/drop images, segment GIFs. |
| `analysis.py` | JSON summary construction and file output. |
| `report.py` | OpenRouter multimodal call, markdown report, HTML export. |
| `requirements.txt` | Python dependencies. |
| `docs/PLAN.md` | Design and implementation plan. |
| `docs/RUNPOD_SETUP.md` | Step-by-step RunPod GPU setup, ports, costs. |
| `docs/TROUBLESHOOTING.md` | Solutions for common errors (PyTorch/Blackwell, hf_transfer, NVRTC, etc.). |

---

## Further documentation

- **[facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)** — upstream TRIBE v2 code, paper link, and Colab demo.
- **[docs/RUNPOD_SETUP.md](docs/RUNPOD_SETUP.md)** — GPU pod creation, disk size, port **7860**, HF login, and costs.
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Solutions for PyTorch + Blackwell GPU issues, hf_transfer, CUDA NVRTC, version conflicts, and more.
- **[docs/PLAN.md](docs/PLAN.md)** — Architecture, data flow, and UI design notes.

---

## License and third-party models

[**TRIBE v2**](https://github.com/facebookresearch/tribev2) is provided by Meta under its own terms; use of `facebook/tribev2` on Hugging Face is subject to that model’s license. This repository does not redistribute model weights.
