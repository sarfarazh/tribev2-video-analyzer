# RunPod Setup Guide — TRIBE Analyzer

## 1. Create a Pod

1. Go to [RunPod](https://www.runpod.io/) and create a new GPU Pod
2. **GPU**: Select **RTX 6000 Pro** (48 GB VRAM)
3. **Template**: Pick **RunPod PyTorch 2.x** (comes with CUDA 12.x, Python 3.12, ffmpeg)
4. **Disk**: At least **30 GB** container disk (model weights ~1 GB, plus cache and temp files)
5. **Expose ports**: Add **7860** under HTTP ports (for Gradio UI). Alternatively, use `share=True` for a public Gradio link (enabled by default).
6. Click **Deploy**

## 2. Connect to the Pod

Once the pod is running:
- Click **Connect** → **Start Web Terminal** (or use SSH if you prefer)

## 3. Clone and Install

```bash
# Clone the repo
git clone https://github.com/<your-username>/tribe_analyzer.git
cd tribe_analyzer

# Install all dependencies (tribev2 + gradio + openai + everything else)
pip install -r requirements.txt
```

This single command installs:
- `tribev2[plotting]` — Meta's TRIBE v2 brain encoding model (pulled from GitHub)
- `gradio` — Web UI
- `openai` — OpenRouter API client (OpenAI-compatible)
- `imageio[ffmpeg]` — GIF generation
- `numpy`, `pandas`, `matplotlib` — Data processing and plotting

### If ffmpeg is missing

Most RunPod ML templates include ffmpeg. If not:

```bash
apt-get update && apt-get install -y ffmpeg
```

### If CUDA NVRTC is missing

If you see errors about CUDA runtime compilation:

```bash
apt-get install -y cuda-nvrtc-12-8
```

(Adjust the version to match your CUDA — run `nvcc --version` to check.)

## 4. HuggingFace Login

TRIBE v2 model weights are hosted on HuggingFace. You need a HF account with access to `facebook/tribev2`.

```bash
# Login to HuggingFace (one-time)
huggingface-cli login
```

Paste your HuggingFace token when prompted. Get one at https://huggingface.co/settings/tokens

**Note**: Check if `facebook/tribev2` requires you to accept a license agreement on the model page first: https://huggingface.co/facebook/tribev2

## 5. Launch the App

```bash
python app.py
```

On first launch:
- The TRIBE v2 model downloads (~1 GB) and loads into GPU memory (~20 GB VRAM)
- Two PlotBrain instances initialize (standard + Schaefer atlas)
- This takes 1–3 minutes on first run, ~30 seconds on subsequent runs (cached)

You'll see:
```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxxx.gradio.live
```

### Access the UI

**Option A** — Public link (works immediately):
Use the `https://xxxxx.gradio.live` URL printed in the terminal.

**Option B** — RunPod proxy:
If you exposed port 7860, go to your pod's **Connect** tab and click the **7860** port link.

## 6. Using the App

1. Paste your **OpenRouter API key** (get one at https://openrouter.ai/keys)
2. Upload a video (max 120 seconds — optimized for Reels/Shorts)
3. Click **Analyze**
4. Wait for processing (~3–5 minutes for a 60s video):
   - Video splits into 20s segments
   - TRIBE v2 processes each segment
   - Brain heatmaps and GIFs render
   - Claude Opus 4.6 generates the report
5. View results: animated GIFs, heatmap gallery, peak moments, full report
6. Download the HTML report and raw JSON data

## 7. Costs

| Item | Estimate |
|------|----------|
| RunPod RTX 6000 Pro | ~$0.74/hr (community cloud) |
| OpenRouter Claude Opus 4.6 | ~$0.50–1.00 per analysis (JSON + ~15 images) |

**Tip**: Stop the pod when not in use. The model re-downloads quickly from cache on restart.

## 8. Troubleshooting

See the full [Troubleshooting Guide](TROUBLESHOOTING.md) for solutions to common issues including:

- PyTorch + Blackwell GPU incompatibility (sm_120)
- hf_transfer not found (whisperx / uv environment)
- CUDA NVRTC errors
- torchvision / torchaudio version conflicts
- huggingface-cli not found
- OOM, ffmpeg, OpenRouter, and Gradio connection issues
