# Troubleshooting Guide — TRIBE Analyzer

## PyTorch + Blackwell GPU incompatibility (sm_120)

**Symptom:**
```
NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA capability sm_120
is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**Cause:** `tribev2` pins `torch<2.7` in its dependencies. When you install tribev2, pip downgrades PyTorch from 2.8.0+cu128 (which supports Blackwell sm_120) to 2.6.0 (which doesn't).

**Fix:** After installing requirements, upgrade PyTorch back to 2.8:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

pip will warn about version conflicts with tribev2 — this is safe to ignore. The torch API is compatible.

---

## hf_transfer not found (whisperx / uv environment)

**Symptom:**
```
ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1)
but 'hf_transfer' package is not available in your environment.
```

Note the traceback paths: `/workspace/.cache/uv/archive-v0/...` — this means whisperx runs inside its own uv-managed virtual environment, separate from system Python.

**Cause:** RunPod templates set `HF_HUB_ENABLE_HF_TRANSFER=1` globally. The uv environment doesn't have the `hf_transfer` package installed.

**Fix (Option A — recommended):** Disable the fast transfer before launching:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
python app.py
```

**Fix (Option B):** Install into the uv env:

```bash
uv pip install hf_transfer
```

To make Option A permanent, add to your shell profile:

```bash
echo 'export HF_HUB_ENABLE_HF_TRANSFER=0' >> ~/.bashrc
```

---

## libnvrtc.so / CUDA NVRTC errors

**Symptom:**
```
OSError: libnvrtc.so.13: cannot open shared object file: No such file or directory
```

This comes from `torchcodec` during whisperx startup. It tries multiple FFmpeg versions and fails on each.

**Cause:** Missing CUDA NVRTC runtime library matching your CUDA version.

**Fix:** Install NVRTC matching your CUDA version:

```bash
# Check CUDA version
nvcc --version

# Install matching NVRTC (e.g., for CUDA 12.8)
apt-get update && apt-get install -y cuda-nvrtc-12-8
```

**Note:** If you've upgraded PyTorch to 2.8.0+cu128 (see above), this bundles `nvidia-cuda-nvrtc-cu12-12.8.93` as a pip package, which usually resolves the issue. If the error persists after the PyTorch upgrade, the apt-get install is the fallback.

---

## torchvision / torchaudio version conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
that are installed. This behaviour is the source of the following dependency conflicts.
torchaudio 2.8.0+cu128 requires torch==2.8.0, but you have torch 2.6.0 which is incompatible.
```

**Cause:** tribev2 pins older torch, creating a mismatch with pre-installed torchaudio/torchvision.

**Fix:** This is resolved by the PyTorch upgrade step above. After running:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

All three packages will be in sync at version 2.8.0+cu128.

---

## huggingface-cli: command not found

**Symptom:**
```
-bash: huggingface-cli: command not found
```

**Cause:** The HuggingFace Hub package is installed but the CLI binary isn't on PATH.

**Fix:** Use Python directly:

```bash
python -c "from huggingface_hub import login; login()"
```

Paste your token when prompted. Get one at https://huggingface.co/settings/tokens

---

## Out of Memory (OOM)

Shouldn't happen on 48 GB, but if it does:
- Check no other processes are using the GPU: `nvidia-smi`
- Restart the pod to clear GPU memory

---

## ffmpeg segment errors

If video splitting fails:
- Ensure the video is a standard format (.mp4, .mov, .webm)
- Re-encode first: `ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4`

---

## OpenRouter errors

- Verify your API key is valid at https://openrouter.ai/keys
- Check you have credits/balance on OpenRouter
- The model ID is `anthropic/claude-opus-4-6` — ensure it's available on your plan

---

## Gradio "connection refused"

- Make sure the app is running (`python app.py`)
- If using RunPod proxy, confirm port 7860 is exposed in pod settings
- Use the `share=True` public link as a fallback (enabled by default)

---

## Model download hangs

- Check internet connectivity from the pod: `curl -I https://huggingface.co`
- Re-run `python -c "from huggingface_hub import login; login()"` if your token expired
