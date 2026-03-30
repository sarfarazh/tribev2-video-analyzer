#!/bin/bash
# TRIBE Analyzer — RunPod setup script
# Run this once after cloning the repo on a new pod.

set -e

VENV_DIR="${HOME}/.venv/tribe"

echo "=== Creating virtual environment ==="
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Installing PyTorch with Blackwell (sm_120) support ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

echo "=== Installing tribev2 (without dependency overrides) ==="
pip install --no-deps "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"

echo "=== Installing core dependencies ==="
pip install gradio>=4.0 openai>=1.0 numpy pandas matplotlib "imageio[ffmpeg]" weasyprint Pillow

echo "=== Installing tribev2 sub-dependencies ==="
pip install neuralset neuraltrain x_transformers einops mne mne_bids nilearn pyvista nibabel \
    safetensors transformers huggingface_hub spacy polars submitit exca confection \
    moviepy pydub soundfile langdetect colorcet julius Levenshtein \
    seaborn scipy scikit-image pydantic tqdm

echo "=== Installing Chatterbox TTS ==="
pip install chatterbox-tts

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-dev ffmpeg

echo ""
echo "=== HuggingFace Login ==="
echo "TRIBE v2 model weights require a HuggingFace account."
echo "Get a token at: https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token: " -s HF_TOKEN
echo ""
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
echo "HuggingFace login complete."

echo ""
echo "=== Setting environment variables ==="
grep -q "TRIBE Analyzer environment" ~/.bashrc 2>/dev/null || {
    cat >> ~/.bashrc << EOF

# TRIBE Analyzer environment
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=4.5
source "$VENV_DIR/bin/activate"
EOF
    echo "Added environment variables and venv activation to ~/.bashrc"
}

export HF_HUB_ENABLE_HF_TRANSFER=0
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=4.5

echo ""
echo "=== Verifying ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
archs = torch.cuda.get_arch_list()
print(f'Architectures: {archs}')
if 'sm_120' in archs:
    print('Blackwell (sm_120): OK')
else:
    print('WARNING: sm_120 not in arch list')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
python -c "import tribev2; print('tribev2: OK')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('Chatterbox TTS: OK')"

echo ""
echo "=== Setup complete! ==="
echo "Venv: $VENV_DIR"
echo "Launch with:"
echo "  source $VENV_DIR/bin/activate"
echo "  python app.py"
