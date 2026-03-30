#!/bin/bash
# TRIBE Analyzer — RunPod setup script
# Run this once after cloning the repo on a new pod.

set -e

echo "=== Installing tribev2 (without torch dependency override) ==="
pip install --no-deps "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"

echo "=== Installing remaining dependencies ==="
pip install gradio>=4.0 openai>=1.0 numpy pandas matplotlib "imageio[ffmpeg]" weasyprint Pillow

echo "=== Installing tribev2 sub-dependencies (excluding torch/torchvision) ==="
pip install neuralset neuraltrain x_transformers einops mne mne_bids nilearn pyvista nibabel \
    safetensors transformers huggingface_hub spacy polars submitit exca confection \
    moviepy pydub soundfile chatterbox-tts langdetect colorcet julius Levenshtein \
    seaborn scipy scikit-image pydantic tqdm

echo "=== Ensuring PyTorch 2.8 + CUDA 12.8 (Blackwell support) ==="
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

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
# Write env vars to .bashrc so they persist across shell sessions
grep -q "HF_HUB_ENABLE_HF_TRANSFER" ~/.bashrc 2>/dev/null || {
    cat >> ~/.bashrc << 'EOF'

# TRIBE Analyzer environment
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=4.5
EOF
    echo "Added environment variables to ~/.bashrc"
}

# Also export for current session
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=4.5

echo ""
echo "=== Verifying ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); assert 'sm_120' in torch.cuda.get_arch_list(), 'Blackwell not supported!'; print('Blackwell (sm_120): OK')"
python -c "import tribev2; print('tribev2: OK')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"

echo ""
echo "=== Setup complete! ==="
echo "Launch with:"
echo "  python app.py"
