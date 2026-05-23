BASE_MODELS_DIR='./models/'
if [ ! -d "${BASE_MODELS_DIR}" ]; then
    echo "${BASE_MODELS_DIR} does not exist or is not a directory."
    exit 1
fi
cd "${BASE_MODELS_DIR}"
if [ ! -d "fish-speech" ]; then
    git clone https://github.com/fishaudio/fish-speech.git
fi
cd fish-speech
git pull
pip install -e ".[stable]"
if [ ! -d "checkpoints/s2-pro" ]; then
    hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
fi
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 0.0.0.0:8080 \
  --half
