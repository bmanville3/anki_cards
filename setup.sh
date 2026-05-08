# Core
pip install genanki

# Word-by-word gloss
pip install fugashi jamdict jamdict-data unidic-lite

# llama.cpp with Metal (Mac GPU)
# just do normal pip install if not on mac
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# Download model
mkdir -p ~/models/
pip install huggingface-hub
if [ ! -f ~/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf ]; then
    hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q5_K_M.gguf --local-dir ~/models
else
  echo "Model already exists, skipping download."
fi

brew install ffmpeg

pip install kokoro-onnx soundfile numpy
# download the model files once:
mkdir -p ~/models/kokoro

if [ ! -f ~/models/kokoro/kokoro-v1.0.onnx ]; then
    curl -L -o ~/models/kokoro/kokoro-v1.0.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
else
    echo "Kokoro model already exists, skipping."
fi

if [ ! -f ~/models/kokoro/voices-v1.0.bin ]; then
    curl -L -o ~/models/kokoro/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
else
    echo "Kokoro voices already exist, skipping."
fi

mkdir -p ./fonts

# Download each font family zip and extract just the ttf files
for font in \
  "NotoSansJP/NotoSansJP-Regular.ttf" \
  "NotoSerifJP/NotoSerifJP-Regular.ttf" \
  "MPLUSRounded1c/MPLUSRounded1c-Regular.ttf" \
  "MPLUS1p/MPLUS1p-Regular.ttf" \
  "KosugiMaru/KosugiMaru-Regular.ttf" \
  "SawarabiGothic/SawarabiGothic-Regular.ttf" \
  "SawarabiMincho/SawarabiMincho-Regular.ttf" \
  "ZenKakuGothicNew/ZenKakuGothicNew-Regular.ttf" \
  "ZenAntique/ZenAntique-Regular.ttf" \
  "ShipporiMincho/ShipporiMincho-Regular.ttf"
do
  name=$(basename "$font")
  if [ ! -f "./fonts/$name" ]; then
      curl -L -o "./fonts/$name" \
        "https://github.com/google/fonts/raw/main/ofl/$font"
  else
      echo "$name already exists, skipping."
  fi
done
