# Core
pip install genanki

# Word-by-word gloss
pip install fugashi jamdict jamdict-data unidic-lite

# llama.cpp with Metal (Mac GPU)
# just do normal pip install if not on mac
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# Download model
pip install huggingface-hub
if [ ! -f ~/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf ]; then
    hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q5_K_M.gguf --local-dir ~/models
else
  echo "Model already exists, skipping download."
fi

brew install ffmpeg
