pip install kokoro-onnx
BASE_MODELS_DIR='./models/'
if [ ! -d "${BASE_MODELS_DIR}" ]; then
    echo "${BASE_MODELS_DIR} does not exist or is not a directory."
    exit 1
fi

MODEL_DIR="${BASE_MODELS_DIR}/kokoro"
mkdir -p "${MODEL_DIR}"

MODEL_LOC="${MODEL_DIR}/kokoro-v1.0.onnx"
VOICES_LOC="${MODEL_DIR}/voices-v1.0.bin"

if [ ! -f "${MODEL_LOC}" ]; then
    curl -L --fail -o "${MODEL_LOC}" \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
else
    echo "Kokoro model already exists, skipping."
fi

if [ ! -f "${VOICES_LOC}" ]; then
    curl -L --fail -o "${VOICES_LOC}" \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
else
    echo "Kokoro voices already exist, skipping."
fi
