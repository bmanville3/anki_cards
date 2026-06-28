# Install Ollama
brew install ollama

# Enable flash attention and quantized KV cache for better performance
launchctl setenv OLLAMA_FLASH_ATTENTION "1"
launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"

ollama serve

# Pull the model (20GB download)
# need to do this elsewhere
# ollama pull gemma4:31b
