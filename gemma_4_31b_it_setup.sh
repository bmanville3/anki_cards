vllm serve google/gemma-4-31b-it \
  --max-model-len 65536 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 32 \
  --port 9090
