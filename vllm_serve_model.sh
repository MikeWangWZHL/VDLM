if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
MODEL_PATH=data/ckpts/PVD-160k-Mistral-7b
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH