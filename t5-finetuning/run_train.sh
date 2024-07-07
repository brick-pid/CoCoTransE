# CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com NCCL_P2P_DISABLE=1 python train.py

HF_ENDPOINT=https://hf-mirror.com NCCL_P2P_DISABLE=1 python train.py aug=False data=julia

# HF_ENDPOINT=https://hf-mirror.com NCCL_P2P_DISABLE=1 python -m debugpy --listen 5678 --wait-for-client train.py 