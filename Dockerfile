# EC2 GPU 인스턴스 직접 실행용 Dockerfile
# 베이스: HuggingFace PyTorch DLC (CUDA 12.1, PyTorch 2.1)
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04

WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flash Attention 2 (Ampere GPU 전용 — g5/p4d 인스턴스)
# V100(p3) 사용 시 이 줄 제거
RUN pip install --no-cache-dir flash-attn --no-build-isolation

COPY . .

ENV PYTHONPATH=/opt/ml/code

ENTRYPOINT ["python", "src/finetune/sft_train.py"]
