"""
AWS SageMaker HuggingFace Estimator로 학습 Job을 실행합니다.

사전 준비:
  1. AWS 자격증명 설정:
       aws configure
       (또는) export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...

  2. configs/sagemaker.yaml 수정:
       - s3_bucket: 실제 버킷명
       - role_arn: SageMaker 실행 역할 ARN

  3. S3에 학습 데이터 업로드 (자동 처리, 수동 불필요)

실행:
  # SFT (8B QLoRA, 권장)
  python train_sagemaker.py --mode sft --model 8b

  # SFT (2.1B, 빠른 테스트용)
  python train_sagemaker.py --mode sft --model 2.1b

  # DPO (SFT 완료 후)
  python train_sagemaker.py --mode dpo
"""

import argparse
import os

import boto3
import sagemaker
import yaml
from sagemaker.huggingface import HuggingFace


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_or_create_bucket(region: str, suffix: str = "guardrail4agent") -> str:
    s3 = boto3.client("s3", region_name=region)
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"{account_id}-{suffix}-{region}"
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"S3 버킷 생성: {bucket}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    return bucket


def upload_training_data(bucket: str, region: str) -> str:
    """학습 데이터를 S3에 업로드하고 S3 URI 반환."""
    s3 = boto3.client("s3", region_name=region)
    prefix = "guardrail4agent/data"

    for fname in ["sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl",
                  "sft_8b_qlora.yaml", "sft_2.1b.yaml"]:
        local_path = None
        if fname.endswith(".jsonl"):
            local_path = f"data/synthetic/{fname}"
        else:
            local_path = f"configs/{fname}"

        if os.path.exists(local_path):
            s3.upload_file(local_path, bucket, f"{prefix}/{fname}")
            print(f"  업로드: s3://{bucket}/{prefix}/{fname}")

    return f"s3://{bucket}/{prefix}/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["sft", "dpo"], default="sft")
    parser.add_argument("--model",  choices=["8b", "2.1b"], default="8b",
                        help="sft 모드에서 사용할 모델 크기")
    parser.add_argument("--config", default="configs/sagemaker.yaml")
    parser.add_argument("--wait",   action="store_true", help="완료될 때까지 대기")
    args = parser.parse_args()

    cfg = load_config(args.config)
    region = cfg.get("region", "ap-northeast-2")

    # ── AWS 세션 / IAM 역할 ─────────────────────────────────────────────
    boto_session = boto3.Session(region_name=region)
    sm_session   = sagemaker.Session(boto_session=boto_session)

    role_arn = (
        cfg.get("role_arn")
        or os.environ.get("SAGEMAKER_ROLE_ARN")
        or sagemaker.get_execution_role(sm_session)
    )
    print(f"IAM Role: {role_arn}")

    # ── S3 버킷 및 데이터 업로드 ────────────────────────────────────────
    bucket = cfg.get("s3_bucket") or get_or_create_bucket(region)
    print(f"S3 버킷: {bucket}")

    print("학습 데이터 S3 업로드 중...")
    s3_data_uri = upload_training_data(bucket, region)
    s3_output_uri = f"s3://{bucket}/guardrail4agent/outputs/"

    # ── 모드별 학습 스크립트 / 하이퍼파라미터 선택 ──────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")

    if args.mode == "sft":
        config_file = f"sft_{args.model}_qlora.yaml" if args.model == "8b" else "sft_2.1b.yaml"
        entry_point = "src/finetune/sft_train.py"
        hyperparameters = {
            "config": f"/opt/ml/input/data/training/{config_file}",
            "push-to-hub": "true",
            "hf-repo": "tristan-kim/kanana-guardrail4agent",
        }
        job_name_prefix = f"guardrail-sft-{args.model}"
        instance_type = cfg.get("instance_type", "ml.g5.2xlarge")

    else:  # dpo
        entry_point = "src/finetune/dpo_train.py"
        hyperparameters = {
            "config": "/opt/ml/input/data/training/dpo_8b_qlora.yaml",
            "push-to-hub": "true",
            "hf-repo": "tristan-kim/kanana-guardrail4agent",
        }
        job_name_prefix = "guardrail-dpo"
        instance_type = cfg.get("instance_type", "ml.g5.2xlarge")

        # DPO 데이터 추가 업로드
        s3 = boto3.client("s3", region_name=region)
        for fname in ["dpo_pairs.jsonl", "dpo_8b_qlora.yaml"]:
            local = f"data/synthetic/{fname}" if fname.endswith(".jsonl") else f"configs/{fname}"
            if os.path.exists(local):
                s3.upload_file(local, bucket, f"guardrail4agent/data/{fname}")
                print(f"  업로드: {fname}")

    # HF 토큰 환경변수로 전달 (하이퍼파라미터에 노출 금지)
    environment = {}
    if hf_token:
        environment["HF_TOKEN"] = hf_token

    # ── HuggingFace Estimator 생성 ──────────────────────────────────────
    estimator = HuggingFace(
        entry_point=entry_point,
        source_dir=".",
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        volume_size=cfg.get("volume_gb", 100),
        max_run=cfg.get("max_runtime_seconds", 28800),
        # HuggingFace DLC: PyTorch 2.1 / Transformers 4.36 / CUDA 12.1
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        environment=environment,
        output_path=s3_output_uri,
        sagemaker_session=sm_session,
        # 체크포인트 중간 저장
        checkpoint_s3_uri=f"{s3_output_uri}checkpoints/",
    )

    # ── 학습 시작 ────────────────────────────────────────────────────────
    print(f"\nSageMaker 학습 Job 시작 ({args.mode.upper()} / {instance_type})")
    print(f"데이터: {s3_data_uri}")
    print(f"출력:   {s3_output_uri}")

    estimator.fit(
        inputs={"training": s3_data_uri},
        job_name=None,  # 자동 생성
        wait=args.wait,
        logs="All" if args.wait else None,
    )

    if not args.wait:
        print(f"\n학습 Job이 백그라운드에서 실행 중입니다.")
        print(f"모니터링: AWS Console → SageMaker → Training Jobs")
        print(f"완료 후 모델: https://huggingface.co/tristan-kim/kanana-guardrail4agent")


if __name__ == "__main__":
    main()
