"""
AWS SageMaker Training Job 실행 스크립트.

사전 준비:
  1. aws configure (또는 IAM Role 설정)
  2. S3에 데이터 업로드:
       python data/generate_synthetic.py --all-domains --s3-bucket <bucket>
  3. ECR에 Docker 이미지 푸시 (또는 HuggingFace DLC 사용):
       ./scripts/build_push_ecr.sh

실행:
  python train_sagemaker.py --config configs/sagemaker.yaml
"""

import argparse
import os
import time

import boto3
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_execution_role(cfg: dict) -> str:
    role = cfg.get("role_arn") or os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        # boto3로 현재 계정 기본 SageMaker 실행 역할 추론
        iam = boto3.client("iam")
        role = iam.get_role(RoleName="AmazonSageMaker-ExecutionRole")["Role"]["Arn"]
    return role


def create_training_job(cfg: dict) -> str:
    sm = boto3.client("sagemaker", region_name=cfg.get("region", "ap-northeast-2"))
    job_name = f"guardrail4agent-sft-{int(time.time())}"

    training_job_params = {
        "TrainingJobName": job_name,
        "RoleArn": get_execution_role(cfg),
        "AlgorithmSpecification": {
            "TrainingImage": cfg["docker_image"],
            "TrainingInputMode": "File",
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": cfg["s3_data_uri"],
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/jsonlines",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": cfg["s3_output_uri"],
        },
        "ResourceConfig": {
            "InstanceType": cfg.get("instance_type", "ml.p3.2xlarge"),
            "InstanceCount": cfg.get("instance_count", 1),
            "VolumeSizeInGB": cfg.get("volume_gb", 100),
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": cfg.get("max_runtime_seconds", 86400),
        },
        "HyperParameters": {
            "config": "/opt/ml/input/data/training/sft_8b_qlora.yaml",
            "sagemaker_program": "src/finetune/sft_train.py",
        },
        "Environment": {
            "PYTHONPATH": "/opt/ml/code",
        },
    }

    # W&B 연동 (선택)
    if os.environ.get("WANDB_API_KEY"):
        training_job_params["Environment"]["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
        training_job_params["Environment"]["WANDB_PROJECT"] = "guardrail4agent"

    sm.create_training_job(**training_job_params)
    print(f"Training job 시작: {job_name}")
    return job_name


def wait_for_job(job_name: str, region: str = "ap-northeast-2") -> None:
    sm = boto3.client("sagemaker", region_name=region)
    print("학습 진행 중... (Ctrl+C로 모니터링 중단, 학습은 계속 진행됩니다)")

    while True:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        print(f"  상태: {status} / {secondary}")

        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(60)

    if status == "Completed":
        output_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
        print(f"완료! 모델 아티팩트: {output_uri}")
    else:
        failure = resp.get("FailureReason", "알 수 없음")
        print(f"실패: {failure}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sagemaker.yaml")
    parser.add_argument("--wait", action="store_true", help="완료될 때까지 대기")
    args = parser.parse_args()

    cfg = load_config(args.config)
    job_name = create_training_job(cfg)

    if args.wait:
        wait_for_job(job_name, cfg.get("region", "ap-northeast-2"))


if __name__ == "__main__":
    main()
