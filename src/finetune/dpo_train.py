"""
DPO (Direct Preference Optimization) 파인튜닝 스크립트.
SFT 모델의 과탐(Over-defense) 감소 및 한국어 근거 설명 품질 향상.
"""

import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from sft_train import resolve_output_dir, resolve_path, supports_flash_attention


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model_id = cfg["base_model_id"]      # SFT 완료된 체크포인트
    use_4bit = cfg.get("use_4bit", False)

    data_path = resolve_path(cfg["data_path"])
    output_dir = resolve_output_dir(cfg["output_dir"])

    print(f"Base model (SFT): {base_model_id}")
    print(f"DPO data: {data_path}")
    print(f"Output: {output_dir}")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO는 left padding 권장

    attn_impl = "flash_attention_2" if supports_flash_attention() else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )

    # DPO는 reference model이 필요 (SFT 모델을 frozen ref로 사용)
    model_ref = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPO 데이터셋 형식: {"prompt": ..., "chosen": ..., "rejected": ...}
    dataset = load_dataset("json", data_files=data_path, split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("epochs", 1),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum", 16),
        learning_rate=cfg.get("lr", 5e-5),
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        beta=cfg.get("beta", 0.1),  # DPO temperature
        max_length=cfg.get("max_seq_length", 1024),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        report_to=report_to,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DPO model saved to {output_dir}")


if __name__ == "__main__":
    main()
