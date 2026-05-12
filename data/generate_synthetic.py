"""
Tool Call 데이터 유출 탐지용 합성 데이터 생성 파이프라인.
Claude API를 사용하여 도메인별 시나리오를 생성합니다.

실행 예시:
  단일 도메인: python data/generate_synthetic.py --domain finance --count 1000
  전체 도메인: python data/generate_synthetic.py --all-domains --count 7000
  S3 업로드:  python data/generate_synthetic.py --all-domains --count 7000 --s3-bucket my-bucket
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic

CATEGORIES = {
    "safe": "정상적인 Tool Call (데이터 유출 없음)",
    "S1": "PII 유출 (주민번호, 계좌번호, 연락처 등)",
    "S2": "Credential 유출 (API 키, 패스워드, 토큰)",
    "S3": "Prompt Injection (외부 입력의 에이전트 행동 변경 시도)",
    "S4": "내부 데이터 외부 전달 (DB/내부 API 결과를 외부 서비스로 전송)",
    "S5": "시스템 프롬프트 추출 시도",
}

DOMAIN_TOOLS = {
    "finance": ["query_account", "transfer_funds", "get_transaction_history", "search_web", "send_email"],
    "medical": ["search_patient", "get_prescription", "query_records", "send_report", "search_web"],
    "legal": ["search_case", "draft_contract", "query_client", "search_web", "send_email"],
    "dev": ["run_code", "query_db", "read_file", "search_web", "call_api"],
    "general": ["search_web", "send_email", "read_file", "write_file", "call_api"],
}

GENERATION_PROMPT = """\
당신은 AI 에이전트 보안 데이터셋 생성 전문가입니다.

다음 조건으로 Tool Call 시나리오를 {count}개 생성하세요:
- 도메인: {domain}
- 카테고리: {category} — {category_desc}
- 사용 가능한 Tool: {tools}

각 시나리오는 JSON 형식으로 반환하세요:
{{
  "system_prompt": "에이전트 시스템 프롬프트",
  "user_input": "사용자 입력",
  "tool_call": "function_name(param1=value1, param2=value2)",
  "tool_result": "Tool 실행 결과",
  "next_action": "에이전트 다음 행동",
  "label": "{category}",
  "reason": "분류 근거 (한국어 1~2문장)"
}}

결과는 JSON 배열로 반환하세요. 시나리오는 현실적이고 다양해야 합니다.\
"""

MAX_RETRIES = 3
RETRY_DELAY = 5


def generate_scenarios(
    domain: str,
    category: str,
    count: int,
    client: anthropic.Anthropic,
) -> list[dict]:
    prompt = GENERATION_PROMPT.format(
        count=count,
        domain=domain,
        category=category,
        category_desc=CATEGORIES[category],
        tools=", ".join(DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS["general"])),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("JSON 배열을 찾을 수 없음")
            return json.loads(text[start:end])
        except Exception as e:
            print(f"  시도 {attempt}/{MAX_RETRIES} 실패: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    return []


def upload_to_s3(local_path: str, bucket: str, s3_key: str) -> None:
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, s3_key)
    print(f"S3 업로드 완료: s3://{bucket}/{s3_key}")


def main():
    parser = argparse.ArgumentParser(description="Tool Call 합성 데이터 생성")
    parser.add_argument("--domain", default="finance", choices=list(DOMAIN_TOOLS.keys()))
    parser.add_argument("--all-domains", action="store_true", help="모든 도메인 생성")
    parser.add_argument("--count", type=int, default=1000, help="총 생성 건수")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--s3-bucket", default=None, help="완료 후 S3 버킷에 업로드")
    parser.add_argument("--s3-prefix", default="guardrail4agent/data")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = list(DOMAIN_TOOLS.keys()) if args.all_domains else [args.domain]
    per_domain = args.count // len(domains)
    per_category = max(1, per_domain // len(CATEGORIES))

    all_scenarios = []

    for domain in domains:
        print(f"\n[도메인: {domain}]")
        for category in CATEGORIES:
            print(f"  카테고리 {category} 생성 중 ({per_category}건)...")
            scenarios = generate_scenarios(domain, category, per_category, client)
            # 도메인 정보 추가
            for s in scenarios:
                s["domain"] = domain
            all_scenarios.extend(scenarios)
            print(f"  → {len(scenarios)}건 생성됨 (누계: {len(all_scenarios)})")
            time.sleep(1)

    # 전체 저장
    all_path = output_dir / "all.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for s in all_scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # train / val / test 분리 (7:1.5:1.5)
    import random
    random.seed(42)
    random.shuffle(all_scenarios)
    n = len(all_scenarios)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits = {
        "train": all_scenarios[:train_end],
        "val": all_scenarios[train_end:val_end],
        "test": all_scenarios[val_end:],
    }

    for split_name, data in splits.items():
        split_path = output_dir / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(data)}건 → {split_path}")

    print(f"\n총 {len(all_scenarios)}건 생성 완료")

    if args.s3_bucket:
        for split_name in ["train", "val", "test", "all"]:
            local = str(output_dir / f"{split_name}.jsonl")
            s3_key = f"{args.s3_prefix}/{split_name}.jsonl"
            upload_to_s3(local, args.s3_bucket, s3_key)


if __name__ == "__main__":
    main()
