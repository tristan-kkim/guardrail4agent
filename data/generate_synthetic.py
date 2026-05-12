"""
Tool Call 데이터 유출 탐지용 합성 데이터 생성 파이프라인.
Claude API를 사용하여 도메인별 시나리오를 생성합니다.
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

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text
    # JSON 블록 추출
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return []


def main():
    parser = argparse.ArgumentParser(description="Tool Call 합성 데이터 생성")
    parser.add_argument("--domain", default="finance", choices=list(DOMAIN_TOOLS.keys()))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default="data/synthetic/generated.jsonl")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_category = max(1, args.count // len(CATEGORIES))
    all_scenarios = []

    for category in CATEGORIES:
        print(f"Generating {per_category} scenarios: domain={args.domain}, category={category}")
        scenarios = generate_scenarios(args.domain, category, per_category, client)
        all_scenarios.extend(scenarios)
        print(f"  → {len(scenarios)} generated")
        time.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        for scenario in all_scenarios:
            f.write(json.dumps(scenario, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_scenarios)} scenarios saved to {output_path}")


if __name__ == "__main__":
    main()
