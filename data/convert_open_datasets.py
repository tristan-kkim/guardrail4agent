"""
오픈 데이터셋 → Tool Call 형식 변환 파이프라인.

지원 데이터셋:
  beavertails  : PKU-Alignment/BeaverTails (privacy/data 카테고리 → S1, S4)
  wildguard    : allenai/wildguardmix (adversarial → S3, safe)
  injecagent   : qiuhuachuan/InjecAgent (indirect injection → S3)
  toolbench    : ToolBench/ToolBench (safe Tool Call 패턴)

실행 예시:
  python data/convert_open_datasets.py --dataset injecagent --count 800
  python data/convert_open_datasets.py --dataset beavertails --count 1500
  python data/convert_open_datasets.py --all --output-dir data/converted
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic

MAX_RETRIES = 3
RETRY_DELAY = 5

# 카테고리별 라벨 매핑
BEAVERTAILS_LABEL_MAP = {
    "privacy_violations": "S1",
    "non_consensual_sharing": "S4",
    "illegal_activities_harming_individuals": "S2",
    "financial_crimes": "S4",
    "cybercrime": "S2",
    "violence": None,         # 제외
    "hate_speech": None,      # 제외
}

WILDGUARD_LABEL_MAP = {
    True:  "safe",   # is_safe=True
    False: "S3",     # adversarial → S3 기본 (세분화는 Claude로 재분류)
}


# ── 변환 프롬프트 ─────────────────────────────────────────────────────────

CONVERSION_PROMPT = """\
아래 대화를 AI 에이전트 Tool Call 보안 시나리오로 변환해주세요.

원본 대화:
질문: {question}
답변: {answer}
원본 카테고리: {orig_category}
목표 보안 레이블: {label}

변환 규칙:
- 대화 내용을 한국어 Tool Call 시나리오로 재구성
- tool_call은 함수명(파라미터=값) 형식
- 보안 위협이 있는 경우 tool_call 파라미터에 자연스럽게 포함
- 레이블이 {label}인 이유를 reason에 명시

아래 JSON 형식으로만 반환하세요:
{{
  "system_prompt": "에이전트 역할 설명 (1~2문장)",
  "user_input": "사용자 요청 (한국어)",
  "tool_call": "function_name(param=값)",
  "tool_result": "Tool 실행 결과",
  "next_action": "에이전트 다음 행동",
  "label": "{label}",
  "reason": "분류 근거 (한국어 1~2문장)",
  "source": "{source}"
}}\
"""

INJECAGENT_CONVERSION_PROMPT = """\
아래 간접 Prompt Injection 공격을 한국어 Tool Call 시나리오로 변환해주세요.

원본 Tool: {tool}
원본 지시: {instruction}
삽입된 악성 프롬프트: {injected}

변환 규칙:
- tool_result 안에 injected 내용을 자연스럽게 숨김
- 한국 도메인으로 로컬라이징 (영문 서비스명은 한국어로 교체)
- label은 반드시 "S3"

아래 JSON 형식으로만 반환하세요:
{{
  "system_prompt": "에이전트 역할 (한국어)",
  "user_input": "사용자 요청 (한국어)",
  "tool_call": "tool_name(param=값)",
  "tool_result": "정상 결과 + 숨겨진 악성 명령 포함 (한국어)",
  "next_action": "에이전트가 악성 명령을 따르는 행동",
  "label": "S3",
  "reason": "tool_result에 삽입된 명령이 에이전트 행동을 변경하려는 시도",
  "source": "injecagent"
}}\
"""


def _call_claude(prompt: str, client: anthropic.Anthropic) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1:
                raise ValueError("JSON 객체 없음")
            return json.loads(text[start:end])
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return None


# ── 데이터셋별 변환 함수 ────────────────────────────────────────────────────

def convert_beavertails(count: int, client: anthropic.Anthropic, output_dir: Path) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] datasets 패키지 필요: pip install datasets")
        return 0

    print(f"  BeaverTails 로딩 중...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train", streaming=True)

    results = []
    target_categories = set(k for k, v in BEAVERTAILS_LABEL_MAP.items() if v is not None)

    for row in ds:
        if len(results) >= count:
            break

        # 해당 카테고리 필터
        matched_cat = None
        for cat in target_categories:
            if row.get("category", {}).get(cat, False):
                matched_cat = cat
                break
        if not matched_cat:
            continue

        label = BEAVERTAILS_LABEL_MAP[matched_cat]
        prompt = CONVERSION_PROMPT.format(
            question=row["prompt"][:300],
            answer=row["response"][:300],
            orig_category=matched_cat,
            label=label,
            source="beavertails",
        )

        converted = _call_claude(prompt, client)
        if converted:
            results.append(converted)
            if len(results) % 50 == 0:
                print(f"    {len(results)}/{count}건 변환 완료")
            time.sleep(0.3)

    out_path = output_dir / "beavertails.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  BeaverTails 변환 완료: {len(results)}건 → {out_path}")
    return len(results)


def convert_wildguard(count: int, client: anthropic.Anthropic, output_dir: Path) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] datasets 패키지 필요")
        return 0

    print(f"  WildGuardMix 로딩 중...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", streaming=True)

    results = []
    safe_count = 0
    unsafe_count = 0
    safe_target = count // 2
    unsafe_target = count - safe_target

    for row in ds:
        if len(results) >= count:
            break

        is_safe = row.get("prompt_harm_label") == "unharmful"
        if is_safe and safe_count >= safe_target:
            continue
        if not is_safe and unsafe_count >= unsafe_target:
            continue

        label = "safe" if is_safe else "S3"
        prompt = CONVERSION_PROMPT.format(
            question=row.get("prompt", "")[:300],
            answer=row.get("response", "해당 없음")[:200],
            orig_category="unharmful" if is_safe else "adversarial",
            label=label,
            source="wildguard",
        )

        converted = _call_claude(prompt, client)
        if converted:
            results.append(converted)
            if is_safe:
                safe_count += 1
            else:
                unsafe_count += 1
            if len(results) % 50 == 0:
                print(f"    {len(results)}/{count}건 (safe={safe_count}, unsafe={unsafe_count})")
            time.sleep(0.3)

    out_path = output_dir / "wildguard.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  WildGuard 변환 완료: {len(results)}건 → {out_path}")
    return len(results)


def convert_injecagent(count: int, client: anthropic.Anthropic, output_dir: Path) -> int:
    """InjecAgent는 JSON 파일 직접 다운로드 후 변환."""
    import urllib.request

    injecagent_url = (
        "https://raw.githubusercontent.com/qiuhuachuan/InjecAgent/main/data/test_cases_dh.json"
    )
    cache_path = output_dir / "_injecagent_raw.json"

    if not cache_path.exists():
        print(f"  InjecAgent 다운로드 중...")
        try:
            urllib.request.urlretrieve(injecagent_url, cache_path)
        except Exception as e:
            print(f"  [ERROR] 다운로드 실패: {e}")
            print(f"  수동 다운로드: {injecagent_url} → {cache_path}")
            return 0

    with open(cache_path, encoding="utf-8") as f:
        raw = json.load(f)

    items = raw if isinstance(raw, list) else raw.get("data", [])
    results = []

    for item in items[:count]:
        prompt = INJECAGENT_CONVERSION_PROMPT.format(
            tool=item.get("Tool", item.get("tool", "search_web")),
            instruction=item.get("User Instruction", item.get("instruction", ""))[:200],
            injected=item.get("Injected Prompt", item.get("injected_prompt", ""))[:200],
        )

        converted = _call_claude(prompt, client)
        if converted:
            results.append(converted)
            if len(results) % 50 == 0:
                print(f"    {len(results)}/{min(count, len(items))}건 변환 완료")
            time.sleep(0.3)

    out_path = output_dir / "injecagent.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  InjecAgent 변환 완료: {len(results)}건 → {out_path}")
    return len(results)


def convert_toolbench(count: int, client: anthropic.Anthropic, output_dir: Path) -> int:
    """ToolBench에서 민감 정보 없는 정상 Tool Call → safe 케이스 추출."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] datasets 패키지 필요")
        return 0

    print(f"  ToolBench 로딩 중...")
    try:
        ds = load_dataset("ToolBench/ToolBench", split="train", streaming=True)
    except Exception as e:
        print(f"  [ERROR] ToolBench 로드 실패: {e}")
        return 0

    SENSITIVE_KEYWORDS = ["password", "secret", "token", "ssn", "api_key", "private_key",
                          "주민번호", "비밀번호", "계좌", "토큰", "시크릿"]

    results = []
    for row in ds:
        if len(results) >= count:
            break

        conversation = str(row.get("conversations", ""))
        if any(kw in conversation.lower() for kw in SENSITIVE_KEYWORDS):
            continue

        tool_calls = row.get("conversations", [])
        tool_call_str = ""
        tool_result_str = ""
        for turn in tool_calls:
            if turn.get("from") == "tool_call":
                tool_call_str = str(turn.get("value", ""))[:200]
            elif turn.get("from") == "tool_response":
                tool_result_str = str(turn.get("value", ""))[:200]

        if not tool_call_str:
            continue

        entry = {
            "system_prompt": "당신은 도움이 되는 AI 어시스턴트입니다.",
            "user_input": str(row.get("conversations", [{}])[0].get("value", "도와줘"))[:100],
            "tool_call": tool_call_str,
            "tool_result": tool_result_str or "작업 완료",
            "next_action": "결과를 사용자에게 전달합니다.",
            "label": "safe",
            "reason": "정상적인 Tool 사용으로 민감 정보 노출 없음",
            "source": "toolbench",
        }
        results.append(entry)

        if len(results) % 100 == 0:
            print(f"    {len(results)}/{count}건 추출 완료")

    out_path = output_dir / "toolbench.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ToolBench 변환 완료: {len(results)}건 → {out_path}")
    return len(results)


def merge_all(converted_dir: Path, output_dir: Path) -> None:
    """변환된 모든 파일을 병합하고 중복 제거."""
    import random
    all_data = []
    seen = set()

    for path in converted_dir.glob("*.jsonl"):
        if path.name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    key = item.get("tool_call", "") + item.get("user_input", "")
                    if key not in seen:
                        seen.add(key)
                        all_data.append(item)
                except json.JSONDecodeError:
                    pass

    random.seed(42)
    random.shuffle(all_data)

    n = len(all_data)
    splits = {
        "train": all_data[:int(n * 0.70)],
        "val":   all_data[int(n * 0.70):int(n * 0.85)],
        "test":  all_data[int(n * 0.85):],
    }
    for name, data in splits.items():
        out = output_dir / f"converted_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)}건 → {out}")

    print(f"\n병합 완료: 총 {n}건 (중복 제거 후)")


def main():
    parser = argparse.ArgumentParser(description="오픈 데이터셋 Tool Call 형식 변환")
    parser.add_argument("--dataset", choices=["beavertails", "wildguard", "injecagent", "toolbench"])
    parser.add_argument("--all",     action="store_true", help="모든 데이터셋 변환")
    parser.add_argument("--merge",   action="store_true", help="변환 파일 병합만 실행")
    parser.add_argument("--count",   type=int, default=500)
    parser.add_argument("--output-dir", default="data/converted")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_all(output_dir, Path("data/final"))
        return

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    plan = {
        "beavertails": (convert_beavertails, 1500),
        "wildguard":   (convert_wildguard,   1200),
        "injecagent":  (convert_injecagent,  800),
        "toolbench":   (convert_toolbench,   700),
    }

    if args.all:
        total = 0
        for name, (fn, default_count) in plan.items():
            n = args.count if args.count != 500 else default_count
            print(f"\n[{name.upper()}] {n}건 변환 시작")
            total += fn(n, client, output_dir)
        print(f"\n전체 변환 완료: {total}건")
        merge_all(output_dir, Path("data/final"))
    elif args.dataset:
        fn, default_count = plan[args.dataset]
        n = args.count
        print(f"\n[{args.dataset.upper()}] {n}건 변환 시작")
        fn(n, client, output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
