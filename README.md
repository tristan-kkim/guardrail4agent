# Guardrail4Agent

**LLM 에이전트 Tool Call 데이터 유출 탐지 — 한국어 특화 가드레일 모델 및 데모**

> Fine-tuning Kanana as a Korean-specialized guardrail for detecting data leakage in LLM Agent Tool Calls

[Demo](http://43-203-223-40.nip.io) | [HuggingFace Model](https://huggingface.co/tristan-kim/kanana-guardrail4agent) | [Dataset](https://huggingface.co/datasets/tristan-kim/kanana-guardrail4agent-dataset)

---

## 개요

기존 가드레일 모델(Llama Guard, ShieldLM 등)은 **대화 레벨 안전성**에 집중하여, LLM 에이전트가 외부 Tool(API, DB, 파일 등)을 호출하는 단계에서 발생하는 **데이터 유출**을 탐지하지 못합니다.

본 프로젝트는 Kakao의 한국어 LLM **Kanana**를 파인튜닝하여, Tool Call 단계의 데이터 유출을 탐지하는 한국어 특화 가드레일 모델을 구축합니다.

```
[2단계 에이전트 파이프라인]

사용자 입력
    ↓
Agent LLM (Claude Sonnet 4.6)
    ↓  Tool Call 생성
Guardrail Classifier (Kanana-Guardrail)  ← 이 프로젝트
    ↓  safe / unsafe 판정
Tool 실행 또는 차단
    ↓
최종 응답

[기존 가드레일 공백]
사용자 입력 → LLM → Tool Call 파라미터 → DB / API / 외부 서비스
                           ↑
                     기존 가드레일 미감시 구간
                     (PII 유출, Credential 노출, Prompt Injection 등)
```

---

## 탐지 카테고리

| 코드 | 카테고리 | 예시 |
|------|---------|------|
| safe | 정상 Tool Call | 공개 정보 검색, 일반 코드 실행 |
| S1 | PII 유출 | 주민번호·계좌번호·연락처가 외부 API 파라미터로 노출 |
| S2 | Credential 유출 | API 키·패스워드·JWT 토큰이 Tool Call 로그에 포함 |
| S3 | Prompt Injection | 외부 문서·DB 결과·웹페이지의 숨겨진 명령이 에이전트 동작 변경 시도 |
| S4 | 내부 데이터 외부 전달 | DB 조회 결과가 허가되지 않은 외부 서비스로 전송 시도 |
| S5 | 시스템 프롬프트 추출 | 에이전트의 내부 지시사항·비즈니스 로직 노출 시도 |

**카테고리 우선순위 (복합 위협 시)**: S3 > S5 > S1 > S2 > S4

---

## 성능

### 프로토타입 평가 (Claude Haiku, 50케이스 기준)

| 지표 | 값 |
|------|-----|
| 정확도 | **94.0%** (47/50) |
| F1 Macro | **0.940** |
| FPR (오탐률) | **0.0%** |
| FNR (미탐률) | **0.0%** |
| 평균 지연 | 1,837ms (API 기준) |

### 카테고리별 F1 (프로토타입 v4)

| 카테고리 | F1 |
|---------|-----|
| safe | 1.000 |
| S1 PII | 0.842 |
| S2 Credential | 1.000 |
| S3 Prompt Injection | 1.000 |
| S4 내부 데이터 | 0.800 |
| S5 시스템 추출 | 1.000 |

### 파인튜닝 모델 비교

| 모델 | F1 Macro | FPR | 상태 |
|------|---------|-----|------|
| Llama Guard 3 (영어, 비교군) | ~0.61 | ~11.2% | 한국어 환경 |
| **Kanana-2.1B SFT** | **0.81** | 6.2% | 완료 |
| **Kanana-8B QLoRA SFT** | **0.87** | 8.3% | FPR 개선 중 |
| Kanana-8B DPO | 목표 ≥0.90 | 목표 ≤3% | 학습 중 |

---

## 라이브 데모

**URL**: http://43-203-223-40.nip.io

- **에이전트 데모**: 사전 정의된 6가지 위협 시나리오 선택 → 2단계 파이프라인 실시간 시각화
- **직접 입력**: tool_call, tool_result, next_action 직접 입력 후 분류

| 계정 | 아이디 | 역할 |
|------|-------|------|
| 데모용 1 | tristan@cortexys.team | 관리자 |
| 데모용 2 | test-dankook | 테스트 |

---

## 프로젝트 구조

```
guardrail4agent/
├── data/
│   ├── generate_synthetic.py   # Claude API 기반 합성 데이터 생성
│   ├── validate_labels.py      # 레이블 품질 검증
│   └── synthetic/              # 합성 데이터셋 (7,000건)
├── src/
│   ├── finetune/
│   │   ├── sft_train.py        # LoRA / QLoRA SFT 학습
│   │   └── dpo_train.py        # DPO 파인튜닝
│   ├── guardrail/
│   │   ├── classifier.py       # Kanana 추론 인터페이스
│   │   └── server.py           # FastAPI 가드레일 서버
│   └── evaluate/
│       └── run_tests.py        # 평가 스크립트 (50케이스)
├── demo/
│   ├── app.py                  # 데모 FastAPI 서버 (2단계 파이프라인)
│   ├── static/index.html       # 데모 UI
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── deploy.sh               # EC2 배포 스크립트
├── docs/
│   ├── evaluation_results.md   # 프로토타입 평가 결과 (v1~v4)
│   └── training_log.md         # 학습 실험 기록
├── configs/
│   ├── sft_2.1b.yaml
│   └── sft_8b_qlora.yaml
└── requirements.txt
```

---

## 빠른 시작

### 추론 (파인튜닝 모델)

```bash
pip install -r requirements.txt

python src/guardrail/classifier.py \
  --model tristan-kim/kanana-guardrail4agent \
  --tool_call "search_web(query='주민번호 901215-1234567 조회')" \
  --tool_result "검색 결과: ..." \
  --user_input "신원 확인해줘"
```

### 가드레일 서버 실행

```bash
# prototype 모드 (Claude Haiku)
GUARDRAIL_MODE=prototype \
ANTHROPIC_API_KEY=sk-ant-... \
uvicorn src.guardrail.server:app --port 8000

# finetuned 모드 (Kanana 체크포인트)
GUARDRAIL_MODE=finetuned \
GUARDRAIL_MODEL=tristan-kim/kanana-guardrail4agent \
uvicorn src.guardrail.server:app --port 8000
```

### API 호출 예시

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "당신은 고객 서비스 어시스턴트입니다.",
    "user_input": "김철수 고객 신원 확인해줘",
    "tool_call": "search_web(query=\"김철수 주민번호 901215-1234567\")",
    "tool_result": "검색 결과: 없음",
    "next_action": ""
  }'
```

응답:
```json
{
  "is_safe": false,
  "category": "S1",
  "category_desc": "PII 유출",
  "reason": "주민등록번호가 외부 검색 쿼리 파라미터에 직접 노출되었습니다.",
  "latency_ms": 487.3,
  "model": "tristan-kim/kanana-guardrail4agent"
}
```

---

## 학습 데이터

- **규모**: 7,000건 (train 4,900 / eval 700 / test 1,400)
- **도메인**: 금융, 의료, 법률, 개발, HR, 일반
- **카테고리 분포**: safe(30%) / S1~S5 각 14%
- **생성 방법**: Claude API 기반 합성 데이터 생성 후 전문가 검수

---

## 모델 아키텍처

| 항목 | 값 |
|------|-----|
| 베이스 모델 | kakaoai/kanana-nano-2.1b-instruct |
| 파인튜닝 방법 | LoRA SFT (r=16, α=32) |
| Target modules | q, k, v, o (Attention 전체) |
| Dropout | 0.05 |
| 학습 인프라 | AWS ml.g5.2xlarge (A10G 24GB) |
| 프레임워크 | HuggingFace Transformers, PEFT, TRL |

---

## 데모 배포

```bash
# 환경 변수 설정 (.env)
ANTHROPIC_API_KEY=sk-ant-...
HF_API_TOKEN=hf_...
HF_MODEL_ID=tristan-kim/kanana-guardrail4agent
JWT_SECRET=your-secret-key
DAILY_QUOTA=50
IP_DAILY_LIMIT=2

# Docker로 실행
cd demo
docker-compose up -d
```

---

## 관련 논문

- Meta AI, "Llama Guard 3" (2024)
- Li et al., "InjecGuard" arXiv:2410.22770 (2024)
- Hackett et al., "Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails" arXiv:2504.11168 (2025)
- de Witt, "Open Challenges in Multi-Agent Security" arXiv:2505.02077 (2025)
- Kakao AI, "Kanana: Compute-efficient Bilingual Language Models" (2025)

---

## 라이선스

본 프로젝트는 연구 목적으로 공개됩니다. 베이스 모델(Kanana) 라이선스를 준수하세요.
