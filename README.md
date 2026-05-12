# Kanana-Guardrail

**Agent Tool Call 데이터 유출 탐지를 위한 한국어 특화 가드레일 모델 파인튜닝**

> Fine-tuning Kanana as a Korean-specialized guardrail for detecting data leakage in LLM Agent Tool Calls

---

## 개요

기존 가드레일 모델(Llama Guard, ShieldLM 등)은 **대화 레벨 안전성**에 집중하여, LLM 에이전트가 외부 Tool(API, DB, 파일 등)을 호출하는 단계에서 발생하는 **데이터 유출**을 탐지하지 못합니다.

본 프로젝트는 Kakao의 한국어 LLM **Kanana**를 베이스 모델로 삼아, Tool Call 단계의 데이터 유출을 탐지하는 한국어 특화 가드레일 모델을 파인튜닝합니다.

```
[기존 가드레일 공백]

사용자 입력 → LLM → Tool Call 파라미터 → DB / API / 외부 서비스
                           ↑
                     ❌ 기존 가드레일 미감시 구간
                     (PII 유출, Credential 노출, Prompt Injection 등)
```

---

## 탐지 카테고리

| 코드 | 카테고리 | 예시 |
|------|---------|------|
| S1 | PII 유출 | 주민번호·계좌번호·연락처가 외부 API 파라미터로 노출 |
| S2 | Credential 유출 | API 키·패스워드가 Tool Call 로그에 포함 |
| S3 | Prompt Injection | 외부 문서의 숨겨진 명령이 에이전트 동작 변경 시도 |
| S4 | 내부 데이터 외부 전달 | DB 조회 결과가 외부 서비스로 전송 시도 |
| S5 | 시스템 프롬프트 추출 | 에이전트의 시스템 프롬프트 노출 시도 |

---

## 프로젝트 구조

```
kanana-guardrail/
├── data/
│   ├── generate_synthetic.py   # 합성 데이터 생성 파이프라인
│   ├── validate_labels.py      # 레이블 품질 검증
│   └── synthetic/              # 생성된 데이터셋 (7,000건)
├── src/
│   ├── finetune/
│   │   ├── sft_train.py        # LoRA / QLoRA SFT 학습
│   │   └── dpo_train.py        # DPO 파인튜닝
│   ├── guardrail/
│   │   └── classifier.py       # 추론 인터페이스
│   └── server/
│       └── api.py              # FastAPI 가드레일 서버
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_sft_evaluation.ipynb
├── configs/
│   ├── sft_2.1b.yaml
│   └── sft_8b_qlora.yaml
├── AI_Security_Project_Plan.md # 전체 구축 계획
└── requirements.txt
```

---

## 빠른 시작

```bash
# 환경 설치
pip install -r requirements.txt

# 합성 데이터 생성
python data/generate_synthetic.py --domain finance --count 1000

# SFT 학습 (2.1B)
python src/finetune/sft_train.py --config configs/sft_2.1b.yaml

# 추론
python src/guardrail/classifier.py \
  --tool_call "search_web(query='계좌번호 1234-5678 확인')" \
  --tool_result "검색 결과: ..."
```

---

## 베이스 모델

- **Kanana-1.5-8b-instruct** (kakaoai/kanana-1.5-8b-instruct)
- **Kanana-nano-2.1b** (경량화 비교 실험용)

---

## 기술 스택

- **파인튜닝**: HuggingFace Transformers, PEFT (LoRA/QLoRA), TRL (SFT/DPO)
- **데이터 생성**: Claude API (Anthropic)
- **서빙**: FastAPI, LangChain 미들웨어
- **평가**: sklearn, 전문가 평가 (5점 척도)

---

## 현재 성능 (SFT 베이스라인, 2026-05-12 기준)

| 모델 | F1 Macro | FPR | 상태 |
|------|---------|-----|------|
| Llama Guard 3 (영어, 비교군) | 0.61 | 11.2% | 한국어 환경 |
| Kanana-2.1B SFT | 0.81 | 6.2% | ✅ 완료 |
| Kanana-8B QLoRA SFT | 0.87 | 8.3% | ⚠️ FPR 개선 필요 |
| Kanana-8B DPO | 목표 0.90 | 목표 ≤3% | 🔄 학습 중 |

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
