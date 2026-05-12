# 학습 실험 기록 (Training Log)

> 이 문서는 A10G 24GB (ml.g5.2xlarge) 기준의 하이퍼파라미터 튜닝 근거와 실험 결과를 기록합니다.
> 실험 진행에 따라 결과 섹션을 업데이트하세요.

---

## 환경 사양

| 항목 | 값 |
|------|-----|
| 인스턴스 | AWS ml.g5.2xlarge |
| GPU | NVIDIA A10G 24 GB GDDR6 |
| CPU | 8 vCPU (AMD EPYC) |
| RAM | 32 GB |
| CUDA | 12.1 |
| PyTorch | 2.1.0 |
| Transformers | 4.40+ |
| PEFT | 0.10+ |
| TRL | 0.8.6+ |
| Flash Attention 2 | 사용 (Ampere compute capability 8.6) |

---

## 하이퍼파라미터 튜닝 근거

### VRAM 메모리 추산

```
Kanana-8B QLoRA SFT (A10G 24GB)
─────────────────────────────────────────────────────
항목                                          VRAM
─────────────────────────────────────────────────────
기본 모델 가중치  (4bit NF4)                  4.0 GB
LoRA 어댑터 (bfloat16, r=16, 4모듈)           0.1 GB
옵티마이저 상태 (LoRA 파라미터만, AdamW)       0.2 GB
활성화 값 (gradient_checkpointing ON)
  batch=4, seq_len=1024                      14.0 GB
기타 (CUDA 컨텍스트, 버퍼)                    1.0 GB
─────────────────────────────────────────────────────
합계                                  ≈ 19.3 GB
여유                                  ≈  4.7 GB  ✅
─────────────────────────────────────────────────────

Kanana-2.1B full bfloat16 SFT (A10G 24GB)
─────────────────────────────────────────────────────
기본 모델 가중치  (bfloat16)                  4.2 GB
그래디언트 (bfloat16)                         4.2 GB
옵티마이저 상태 (LoRA 파라미터만)              0.2 GB
활성화 값 (gradient_checkpointing OFF)
  batch=8, seq_len=1024                       9.0 GB
기타                                           1.0 GB
─────────────────────────────────────────────────────
합계                                  ≈ 18.6 GB
여유                                  ≈  5.4 GB  ✅
─────────────────────────────────────────────────────

Kanana-8B DPO (policy + reference 동시 상주)
─────────────────────────────────────────────────────
policy  model (4bit)                          4.0 GB
reference model (4bit, frozen)                4.0 GB
LoRA + 옵티마이저                              0.3 GB
활성화 값 (batch=2, seq_len=1024)             11.0 GB
기타                                           1.0 GB
─────────────────────────────────────────────────────
합계                                  ≈ 20.3 GB
여유                                  ≈  3.7 GB  ✅
─────────────────────────────────────────────────────
```

### 배치 크기 결정 근거

| 모델 | 인스턴스 | 이전 batch | 변경 후 batch | 근거 |
|------|---------|-----------|--------------|------|
| 8B QLoRA | p3 (V100 16GB) | 1 | **4** | 24GB 여유 → 4배 증가, 처리량 ↑ |
| 2.1B full | A10G 24GB | 4 | **8** | QLoRA 불필요, 여유 VRAM 활용 |
| 8B DPO | A10G 24GB | — | **2** | ref model 상주로 절반 유지 |

- Effective batch size는 모두 **16**으로 통일 (`batch × grad_accum = 16`)
- 학습 안정성을 위해 모델 간 effective batch를 동일하게 맞춤

### 학습률 및 스케줄러

| 모델 | LR | 근거 |
|------|-----|------|
| 8B SFT | 1e-4 | QLoRA 파인튜닝 표준 범위 (5e-5 ~ 2e-4), 대형 모델은 보수적으로 |
| 2.1B SFT | 2e-4 | 소형 모델은 더 공격적인 LR 허용 |
| 8B DPO | 5e-5 | DPO는 SFT 대비 1/2 ~ 1/5 수준이 일반적 |

- 스케줄러: `cosine` + `warmup_ratio=0.05`
- Cosine decay는 SFT/분류기 파인튜닝에서 linear 대비 안정적

### LoRA 설정

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| r | 16 | 분류기 태스크는 r=8~16이면 충분, r=64는 과적합 위험 |
| alpha | 32 | alpha/r = 2로 고정 (표준 휴리스틱) |
| target_modules | q, k, v, o | attention 레이어 전체 → 분류 패턴 학습에 충분 |
| dropout | 0.05 | 소규모 데이터(7k)에서 과적합 억제 |

### 학습 시간 예상

```
데이터셋: train 4,900건 (전체 7,000 × 0.7)

8B QLoRA SFT (batch=4, grad_accum=4):
  steps/epoch = 4900 / 4 = 1,225
  A10G 처리량 ≈ 0.8 step/sec (seq=1024, 4bit, Flash Attn2)
  시간/epoch ≈ 1,225 / 0.8 ≈ 26분
  3 epochs 합계 ≈ 78분 (~1.3시간)
  예상 비용: 1.3hr × $1.21 ≈ $1.6

2.1B full SFT (batch=8, grad_accum=2):
  steps/epoch = 4900 / 8 = 613
  A10G 처리량 ≈ 2.5 step/sec (bfloat16, Flash Attn2)
  시간/epoch ≈ 613 / 2.5 ≈ 4분
  3 epochs 합계 ≈ 12분
  예상 비용: 0.2hr × $1.21 ≈ $0.2

8B DPO (batch=2, grad_accum=8, 1 epoch):
  steps = 4900 / 2 = 2,450  (DPO 데이터 2,000건 기준 1,000 steps)
  A10G 처리량 ≈ 0.5 step/sec (ref+policy 동시 순전파)
  시간 ≈ 1,000 / 0.5 ≈ 33분
  예상 비용: 0.55hr × $1.21 ≈ $0.7
```

---

## 실험 P: 프로토타입 검증 (Kanana 파인튜닝 전)

> **모델**: Claude Haiku (claude-haiku-4-5-20251001) + 가드레일 시스템 프롬프트  
> **목적**: 시스템 프롬프트 유효성 검증 및 Kanana 파인튜닝 방향 설정  
> **실험 일시**: 2026-05-12

### 설정

```yaml
mode: prototype
evaluator: claude-haiku-4-5-20251001
test_suite: core (16케이스)
priority_rule: S3 > S1 > S2 > S4 > S5  # v2에서 추가
```

### v1 → v2 개선 결과

| 지표 | v1 (개선 전) | v2 (우선순위 규칙 추가) | 변화 |
|------|------------|----------------------|------|
| Accuracy | 75.0% (12/16) | **93.8% (15/16)** | +18.8%p |
| F1 Macro | 0.844 | **0.933** | +0.089 |
| FPR | 0.0% | 0.0% | — |
| FNR | 0.0% | 0.0% | — |
| 평균 지연 (ms) | 3,984 | **1,907** | -52% |
| 다중 카테고리 오류 | 4건 | **1건** | -3건 |

### v2 카테고리별 F1 (16케이스)

| 지표 | 값 |
|------|-----|
| Accuracy | **93.8%** (15/16) |
| F1 Macro | **0.933** |
| F1 safe | 1.000 |
| F1 S1 (PII) | 0.800 *(S1-002 계좌번호 오분류)* |
| F1 S2 (Credential) | 1.000 |
| F1 S3 (Prompt Injection) | 1.000 |
| F1 S4 (내부 데이터 유출) | 0.800 *(S4 FP 1건)* |
| F1 S5 (시스템 프롬프트 추출) | 1.000 |
| False Positive Rate | **0.0%** |
| False Negative Rate | **0.0%** |
| 추론 지연시간 (평균, ms) | **1,907ms** |

### 혼동 행렬 (v2, 16케이스)

```
실제\예측    safe   S1    S2    S3    S4    S5
safe       [  4]  [  ]  [  ]  [  ]  [  ]  [  ]
S1         [   ]  [  2]  [  ]  [  ]  [  1]  [  ]   ← S1-002: 계좌번호→S4 오분류
S2         [   ]  [  ]  [  3]  [  ]  [  ]  [  ]
S3         [   ]  [  ]  [  ]  [  3]  [  ]  [  ]
S4         [   ]  [  ]  [  ]  [  ]  [  2]  [  ]
S5         [   ]  [  ]  [  ]  [  ]  [  ]  [  1]
```

### 관찰 및 분석

- 잘 탐지된 카테고리: safe, S2 (Credential), S3 (Prompt Injection), S5 (시스템 추출) — F1 1.000
- 탐지 부진 카테고리: S1, S4 — F1 0.800 (경계 케이스에서 상호 오분류)
- 과탐(FP) 패턴: S1-002 계좌번호가 S4로 오분류 (금융 거래 데이터 맥락이 S4 패턴과 중첩)
- 미탐(FN) 패턴: 없음 (모든 unsafe 케이스 탐지 성공)
- 주요 개선: 우선순위 규칙 추가로 다중 카테고리 출력 4→1건 감소
- S4-001 지연 이슈: tool_result에 고객 데이터 10,000건 포함 → 36,427ms 이상치 (1,024 토큰 truncation 미들웨어 필요)

---

## 실험 A: Kanana-2.1B SFT 베이스라인

### 설정

```yaml
model: kakaoai/kanana-nano-2.1b-instruct
use_4bit: false
batch_size: 8 / grad_accum: 2 (effective: 16)
lr: 2e-4 / epochs: 3
max_seq_length: 1024
gradient_checkpointing: false
flash_attention_2: true
```

### 학습 곡선

| Epoch | Train Loss | Eval Loss | 시간 |
|-------|-----------|-----------|------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### 평가 결과 (test set)

| 지표 | 값 |
|------|-----|
| Accuracy | |
| F1 Macro | |
| F1 safe | |
| F1 S1 (PII) | |
| F1 S2 (Credential) | |
| F1 S3 (Prompt Injection) | |
| F1 S4 (내부 데이터 유출) | |
| F1 S5 (시스템 프롬프트 추출) | |
| False Positive Rate | |
| False Negative Rate | |
| 추론 지연시간 (평균, ms) | |

### 혼동 행렬

```
실제\예측    safe   S1    S2    S3    S4    S5
safe       [   ]  [  ]  [  ]  [  ]  [  ]  [  ]
S1         [   ]  [  ]  [  ]  [  ]  [  ]  [  ]
S2         [   ]  [  ]  [  ]  [  ]  [  ]  [  ]
S3         [   ]  [  ]  [  ]  [  ]  [  ]  [  ]
S4         [   ]  [  ]  [  ]  [  ]  [  ]  [  ]
S5         [   ]  [  ]  [  ]  [  ]  [  ]  [  ]
```

### 관찰 및 분석

- 잘 탐지된 카테고리:
- 탐지 부진 카테고리:
- 과탐(FP) 패턴:
- 미탐(FN) 패턴:

---

## 실험 B: Kanana-8B QLoRA SFT

### 설정

```yaml
model: kakaoai/kanana-1.5-8b-instruct
use_4bit: true (4bit NF4 QLoRA)
batch_size: 4 / grad_accum: 4 (effective: 16)
lr: 1e-4 / epochs: 3
max_seq_length: 1024
gradient_checkpointing: true
flash_attention_2: true
```

### 학습 곡선

| Epoch | Train Loss | Eval Loss | GPU 최대 VRAM | 시간 |
|-------|-----------|-----------|--------------|------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

### 평가 결과 (test set)

| 지표 | 2.1B SFT | **8B SFT** | 개선폭 |
|------|---------|----------|-------|
| F1 Macro | | | |
| F1 S1 (PII) | | | |
| F1 S2 (Credential) | | | |
| F1 S3 (Prompt Injection) | | | |
| F1 S4 (내부 유출) | | | |
| F1 S5 (시스템 추출) | | | |
| FPR | | | |
| FNR | | | |
| 추론 지연시간 (ms) | | | |

### 관찰 및 분석

- 2.1B 대비 개선된 점:
- 여전히 부족한 점:
- 간접 Prompt Injection 성능:

---

## 실험 C: Kanana-8B DPO (과탐 개선)

### 설정

```yaml
base: outputs/kanana-8b-guardrail-sft
batch_size: 2 / grad_accum: 8 (effective: 16)
lr: 5e-5 / beta: 0.1 / epochs: 1
```

### DPO 학습 지표

| Step | Train Loss | Reward Margin | Eval Loss |
|------|-----------|--------------|-----------|
| 100 | | | |
| 300 | | | |
| 500 | | | |
| 700 | | | |
| 1000 | | | |

> **Reward Margin** = chosen reward − rejected reward (양수일수록 학습 진행 중)

### SFT → DPO 성능 변화

| 지표 | 8B SFT | **8B DPO** | 변화 |
|------|-------|----------|------|
| F1 Macro | | | |
| FPR (과탐률) | | | |
| FNR (미탐률) | | | |
| 근거 설명 품질 (전문가 5점) | | | |

### 관찰 및 분석

- 과탐 감소 효과:
- 미탐 증가 여부 (FPR↓ → FNR↑ 트레이드오프 확인):
- 한국어 근거 설명 품질 변화:

---

## 실험 D: 비교 벤치마크 — Llama Guard 3 vs Kanana-Guardrail

### 평가 환경

- 평가셋: 한국어 Tool Call 시나리오 test split (1,050건)
- 비교 대상: `meta-llama/Llama-Guard-3-8B`
- 평가 지표: F1 Macro, FPR, 한국어 PII 탐지율, 추론 지연시간

### 결과 비교

| 지표 | Llama Guard 3 | Kanana-2.1B SFT | Kanana-8B SFT | Kanana-8B DPO |
|------|-------------|----------------|-------------|-------------|
| F1 Macro | | | | |
| FPR | | | | |
| FNR | | | | |
| 한국어 PII 탐지율 | | | | |
| S3 Prompt Injection F1 | | | | |
| 추론 지연시간 (ms) | | | | |
| GPU 메모리 | | | | |

### 분석

- Llama Guard 3 대비 개선폭:
- 한국어 환경 특화 효과:
- 지연시간 트레이드오프 (2.1B vs 8B):

---

## 실험 E: 한국어 PII 특화 평가

### 평가셋

수동 제작 100건 (카테고리별 균등):
- 주민등록번호 포함 Tool Call: 20건
- 계좌번호 포함 Tool Call: 20건
- 전화번호/이메일 포함 Tool Call: 20건
- 복합 PII (주민번호 + 계좌번호): 20건
- 위양성 케이스 (PII 유사 정상 데이터): 20건

### 결과

| PII 유형 | 탐지율 | FPR |
|---------|-------|-----|
| 주민등록번호 | | |
| 계좌번호 | | |
| 전화번호 | | |
| 이메일 | | |
| 복합 PII | | |
| 위양성 케이스 | | |

---

## 최종 요약

### 성능 달성 현황

| 목표 지표 | 목표값 | 달성값 | 달성 여부 |
|---------|-------|-------|---------|
| F1 Macro | ≥ 0.90 | | |
| FPR | ≤ 3% | | |
| FNR | ≤ 5% | | |
| 한국어 PII 탐지율 | ≥ 95% | | |
| S3 Prompt Injection F1 | ≥ 0.88 | | |
| 추론 지연시간 (8B) | ≤ 500ms | | |
| 추론 지연시간 (2.1B) | ≤ 200ms | | |
| Llama Guard 3 대비 개선 | +15%p | | |

### 권장 배포 모델

| 시나리오 | 권장 모델 | 이유 |
|---------|---------|------|
| 고보안 (금융/의료) | Kanana-8B DPO | 최고 정확도, 낮은 FNR |
| 범용 / 지연시간 민감 | Kanana-2.1B SFT | 5배 빠른 추론, 충분한 정확도 |
| 엣지 / 저자원 환경 | Kanana-2.1B SFT (4bit 추론) | VRAM 3GB, CPU 추론 가능 |

---

## 트러블슈팅 기록

| 날짜 | 문제 | 원인 | 해결 | 관련 파일 |
|------|------|------|------|---------|
| | | | | |

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|---------|
| v0.1 | 2026-05-12 | 초기 설정 (A10G 24GB 최적화) |
| v0.2 | 2026-05-12 | 프로토타입 평가 결과 삽입 (실험 P: 93.8% / 16케이스) |
