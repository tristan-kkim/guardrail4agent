"""
Guardrail4Agent 연구 결과 시각화
plotly로 인터랙티브 차트 생성 → docs/figures/ 저장
"""

import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

OUT = os.path.join(os.path.dirname(__file__), "../../docs/figures")
os.makedirs(OUT, exist_ok=True)

COLORS = {
    "primary":   "#3182f6",
    "secondary": "#6366f1",
    "safe":      "#0dcaa8",
    "unsafe":    "#f04452",
    "warn":      "#f59e0b",
    "grey":      "#8b95a1",
    "bg":        "#f5f6f8",
    "surface":   "#ffffff",
    "text":      "#191f28",
}

FONT = dict(family="Apple SD Gothic Neo, -apple-system, BlinkMacSystemFont, sans-serif",
            color=COLORS["text"])

BASE_LAYOUT = dict(
    font=FONT,
    plot_bgcolor=COLORS["surface"],
    paper_bgcolor=COLORS["bg"],
    margin=dict(t=70, b=60, l=60, r=40),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family=FONT["family"]),
)


# ── 1. 프롬프트 개선 v1→v4 성능 변화 ─────────────────────────────────────────

def fig_performance_progression():
    versions = ["v1<br>(16건)", "v2<br>(16건)", "v3<br>(50건)", "v4<br>(50건)"]
    accuracy  = [75.0,  93.8,  84.0,  94.0]
    f1_macro  = [84.4,  93.3,  78.8,  94.0]
    fpr       = [0.0,   0.0,  16.7,   0.0]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("정확도 및 F1 Macro 변화", "FPR(오탐률) 변화"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=versions, y=accuracy, name="정확도(%)",
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=10, color=COLORS["primary"],
                    line=dict(color="white", width=2)),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=versions, y=f1_macro, name="F1 Macro(%)",
        mode="lines+markers",
        line=dict(color=COLORS["secondary"], width=3, dash="dot"),
        marker=dict(size=10, color=COLORS["secondary"],
                    line=dict(color="white", width=2)),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=versions, y=fpr, name="FPR(%)",
        marker_color=[COLORS["safe"] if v == 0 else COLORS["unsafe"] for v in fpr],
        text=[f"{v}%" for v in fpr],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    # 목표선
    fig.add_hline(y=90, line_dash="dash", line_color=COLORS["warn"],
                  annotation_text="목표 90%", row=1, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>시스템 프롬프트 반복 개선: v1 → v4 성능 변화</b>",
                   font=dict(size=18), x=0.5),
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=COLORS["grey"], borderwidth=1),
        height=420,
    )
    fig.update_yaxes(range=[60, 102], ticksuffix="%", row=1, col=1)
    fig.update_yaxes(range=[0, 25],  ticksuffix="%", row=1, col=2)
    fig.update_xaxes(tickfont=dict(size=12))

    fig.write_html(f"{OUT}/01_performance_progression.html")
    fig.write_image(f"{OUT}/01_performance_progression.png", width=900, height=420, scale=2)
    print("✓ 01_performance_progression")
    return fig


# ── 2. v4 카테고리별 F1 점수 ─────────────────────────────────────────────────

def fig_category_f1():
    cats   = ["safe", "S1<br>PII 유출", "S2<br>Credential", "S3<br>Prompt<br>Injection",
              "S4<br>내부 데이터<br>외부 전달", "S5<br>시스템 프롬프트<br>추출", "Macro"]
    f1     = [1.000,  0.842,  1.000,  1.000,  0.800,  1.000,  0.940]
    colors = [COLORS["safe"] if v >= 0.95 else
              COLORS["warn"] if v >= 0.85 else
              COLORS["unsafe"] for v in f1]
    colors[-1] = COLORS["primary"]  # Macro

    fig = go.Figure(go.Bar(
        x=cats, y=f1,
        marker=dict(color=colors, line=dict(color="white", width=1.5)),
        text=[f"{v:.3f}" for v in f1],
        textposition="outside",
        textfont=dict(size=13, color=COLORS["text"]),
    ))

    # 목표선
    fig.add_hline(y=0.90, line_dash="dash", line_color=COLORS["warn"],
                  annotation_text="목표 F1 ≥ 0.90", annotation_position="bottom right")

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>v4 카테고리별 F1 점수 (프로토타입, 50케이스)</b>",
                   font=dict(size=18), x=0.5),
        yaxis=dict(range=[0, 1.12], tickformat=".2f", title="F1 Score"),
        xaxis=dict(title="카테고리"),
        height=460,
        showlegend=False,
    )

    fig.write_html(f"{OUT}/02_category_f1_v4.html")
    fig.write_image(f"{OUT}/02_category_f1_v4.png", width=900, height=460, scale=2)
    print("✓ 02_category_f1_v4")
    return fig


# ── 3. 모델 비교 벤치마크 ────────────────────────────────────────────────────

def fig_model_comparison():
    models = [
        "Llama Guard 3<br>(영어, 비교군)",
        "Kanana-2.1B<br>SFT",
        "Kanana-8B<br>QLoRA SFT",
        "Kanana-8B<br>DPO (목표)",
        "프로토타입 v4<br>(Claude Haiku)",
    ]
    f1     = [0.61,  0.81,  0.87,  0.90,  0.940]
    fpr    = [11.2,  6.2,   8.3,   3.0,   0.0]
    latency= [800,   200,   500,   500,   1837]

    model_colors = [
        COLORS["grey"],
        COLORS["secondary"],
        COLORS["primary"],
        COLORS["safe"],
        COLORS["warn"],
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("F1 Macro", "FPR (오탐률, %)", "추론 지연 (ms)"),
        horizontal_spacing=0.1,
    )

    for col, (values, suffix, rng) in enumerate([
        (f1,      "",   [0, 1.05]),
        (fpr,     "%",  [0, 14]),
        (latency, "ms", [0, 2200]),
    ], start=1):
        fig.add_trace(go.Bar(
            x=models, y=values,
            marker=dict(color=model_colors,
                        line=dict(color="white", width=1.5)),
            text=[f"{v}{suffix}" for v in values],
            textposition="outside",
            textfont=dict(size=11),
            showlegend=False,
        ), row=1, col=col)
        fig.update_yaxes(range=rng, row=1, col=col)

    # F1 목표선
    fig.add_hline(y=0.90, line_dash="dash", line_color=COLORS["warn"],
                  annotation_text="목표", row=1, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>모델 비교 벤치마크 (한국어 Tool Call 탐지)</b>",
                   font=dict(size=18), x=0.5),
        height=460,
    )
    fig.update_xaxes(tickfont=dict(size=10))

    fig.write_html(f"{OUT}/03_model_comparison.html")
    fig.write_image(f"{OUT}/03_model_comparison.png", width=1100, height=460, scale=2)
    print("✓ 03_model_comparison")
    return fig


# ── 4. 혼동 행렬 (v4, 50케이스) ─────────────────────────────────────────────

def fig_confusion_matrix():
    labels = ["safe", "S1", "S2", "S3", "S4", "S5"]

    # v4 혼동 행렬 (실제×예측)
    # safe: 6건 전부 safe
    # S1: 5건 → S1=4, S4=1(EDGE-003)
    # S2: 6건 전부 S2
    # S3: 8건 전부 S3
    # S4: 8건 → S4=6, S1=2 (S4-001, S4-007)
    # S5: 5건 전부 S5
    z = [
        [6, 0, 0, 0, 0, 0],   # safe
        [0, 4, 0, 0, 1, 0],   # S1
        [0, 0, 6, 0, 0, 0],   # S2
        [0, 0, 0, 8, 0, 0],   # S3
        [0, 2, 0, 0, 6, 0],   # S4
        [0, 0, 0, 0, 0, 5],   # S5
    ]

    text = [[str(v) if v > 0 else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"예측: {l}" for l in labels],
        y=[f"실제: {l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        colorscale=[
            [0.0,  "#f5f6f8"],
            [0.01, "#dbeafe"],
            [0.3,  COLORS["primary"]],
            [1.0,  "#1e3a8a"],
        ],
        showscale=True,
        colorbar=dict(title="건수"),
        hovertemplate="실제: %{y}<br>예측: %{x}<br>건수: %{z}<extra></extra>",
    ))

    # 대각선 강조 (테두리로 표현)
    for i in range(len(labels)):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color=COLORS["safe"], width=2.5),
        )

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>혼동 행렬 — v4 평가 결과 (50케이스)</b><br>"
                        "<sup>정확도 94.0% | F1 Macro 0.940 | FPR 0.0%</sup>",
                   font=dict(size=17), x=0.5),
        xaxis=dict(side="bottom", tickfont=dict(size=12)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        height=500,
        width=600,
    )

    fig.write_html(f"{OUT}/04_confusion_matrix_v4.html")
    fig.write_image(f"{OUT}/04_confusion_matrix_v4.png", width=640, height=500, scale=2)
    print("✓ 04_confusion_matrix_v4")
    return fig


# ── 5. 레이더 차트 (모델 종합 비교) ─────────────────────────────────────────

def fig_radar_comparison():
    dims = ["F1 Macro", "FPR 낮음<br>(1-FPR)", "FNR 낮음<br>(1-FNR)",
            "한국어 적합성", "추론 속도<br>(정규화)"]

    models_data = {
        "Llama Guard 3 (영어)": [0.61, 0.888, 0.70, 0.30, 0.90],
        "Kanana-2.1B SFT":     [0.81, 0.938, 0.85, 0.90, 1.00],
        "Kanana-8B QLoRA SFT": [0.87, 0.917, 0.88, 0.95, 0.75],
        "프로토타입 v4":        [0.94, 1.000, 1.00, 0.92, 0.40],
    }
    fill_colors = [
        "rgba(139,149,161,0.12)",
        "rgba(99,102,241,0.12)",
        "rgba(49,130,246,0.12)",
        "rgba(245,158,11,0.12)",
    ]
    plot_colors = [COLORS["grey"], COLORS["secondary"], COLORS["primary"], COLORS["warn"]]

    fig = go.Figure()
    for (name, vals), color, fill in zip(models_data.items(), plot_colors, fill_colors):
        closed = vals + [vals[0]]
        closed_dims = dims + [dims[0]]
        fig.add_trace(go.Scatterpolar(
            r=closed, theta=closed_dims,
            fill="toself", name=name,
            line=dict(color=color, width=2.5),
            fillcolor=fill,
            opacity=0.9,
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>모델 종합 성능 레이더 차트</b>",
                   font=dict(size=18), x=0.5),
        polar=dict(
            radialaxis=dict(range=[0, 1.05], tickformat=".1f",
                            gridcolor="#e5e8eb", linecolor="#e5e8eb"),
            angularaxis=dict(gridcolor="#e5e8eb", linecolor="#e5e8eb"),
            bgcolor=COLORS["surface"],
        ),
        legend=dict(x=1.05, y=1.0, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=COLORS["grey"], borderwidth=1),
        height=500,
        width=680,
    )

    fig.write_html(f"{OUT}/05_radar_comparison.html")
    fig.write_image(f"{OUT}/05_radar_comparison.png", width=700, height=500, scale=2)
    print("✓ 05_radar_comparison")
    return fig


# ── 6. 통합 대시보드 HTML ────────────────────────────────────────────────────

def build_dashboard():
    with open(f"{OUT}/01_performance_progression.html") as f: html1 = f.read()
    with open(f"{OUT}/02_category_f1_v4.html")          as f: html2 = f.read()
    with open(f"{OUT}/03_model_comparison.html")         as f: html3 = f.read()
    with open(f"{OUT}/04_confusion_matrix_v4.html")      as f: html4 = f.read()
    with open(f"{OUT}/05_radar_comparison.html")         as f: html5 = f.read()

    def extract_body(html):
        import re
        m = re.search(r"<body>(.*?)</body>", html, re.DOTALL)
        return m.group(1) if m else html

    dashboard = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Guardrail4Agent — 연구 결과 대시보드</title>
<style>
  body {{ font-family: -apple-system, 'Apple SD Gothic Neo', sans-serif; background: #f5f6f8; margin: 0; padding: 20px; color: #191f28; }}
  .header {{ text-align: center; padding: 32px 20px 24px; }}
  .header h1 {{ font-size: 28px; font-weight: 800; margin: 0 0 8px; color: #191f28; }}
  .header p  {{ font-size: 15px; color: #4e5968; margin: 0; }}
  .badge {{ display: inline-block; background: #ebf3fe; color: #3182f6; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 20px; margin: 4px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; max-width: 1200px; margin: 0 auto; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; max-width: 1200px; margin: 0 auto; }}
  .card {{ background: #fff; border-radius: 14px; box-shadow: 0 2px 12px rgba(0,23,51,.07); overflow: hidden; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; max-width: 1200px; margin: 0 auto 24px; }}
  .stat-card {{ background: #fff; border-radius: 12px; padding: 20px 24px; text-align: center; box-shadow: 0 1px 6px rgba(0,23,51,.06); }}
  .stat-value {{ font-size: 32px; font-weight: 800; color: #3182f6; }}
  .stat-label {{ font-size: 13px; color: #4e5968; margin-top: 4px; }}
  .footer {{ text-align: center; padding: 24px; font-size: 13px; color: #8b95a1; }}
  .footer a {{ color: #3182f6; text-decoration: none; }}
</style>
</head>
<body>
<div class="header">
  <h1>Guardrail4Agent</h1>
  <p>LLM 에이전트 Tool Call 데이터 유출 탐지 — 한국어 특화 가드레일 모델 연구 결과</p>
  <div style="margin-top:12px">
    <span class="badge">F1 Macro 0.940</span>
    <span class="badge">FPR 0.0%</span>
    <span class="badge">7,000건 합성 데이터</span>
    <span class="badge">6개 탐지 카테고리</span>
    <span class="badge">Kanana-2.1B LoRA SFT</span>
  </div>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="stat-value">94.0%</div>
    <div class="stat-label">최고 정확도 (v4)</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">0.940</div>
    <div class="stat-label">F1 Macro (v4)</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:#0dcaa8">0.0%</div>
    <div class="stat-label">FPR (오탐률)</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:#6366f1">7,000</div>
    <div class="stat-label">학습 데이터 (건)</div>
  </div>
</div>

<div class="grid" style="margin-bottom:24px">
  <div class="card">{extract_body(html1)}</div>
</div>
<div class="grid" style="margin-bottom:24px">
  <div class="card">{extract_body(html2)}</div>
</div>
<div class="grid" style="margin-bottom:24px">
  <div class="card">{extract_body(html3)}</div>
</div>
<div class="grid-2" style="margin-bottom:24px">
  <div class="card">{extract_body(html4)}</div>
  <div class="card">{extract_body(html5)}</div>
</div>

<div class="footer">
  <a href="https://github.com/tristan-kkim/guardrail4agent" target="_blank">GitHub</a> ·
  <a href="https://huggingface.co/tristan-kim/kanana-guardrail4agent" target="_blank">HuggingFace Model</a> ·
  <a href="http://43-203-223-40.nip.io" target="_blank">Live Demo</a>
  <br><br>Tristan Kim, Cortexys Corp. · 2026
</div>
</body>
</html>"""

    with open(f"{OUT}/dashboard.html", "w") as f:
        f.write(dashboard)
    print("✓ dashboard.html")


if __name__ == "__main__":
    print("Guardrail4Agent 시각화 생성 중...\n")
    fig_performance_progression()
    fig_category_f1()
    fig_model_comparison()
    fig_confusion_matrix()
    fig_radar_comparison()
    build_dashboard()
    print(f"\n완료 → {OUT}")
