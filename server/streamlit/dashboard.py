import streamlit as st
import requests
import pandas as pd

# =====================================
# 1. 페이지 테마 및 레이아웃 설정
# =====================================
st.set_page_config(
    page_title="The Negotiator | 정밀 감정 분석 대시보드",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (KPI 카드 글씨 밝기 보정 및 XAI 박스 스타일)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    /* 변동 표시 카드 내부의 글씨체들을 강제로 밝은 흰색/하늘색 계열로 덮어쓰기 */
    .stMetric { 
        background-color: #1a1c24 !important; 
        padding: 20px !important; 
        border-radius: 12px !important; 
        border-left: 5px solid #00f2ff !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
    }
    .stMetric label { color: #e0e4ed !important; font-size: 16px !important; font-weight: 600 !important; }
    .stMetric div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 32px !important; font-weight: 700 !important; }
    
    .xai-box { background-color: #161922; padding: 25px; border-radius: 12px; border: 1px solid #2d313f; line-height: 2.2; font-size: 19px; }
    .th-container { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. 사이드바 (정보 및 고정 가이드)
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.title("Settings")
    st.info("💡 **정석 XAI (6Class)** 엔진이 활성화되었습니다. AI가 직접 계산한 토큰별 감정 가중치를 기반으로 시각화합니다.")
    st.markdown("---")
    st.subheader("Model Status")
    st.write("🟢 TF-IDF v6 (Realtime XAI)")
    st.caption("Last Update: v2.1 Alpha")

# 3. 메인 대시보드 대문 구역
st.title("🎮 The Negotiator")
st.markdown("### 차세대 감정 지능 분석 시스템")
st.caption("외부 FastAPI 서버의 TF-IDF + 로지스틱 회귀 다중 클래스(6Class) 엔진과 실시간 통신하여 언리얼 연동 데이터를 시각화합니다.")
st.markdown("---")

FASTAPI_URL = "https://rudwns67-emotion-api.hf.space/predict"

# 4. 사용자 대사 입력 구역
player_input = st.text_input("💬 플레이어 대사 입력", placeholder="협상 대사를 입력하고 엔터를 누르세요...")

if player_input:
    payload = {
        "player_input": player_input,
        "ai_text": "dashboard",
        "emotion_state": "analysis"
    }
    
    with st.spinner("🧠 백엔드 AI 엔진에서 진짜 감정 가중치를 계산하고 있습니다..."):
        try:
            response = requests.post(FASTAPI_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # 데이터 파싱
                pred = data.get("prediction", "N/A").upper()
                stab_delta = data.get("stability_delta", 0)
                ang_delta = data.get("anger_delta", 0)
                good_score = data.get("good", 0.0)
                
                # --- [추가기능] Threshold 판정 시각화 바 구성 ---
                threshold_value = 0.5  # 서버와 싱크 맞춘 임계값 기준선
                
                st.markdown('<div class="th-container">', unsafe_allow_html=True)
                th_col1, th_col2 = st.columns([3, 7])
                with th_col1:
                    st.markdown(f"### 🎯 판정 기준선 (Threshold)")
                    st.markdown(f"**현재 문장의 우호도(Good):** `{good_score*100:.1f}%`  \n**합격 기준선:** `{threshold_value*100:.0f}%` 이상 필요")
                with th_col2:
                    st.write("")  # 패딩용
                    # 진행 바 형태로 현재 우호도 스코어 표시
                    st.progress(min(max(float(good_score), 0.0), 1.0), text=f"Good Score: {good_score*100:.1f}%")
                    if good_score >= threshold_value:
                        st.markdown(f"🟢 **판정 결과:** 기준치({threshold_value*100:.0f}%)를 넘었으므로 **GOOD** ")
                    else:
                        st.markdown(f"🔴 **판정 결과:** 우호도가 기준치({threshold_value*100:.0f}%) 미만이므로 **BAD**")
                st.markdown('</div>', unsafe_allow_html=True)

                # --- [시각화 강화 섹션 - 변동 표시 레이아웃] ---
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    if pred == "GOOD":
                        st.success(f"### 최종 판정: {pred}")
                    else:
                        st.error(f"### 최종 판정: {pred}")
                with kpi2:
                    # CSS 덮어쓰기로 이제 아주 밝고 선명하게 출력됩니다!
                    st.metric(label="협상 안정도 변동 (Stability Delta)", value=f"{stab_delta}%", delta=stab_delta)
                with kpi3:
                    st.metric(label="용의자 분노 변동 (Anger Delta)", value=f"{ang_delta}%", delta=ang_delta, delta_color="inverse")

                st.markdown("---")
                
                # 메인 주 구역 레이아웃 분할
                col_left, col_right = st.columns([5, 4])
                
                # --- 5. Explainable AI (XAI) 구역 (왼쪽 열) ---
                with col_left:
                    st.subheader("🧐 Explainable AI: AI 판단 근거 정밀 분석")
                    st.markdown("##### 모델이 해당 감정 텐션에 집중한 단어별 실제 가중치 수치:")
                    
                    xai_data = data.get("xai_data", [])
                    
                    if not xai_data:
                        st.info("※ 해당 문장에는 6대 감정에 직접 기여한 핵심 단어가 검출되지 않았습니다.")
                    else:
                        html_elements = []
                        for word, score in xai_data:
                            if score > 0.01:
                                alpha = min(score * 1.5, 0.8)
                                html_elements.append(f'<span style="background-color: rgba(0, 255, 0, {alpha}); padding: 4px 8px; border-radius: 5px; margin-right: 5px; border: 1px solid rgba(0, 255, 0, 0.5); font-weight: bold; color: white;">{word}</span>')
                            elif score < -0.01:
                                alpha = min(abs(score) * 1.5, 0.8)
                                html_elements.append(f'<span style="background-color: rgba(255, 0, 0, {alpha}); padding: 4px 8px; border-radius: 5px; margin-right: 5px; border: 1px solid rgba(255, 0, 0, 0.5); font-weight: bold; color: white;">{word}</span>')
                            else:
                                html_elements.append(f'<span>{word}</span>')
                        
                        rendered_html = " ".join(html_elements)
                        st.markdown(f'<div class="xai-box">{rendered_html}</div>', unsafe_allow_html=True)
                        st.caption("※ 색상이 진할수록 AI 모델이 해당 감정 판정에 결정적 영향을 준 단어 조각입니다.")
                    
                    st.markdown("---")
                    st.subheader("💡 협상 가이드")
                    if pred == "GOOD":
                        st.success("✅ **협상 우위 확보**\n용의자의 방어 기제가 약해졌습니다. 구체적인 조건을 제시하거나 공감적 태도를 취하세요.")
                    else:
                        st.error("⚠️ **적대감 상승**\n용의자가 위협을 느끼거나 공격적으로 변했습니다. 화제를 돌리거나 침착하게 대화를 유도하세요.")

                # --- 6. 결과 시각화 강화 구역 (오른쪽 열) ---
                with col_right:
                    st.subheader("📊 세부 감정 확률 분포")
                    
                    emotions_dict = {
                        "Good (종합)": data.get("good", 0.0),
                        "Bad (종합)": data.get("bad", 0.0),
                        "분노 (Anger)": data.get("anger", 0.0),
                        "불안 (Anxiety)": data.get("anxiety", 0.0),
                        "슬픔 (Sadness)": data.get("sadness", 0.0),
                        "상처 (Hurt)": data.get("hurt", 0.0),
                        "당황 (Embarrassment)": data.get("embarrassment", 0.0)
                    }
                    
                    df = pd.DataFrame(list(emotions_dict.items()), columns=["감정 유형", "확률 스코어"])
                    st.bar_chart(df.set_index("감정 유형"))
                    st.caption("TF-IDF와 로지스틱 회귀 기반으로 연산된 확률 분포도입니다.")

                # 7. 하단 디버깅용 원본 데이터 아코디언
                with st.expander("🔍 API 서버 응답 원본 JSON 데이터 (설명 가능성 검증용)"):
                    st.json(data)

            else:
                st.error(f"❌ 외부 FastAPI 서버 연결 실패 (에러 코드: {response.status_code})")
        except Exception as e:
            st.error(f"📡 백엔드 서버와 실시간 통신 불가: {e}")