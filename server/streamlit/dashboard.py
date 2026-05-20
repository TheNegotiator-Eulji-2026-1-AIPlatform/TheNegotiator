import streamlit as st
import requests

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="The Negotiator - 정밀 감정 분석기", layout="wide")

st.title("👨‍⚖️ The Negotiator: 정밀 심리 분석 시스템")
st.markdown("---")

# FastAPI 서버 주소 (기존 백엔드 스페이스 주소 확인!)
FASTAPI_URL = "https://rudwns67-emotion-api.hf.space/predict"

# 2. 레이아웃 분할 (입력창)
player_input = st.text_input("📝 플레이어 대사 입력", placeholder="용의자에게 건넬 말을 입력하세요...")

if player_input:
    # FastAPI 규격에 맞춘 데이터 전송
    payload = {
        "player_input": player_input,
        "ai_text": "대시보드 테스트",
        "emotion_state": "analysis"
    }

    with st.spinner("🚀 BERT 모델 분석 중..."):
        try:
            response = requests.post(FASTAPI_URL, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()

                # --- 데이터 추출 ---
                pred = data.get("prediction", "N/A")
                scores = {
                    "Good (종합)": data.get("good", 0.0),
                    "Bad (종합)": data.get("bad", 0.0),
                    "분노 (Anger)": data.get("anger", 0.0),
                    "불안 (Anxiety)": data.get("anxiety", 0.0),
                    "슬픔 (Sadness)": data.get("sadness", 0.0),
                    "상처 (Hurt)": data.get("hurt", 0.0),
                    "당황 (Embarrassment)": data.get("embarrassment", 0.0)
                }
                stab_delta = data.get("stability_delta", 0)
                ang_delta = data.get("anger_delta", 0)

                # 3. 상단 결과 요약 (Prediction & Deltas)
                st.subheader("🎯 분석 결과 요약")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if pred.lower() == "good":
                        st.success(f"### 판정: {pred.upper()}")
                    else:
                        st.error(f"### 판정: {pred.upper()}")
                
                with col2:
                    st.metric(label="협상 안정도 변동 (Stability)", value=f"{stab_delta}%", delta=stab_delta)
                
                with col3:
                    st.metric(label="용의자 분노 변동 (Anger)", value=f"{ang_delta}%", delta=ang_delta, delta_color="inverse")

                st.markdown("---")

                # 4. 세부 감정 수치 시각화 (그래프 구역)
                st.subheader("📊 세부 감정 확률 데이터")
                
                # 왼쪽: 핵심 지표 (Good / Bad)
                # 오른쪽: 5대 세부 감정
                left_col, right_col = st.columns(2)

                with left_col:
                    st.write("**[핵심 판정 지표]**")
                    for label in ["Good (종합)", "Bad (종합)"]:
                        val = scores[label]
                        st.write(f"{label}: {val*100:.1f}%")
                        st.progress(min(max(val, 0.0), 1.0))

                with right_col:
                    st.write("**[5대 심리 세부 스코어]**")
                    for label in ["분노 (Anger)", "불안 (Anxiety)", "슬픔 (Sadness)", "상처 (Hurt)", "당황 (Embarrassment)"]:
                        val = scores[label]
                        st.write(f"{label}: {val*100:.1f}%")
                        st.progress(min(max(val, 0.0), 1.0))

                # 5. 하단 JSON 원본 데이터 확인 (디버깅용)
                with st.expander("🔍 API 서버 응답 원본 데이터 보기"):
                    st.json(data)

            else:
                st.error(f"❌ 서버 응답 에러: {response.status_code}")
        except Exception as e:
            st.error(f"📡 연결 실패: {e}")