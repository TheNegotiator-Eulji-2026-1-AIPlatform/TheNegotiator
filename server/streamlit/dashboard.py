import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="감정 분석 대시보드")
st.title("🤝 협상 게임 실시간 모니터링")

placeholder = st.empty()

while True:
    try:
        r = requests.get("https://rudwns67-emotion-api.hf.space").json()
        with placeholder.container():
            st.metric("현재 총 감정 점수", f"{r['total_score']} 점")
            st.write(f"현재 진행 턴: {r['current_turn']} / 5")
            
            if r['history']:
                df = pd.DataFrame(r['history'])
                # 'score' 컬럼을 기반으로 그래프 그리기
                st.line_chart(df.set_index('turn')['score'])
                st.table(df)
    except Exception as e:
        st.error(f"서버 연결 대기 중... ({e})")
    time.sleep(1)