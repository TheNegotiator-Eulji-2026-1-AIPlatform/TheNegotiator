import pandas as pd

# =========================
# 감정별 효과 매핑
# =========================

EMOTION_EFFECT_MAP = {
    "분노": {
        "stability_delta": 20,
        "anger_delta": -15,
        "trust_delta": 10
    },

    "불안": {
        "stability_delta": 18,
        "anger_delta": -5,
        "trust_delta": 12
    },

    "슬픔": {
        "stability_delta": 15,
        "anger_delta": -3,
        "trust_delta": 15
    },

    "상처": {
        "stability_delta": 17,
        "anger_delta": -7,
        "trust_delta": 14
    },

    "당황": {
        "stability_delta": 10,
        "anger_delta": -2,
        "trust_delta": 8
    },

    "기쁨": {
        "stability_delta": 5,
        "anger_delta": -1,
        "trust_delta": 5
    }
}


# =========================
# 전처리 함수
# =========================

def preprocess_dataset(input_path, output_path):

    df = pd.read_csv(input_path, encoding="cp949")

    # 필요한 컬럼만 추출
    df = df[[
        "감정_대분류",
        "사람문장1",
        "시스템문장1"
    ]]

    # 컬럼명 변경
    df = df.rename(columns={
        "감정_대분류": "emotion_state",
        "사람문장1": "ai_text",
        "시스템문장1": "good_response"
    })

    # 결측 제거
    df = df.dropna()

    # 문자열 정리
    df["ai_text"] = df["ai_text"].str.strip()
    df["good_response"] = df["good_response"].str.strip()

    # effect 값 추가
    df["stability_delta"] = df["emotion_state"].apply(
        lambda x: EMOTION_EFFECT_MAP[x]["stability_delta"]
    )

    df["anger_delta"] = df["emotion_state"].apply(
        lambda x: EMOTION_EFFECT_MAP[x]["anger_delta"]
    )

    df["trust_delta"] = df["emotion_state"].apply(
        lambda x: EMOTION_EFFECT_MAP[x]["trust_delta"]
    )

    # 빈 문자열 제거
    df = df[
        (df["ai_text"] != "") &
        (df["good_response"] != "")
    ]

    # 저장
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {output_path}")
    print(df.head())


# =========================
# 실행
# =========================

preprocess_dataset(
    input_path="raw/train.csv",
    output_path="processed/train_processed.csv"
)

preprocess_dataset(
    input_path="raw/valid.csv",
    output_path="processed/valid_processed.csv"
)