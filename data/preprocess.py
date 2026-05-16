import os
import random
import pandas as pd

# =====================================
# 경로 설정
# =====================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_PATH = os.path.join(CURRENT_DIR, "..", "data", "raw")
PROCESSED_PATH = os.path.join(CURRENT_DIR, "..", "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

# =====================================
# 원본 데이터 로드
# =====================================

train_raw = pd.read_csv(
    os.path.join(RAW_PATH, "train.csv"),
    encoding="cp949"
)

test_raw = pd.read_csv(
    os.path.join(RAW_PATH, "valid.csv"),
    encoding="cp949"
)

# =====================================
# 부정 문장 템플릿
# =====================================

bad_templates = [

    # 비난형
    "그건 네 잘못이잖아",
    "그러게 잘 좀 하지",
    "왜 그렇게 행동했어?",
    "네가 문제네",
    "당연한 결과 아니야?",
    "스스로 자초한 일이네",
    "그건 변명 같은데",

    # 무시형
    "어쩌라고",
    "그래서 뭐?",
    "별것도 아닌데",
    "다들 그렇게 살아",
    "그 정도 가지고?",
    "유난이네",
    "너무 예민한 거 아냐?",
    "신경 끄면 되잖아",

    # 냉소형
    "참 힘들겠다~",
    "그래서 세상이 바뀌냐?",
    "울면 해결돼?",
    "누가 들으면 큰일 난 줄",
    "그게 그렇게 심각해?",
    "다 핑계 아니야?",

    # 회피형
    "몰라",
    "관심 없어",
    "귀찮게 하지 마",
    "알아서 해",
    "난 모르겠는데",
    "내가 왜 신경 써야 해?",

    # 맥락 무시
    "오늘 점심 뭐 먹었어?",
    "게임 재밌더라",
    "날씨 좋네",
    "축구 좋아해?",
    "배고프다",
    "졸리다",

    # 의미 불명
    "ㅁㄴㅇㄹ",
    "asdf",
    "ㅋㅋㅋㅋㅋㅋ",
    "ㅎㅇ",
    "ㅂㅂ",
    "마ㅣㅓㄴ아ㅣ러",
    ".......",
    "???"
]

# =====================================
# 감정 매핑
# =====================================

emotion_mapping = {

    "분노": "anger",
    "불안": "anxiety",
    "슬픔": "sadness",
    "상처": "hurt",
    "당황": "embarrassment"
}

# =====================================
# 데이터 생성 함수
# =====================================

def build_dataset(raw_df):

    # ---------------------------------
    # 필요한 컬럼만 사용
    # ---------------------------------

    raw_df = raw_df[[
        "감정_대분류",
        "사람문장1",
        "시스템문장1"
    ]]

    # ---------------------------------
    # 기쁨 제거
    # ---------------------------------

    raw_df = raw_df[
        raw_df["감정_대분류"] != "기쁨"
    ]

    # ---------------------------------
    # 결측치 제거
    # ---------------------------------

    raw_df = raw_df.dropna()

    dataset = []

    # =================================
    # GOOD 데이터 생성
    # =================================

    for _, row in raw_df.iterrows():

        emotion_kr = str(
            row["감정_대분류"]
        ).strip()

        # 매핑 안 된 감정 스킵
        if emotion_kr not in emotion_mapping:
            continue

        emotion_label = emotion_mapping[
            emotion_kr
        ]

        ai_text = str(
            row["사람문장1"]
        ).strip()

        player_input = str(
            row["시스템문장1"]
        ).strip()

        dataset.append({

            "ai_text": ai_text,

            "player_input": player_input,

            "label": emotion_label
        })

    # =================================
    # BAD 데이터 생성
    # =================================

    for _, row in raw_df.iterrows():

        ai_text = str(
            row["사람문장1"]
        ).strip()

        # 상황당 랜덤 부정문 1개
        sampled_bad = random.sample(
            bad_templates,
            1
        )

        for bad_text in sampled_bad:

            dataset.append({

                "ai_text": ai_text,

                "player_input": bad_text,

                "label": "bad"
            })

    # =================================
    # DataFrame 생성
    # =================================

    final_df = pd.DataFrame(dataset)

    # 셔플
    final_df = final_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return final_df

# =====================================
# Train/Test 생성
# =====================================

train_processed = build_dataset(
    train_raw
)

test_processed = build_dataset(
    test_raw
)

# =====================================
# 저장
# =====================================

train_save_path = os.path.join(
    PROCESSED_PATH,
    "train_processed.csv"
)

test_save_path = os.path.join(
    PROCESSED_PATH,
    "valid_processed.csv"
)

train_processed.to_csv(
    train_save_path,
    index=False,
    encoding="utf-8-sig"
)

test_processed.to_csv(
    test_save_path,
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# 출력
# =====================================

print("=====================================")
print("데이터 전처리 완료")
print("=====================================")

print(f"\nTrain 데이터 수: {len(train_processed)}")
print(f"Test 데이터 수: {len(test_processed)}")

print("\nTrain 라벨 분포")
print(
    train_processed["label"].value_counts()
)

print("\nTest 라벨 분포")
print(
    test_processed["label"].value_counts()
)

print(f"\nTrain 저장 위치:")
print(train_save_path)

print(f"\nTest 저장 위치:")
print(test_save_path)

