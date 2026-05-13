from fastapi import FastAPI
import joblib
import os
import pandas as pd

app = FastAPI()

# =====================================
# 경로 설정
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================
# 모델 로드
# =====================================
model = joblib.load(
    os.path.join(BASE_DIR, "models", "response_quality_model.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "models", "response_tfidf.pkl")
)

# =====================================
# 데이터셋 로드
# =====================================
dataset_path = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "processed",
    "train_processed.csv"
)

dataset = pd.read_csv(
    dataset_path,
    encoding="utf-8-sig"
)

# =====================================
# 랜덤 상황 API
# =====================================
@app.get("/scenario")
async def get_scenario():
    # 랜덤 행 선택(sample 함수 자체가 랜덤추출)
    row = dataset.sample(1).iloc[0]

    # 언리얼로 반환
    return {
        "emotion_state": row["emotion_state"],
        "ai_text": row["ai_text"],
        "good_response": row["good_response"]
    }

# =====================================
# 플레이어 응답 평가 API
# =====================================
@app.post("/predict")
async def evaluate_response(user_input: dict):

    # ---------------------------------
    # 언리얼에서 받은 값 (감정 상태 추가!)
    # ---------------------------------
    # 언리얼에서 감정을 안 보내면 기본값으로 '분노'를 쓰도록 방어 코드 추가
    emotion_state = user_input.get("emotion_state", "분노") 
    ai_text = user_input.get("ai_text", "")
    player_input = user_input.get("player_input", "")

    # ---------------------------------
    # 모델 입력 형식 (새로운 태그 조립 방식 적용)
    # ---------------------------------
    combined_text = f"[{emotion_state}] {ai_text} [SEP] {player_input}"

    # ---------------------------------
    # TF-IDF 변환
    # ---------------------------------
    text_tfidf = tfidf.transform([combined_text])

    # ---------------------------------
    # 예측
    # ---------------------------------
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    class_map = {
        model.classes_[i]: probabilities[i]
        for i in range(len(model.classes_))
    }

    good_score = class_map.get("good", 0.0)
    bad_score = class_map.get("bad", 0.0)

    # ---------------------------------
    # 게이지 계산
    # ---------------------------------
    stability_delta = 0
    anger_delta = 0

    if good_score >= 0.4:
        stability_delta = 40
        anger_delta = -20
        status = "success"

    elif bad_score >= 0.5:
        stability_delta = -20
        anger_delta = 40
        status = "fail"

    else:
        stability_delta = -10
        anger_delta = 10
        status = "neutral"

    # ---------------------------------
    # 디버그 출력
    # ---------------------------------
    print("\n========================")
    print("EMOTION STATE:", emotion_state)
    print("AI TEXT:", ai_text)
    print("PLAYER INPUT:", player_input)
    print(f"-> 조립된 문장: {combined_text}") # 디버깅용 조립 문장 확인!
    print("-" * 24)
    print("Prediction:", prediction)
    print("status:", status)
    print(f"GOOD SCORE: {good_score:.4f}")
    print(f"BAD SCORE: {bad_score:.4f}")
    print("stability_delta:", stability_delta)
    print("anger_delta:", anger_delta)
    print("========================\n")
    print(model.classes_)

    # ---------------------------------
    # 언리얼 반환
    # ---------------------------------
    return {
        "status": status,
        "prediction": prediction,
        "good_score": float(good_score),
        "bad_score": float(bad_score),
        "stability_delta": stability_delta,
        "anger_delta": anger_delta
    }