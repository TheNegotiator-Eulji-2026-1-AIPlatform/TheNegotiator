from fastapi import FastAPI
import joblib
import os

app = FastAPI()

# =====================================
# 모델 로드
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "pkls", "response_quality_model.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "pkls", "response_tfidf.pkl")
)


# =====================================
# API
# =====================================

@app.post("/predict")
async def evaluate_response(user_input: dict):

    # ---------------------------------
    # 언리얼에서 받은 값
    # ---------------------------------

    ai_text = user_input.get("ai_text", "")
    player_input = user_input.get("player_input", "")

    # ---------------------------------
    # 모델 입력 형식
    # ---------------------------------

    combined_text = f"{ai_text} [SEP] {player_input}"

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

    if prediction == "good":

        stability_delta = 20
        anger_delta = -15

        status = "success"

    else:

        stability_delta = -10
        anger_delta = 15

        status = "fail"

    # ---------------------------------
    # 디버그 출력
    # ---------------------------------

    print("\n========================")
    print("AI TEXT:", ai_text)
    print("PLAYER INPUT:", player_input)

    print("Prediction:", prediction)

    print("GOOD SCORE:", good_score)
    print("BAD SCORE:", bad_score)

    print("stability_delta:", stability_delta)
    print("anger_delta:", anger_delta)
    print("========================")

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