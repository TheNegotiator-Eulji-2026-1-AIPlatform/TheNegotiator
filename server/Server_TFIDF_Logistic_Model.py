from fastapi import FastAPI
import joblib
import os

app = FastAPI()

# =====================================
# 모델 로드
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "tfidf_2class_model.pkl")
)
print(model)
# =====================================
# API
# =====================================

@app.post("/predict")
async def evaluate_response(user_input: dict):

    ai_text = user_input.get("ai_text", "")
    player_input = user_input.get("player_input", "")
    emotion_state = user_input.get("emotion_state", "")

    # combined_text = f"{ai_text} [SEP] {player_input}"

    # player_input만 예측
    # combined_text = player_input

    # Pipeline 내부에서 자동 TF-IDF 처리
    prediction = model.predict([player_input])[0]

    probabilities = model.predict_proba([player_input])[0]

    class_map = {
        model.classes_[i]: probabilities[i]
        for i in range(len(model.classes_))
    }

    labels = model.classes_ 
    score_dict = dict(zip(labels, probabilities)) 
    good_score = score_dict["good"] 
    # bad_score = score_dict["bad"]

    # threshold 적용 
    if good_score >= 0.7: 
        prediction = "good"
    else: 
        prediction = "bad"

    # 게이지 계산
    stability_delta = 20
    anger_delta = -15

    if prediction == "bad":

        stability_delta = -10
        anger_delta = 15

    elif prediction == "good":
        # 3턴 만에 결판을 내기 위해 한 방의 임팩트(35%)를 키웁니다!
        stability_delta = 35  
        anger_delta = -25
    
    print("========================")
    print("AI TEXT:", ai_text) 
    print("PLAYER INPUT:", player_input) 
    print("Emotion State:",emotion_state)
    print("Prediction:", prediction) 

    print("stability_delta:", stability_delta) 
    print("anger_delta:", anger_delta) 
    print("========================")


    return {

        "prediction": prediction,

        "anger": float(class_map.get("anger", 0.0)),
        "anxiety": float(class_map.get("anxiety", 0.0)),
        "sadness": float(class_map.get("sadness", 0.0)),
        "hurt": float(class_map.get("hurt", 0.0)),
        "embarrassment": float(class_map.get("embarrassment", 0.0)),
        "bad": float(class_map.get("bad", 0.0)),

        "stability_delta": stability_delta,
        "anger_delta": anger_delta
    }

