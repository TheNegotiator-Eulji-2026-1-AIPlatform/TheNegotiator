import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# =====================================
# 1. 6클래스 TF-IDF 파이프라인 파일 딱 하나만 로드!
# =====================================
pipeline = joblib.load("./tfidf_6class_model.pkl")

# 파이프라인 내부에서 벡터라이저와 로지스틱 회귀 모델 추출
tfidf_vectorizer = pipeline.named_steps['tfidf']
lr_model = pipeline.named_steps['clf']

# 모델이 학습한 6개 감정의 방 번호 순서 리스트 확보
# 예: ['anger', 'anxiety', 'bad', 'embarrassment', 'hurt', 'sadness']
class_labels = list(lr_model.classes_)

class PredictRequest(BaseModel):
    player_input: str
    ai_text: str = ""
    emotion_state: str = ""

@app.post("/predict")
async def predict(request: PredictRequest):
    text = request.player_input
    
    # -------------------------------------------------------------
    # [Step 1] 6대 감정 확률 초고속 추론 (BERT 대신 TF 모델이 연산)
    # -------------------------------------------------------------
    # 입력된 문장의 6개 클래스별 확률값(Probability)을 통째로 뽑아냅니다.
    probabilities = lr_model.predict_proba(tfidf_vectorizer.transform([text]))[0]
    
    # 프론트엔드가 요구하는 변수명 스펙에 맞게 딕셔너리로 깔끔하게 매핑
    score_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
    
    # 가장 확률이 높은 최고 존엄 감정(Primary Emotion) 찾기
    primary_emotion = class_labels[np.argmax(probabilities)]
    
    # -------------------------------------------------------------
    # [Step 2] 인게임 기획 알고리즘 연산 (Good/Bad 및 델타값)
    # -------------------------------------------------------------
    # 예시: 최고 감정이 'sadness'나 'anger' 등이면 'bad' 판정, 그 외엔 'good' 판정하는 기존 기획 적용
    # 1. 6개 감정 중 가장 확률이 높은(가장 큰 값을 가진) 감정 키값 찾기
    primary_score = score_dict.get(primary_emotion, 0.0)

    # 2. 판정 알고리즘 적용 (bad가 아니면 전부 good 처리)
    if primary_emotion == "bad":
        prediction = "bad"
        good_score = 0.0 # 굳이 채워넣자면 bad의 여사건 확률
        bad_score = primary_score
    else:
        # bad가 아닌 나머지 감정(anger, anxiety 등)이 가장 높다면? 무조건 good!
        prediction = "good"
        # [핵심 아이디어 반영] 그 중 가장 높은 확률 값을 가진 감정 스코어를 good_score에 대입!
        good_score = primary_score
        bad_score = score_dict.get("bad", 0.0)

    # 임계값(Threshold) 판정 알고리즘 적용
    if good_score >= 0.5: 
        prediction = "good"
    else: 
        prediction = "bad"

    # =====================================
    # 언리얼 연동용 게이지 델타 값 연산
    # =====================================
    stability_delta = 20
    anger_delta = -15

    if prediction == "bad":
        stability_delta = -10
        anger_delta = 30
    elif prediction == "good":
        stability_delta = 35
        anger_delta = -25

    # -------------------------------------------------------------
    # [Step 3] 진짜 6클래스 TF-IDF 단어별 기여도 가중치(XAI) 추출
    # -------------------------------------------------------------
    clean_xai_data = []
    words = text.split()
    
    # 현재 문장에서 가장 지배적인 감정 방의 가중치 행렬을 조준합니다.
    target_emotion = primary_emotion
    if target_emotion in class_labels:
        target_class_idx = class_labels.index(target_emotion)
        model_coefficients = lr_model.coef_[target_class_idx]
        
        for word in words:
            clean_word = word.strip(",.?\"'!")
            if clean_word in tfidf_vectorizer.vocabulary_:
                word_idx = tfidf_vectorizer.vocabulary_[clean_word]
                word_weight = float(model_coefficients[word_idx])
                clean_xai_data.append([word, word_weight])
            else:
                clean_xai_data.append([word, 0.0])
    else:
        clean_xai_data = [[w, 0.0] for w in words]

    # -------------------------------------------------------------
    # [Step 4] Streamlit과 언리얼 엔진이 요구하는 11개 아웃풋 완벽 반환!
    # -------------------------------------------------------------
    return {
        "prediction": prediction,
        "anger": score_dict.get("anger", 0.0),
        "anxiety": score_dict.get("anxiety", 0.0),
        "sadness": score_dict.get("sadness", 0.0),
        "hurt": score_dict.get("hurt", 0.0),
        "embarrassment": score_dict.get("embarrassment", 0.0),
        "bad": bad_score,
        "good": good_score,
        "stability_delta": stability_delta,
        "anger_delta": anger_delta,
        
        # 진짜 TF 모델이 뱉어낸 기여도 데이터 전달
        "xai_data": clean_xai_data 
    }