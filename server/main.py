from fastapi import FastAPI
import joblib
import os

app = FastAPI()

# 1. 서버 시작 시 모델 미리 로드 (빠른 속도를 위해!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "pkls", "emotion_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "pkls", "tfidf_vectorizer.pkl"))

@app.post("/chat")
async def predict_emotion(user_input: dict):
    text = user_input.get("text", "")
    
    # 2. 텍스트를 숫자로 변환 후 예측
    text_tfidf = tfidf.transform([text])
    probs = model.predict_proba(text_tfidf)[0] # [기쁨, 분노, 슬픔] 확률
    
    # 3. 언리얼로 보낼 점수 가공 (아까 짠 2개 축 로직!)
    classes = sorted(['기쁨', '분노', '슬픔'])
    result = {classes[i]: probs[i] for i in range(len(classes))}
    
    # 분노와 슬픔 중 더 큰 값을 부정 게이지로 사용
    negativity_score = max(result['분노'], result['슬픔'])  

    # 만약 가장 높은 점수가 0.4 미만이라면? (확신이 없는 상태)
    if result['기쁨'] < 0.4 and negativity_score < 0.4:
        print("미안 , 무슨 말인지 잘 모르겠어 . 다시 말해줄래 ?")
        print("happiness_score:", result['기쁨'])
        print("angersadness_score:", negativity_score)
        return {
            "status": "confused", 
            "happiness_score": 0.0, 
            "angersadness_score": 0.0,
        }

    print("user_input:", text)
    print("happiness_score:", result['기쁨'])
    print("angersadness_score:", negativity_score)
    
    return {
        "happiness_score": result['기쁨'],
        "angersadness_score": negativity_score
    }
    
