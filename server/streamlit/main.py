from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List

app = FastAPI()

# 모델 로드
classifier = pipeline("text-classification", model="jaehyeong/koelectra-base-v3-generalized-sentiment-analysis")

# 현재 상태 저장용 (대시보드 호환용)
game_state = {"total_score": 0, "current_turn": 0, "history": []}

class UserInput(BaseModel):
    user_id: str
    text: str

class Highlight(BaseModel):
    word: str
    score: float

class AnalysisResponse(BaseModel):
    emotion: str
    confidence: float
    highlights: List[Highlight]

@app.post("/predict", response_model=AnalysisResponse)
async def predict(data: UserInput):
    result = classifier(data.text)[0]
    
    # 1. 모델이 실제로 내뱉는 label 확인 (결과가 '0', '1' 인지 'LABEL_0' 인지 확인)
    raw_label = str(result['label']).upper() # 대문자로 통일해서 비교

    # 2. 모델(koelectra-base-v3-generalized) 기준 매핑 설정
    # 보통 이 모델은 0이 부정, 1이 긍정입니다. 
    # 모델에 따라 '0', '1' 혹은 'LABEL_0', 'LABEL_1'로 나옵니다.
    if "0" in raw_label: # '0'이나 'LABEL_0'이 포함되어 있다면
        emotion_korean = "부정"
        score = -2
    elif "1" in raw_label: # '1'이나 'LABEL_1'이 포함되어 있다면
        emotion_korean = "긍정"
        score = 2
    else:
        emotion_korean = "중립"
        score = 0
    
    # 게임 상태 업데이트
    game_state["total_score"] += score
    game_state["current_turn"] += 1
    game_state["history"].append({
        "turn": game_state["current_turn"], 
        "text": data.text, 
        "emotion": emotion_korean, 
        "score": score
    })

    response = {
        "emotion": emotion_korean,
        "confidence": round(result['score'], 2),
        "highlights": [{"word": "샘플", "score": 0.5}] 
    }
    return response

# 대시보드가 에러 안 나게 하려면 이 엔드포인트가 필요합니다!
@app.get("/status")
async def get_status():
    return game_state