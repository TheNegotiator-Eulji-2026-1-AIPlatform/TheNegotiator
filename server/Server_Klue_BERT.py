import os
import torch
from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

app = FastAPI()

# =====================================
# 1. 모델 및 토크나이저 로드
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bert_model.pt")
# 팀원이 사용한 BERT 모델명 (예: 'klue/bert-base')에 맞춰야 합니다 .
TOKENIZER_NAME = "klue/bert-base" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 구조 선언 및 가중치(.pt) 로드
# num_labels는 팀원이 설정한 라벨 수(good, bad 2개면 2)에 맞춰야 합니다 .
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

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
# 2. 플레이어 응답 평가 API (BERT 버전)
# =====================================
@app.post("/predict")
async def evaluate_response_bert(user_input: dict):
    emotion_state = user_input.get("emotion_state", "분노")
    ai_text = user_input.get("ai_text", "")
    player_input = user_input.get("player_input", "")


    # BERT 입력 형식 조립
    combined_text = player_input

    # 토큰화 및 모델 입력 생성
    inputs = tokenizer(
        combined_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding=True
    ).to(device)

    # 예측 (Gradient 계산 비활성화)
    with torch.no_grad():
        outputs = model(**inputs)
        # 6개 감정에 대한 Softmax 확률 계산
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
    # 팀원에게 받은 라벨 순서에 맞춰 리스트를 만드세요
    label_names = ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]
    
    # 전체 결과값 조립 (각 감정별 % 계산)
    full_results = {}
    for i in range(len(label_names)):
        full_results[label_names[i]] = float(probs[i]) * 100

    # 가장 높은 확률을 가진 감정 찾기
    max_idx = torch.argmax(probs).item()
    predicted_emotion = label_names[max_idx]
    confidence = full_results[predicted_emotion]

    # ---------------------------------
    # 디버그 출력 (전체 결과값 확인용)
    # ---------------------------------
    print("\n" + "="*40)
    print(f"입력 문장: {combined_text}")
    print("-" * 40)
    print("[ 전체 감정 분석 결과 ]")
    for name, value in full_results.items():
        print(f" - {name}: {value:.2f}%")
    print("-" * 40)
    print(f"최종 예측: {predicted_emotion} ({confidence:.2f}%)")
    print("="*40 + "\n")


    # 2. 언리얼에서 보낸 현재 상황 태그
    current_context = emotion_state 

    # 3. 판정 : 상황이 일치하고 + 그 확신도가 60% 를 넘어야 함
    if current_context == predicted_emotion and confidence >= 60.0:
        # 대성공 ! 상황도 맞고 모델도 확신함
        status = "success"
        stability_delta = 20
        anger_delta = -15
    else:
        # 상황이 다르거나 , 상황은 맞아도 확신도가 60% 미만이면 실패 !
        status = "fail"
        stability_delta = -15
        anger_delta = 20

    # ---------------------------------
    # 디버그 출력 ( 60% 통과 여부 확인 )
    # ---------------------------------
    print(f"현재 상황: {current_context}")
    print(f"모델 판단: {predicted_emotion} ({confidence:.2f}%)")
    
    if current_context == predicted_emotion:
        if confidence >= 60.0:
            print("결과: [ SUCCESS ] 60% 돌파 ! 적절한 위로입니다 .")
        else:
            print("결과: [ FAIL ] 상황은 맞지만 확신도가 부족합니다 ( 60% 미만 ) .")
    else:
        print("결과: [ FAIL ] 상황에 맞지 않는 답변입니다 .")

    return {
        "status": status,
        # "good_score": good_score,
        # "bad_score": bad_score,
        "stability_delta": stability_delta,
        "anger_delta": anger_delta
    }
    