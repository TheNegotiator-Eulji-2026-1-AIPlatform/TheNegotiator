from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

# =====================================
# 1. 허깅페이스 모델 및 토크나이저 로드
# =====================================
model = AutoModelForSequenceClassification.from_pretrained("./")
tokenizer = AutoTokenizer.from_pretrained("./")


# 추론 전용 모드 세팅 및 CPU/GPU 자동 할당
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"✅ 모델 로드 완료! 연산 장치: {device}")


# 클래스 맵 매핑 규칙 정의
# 학습(Fine-tuning) 시키셨을 때 사용한 라벨 순서(0, 1, 2...)와 정확히 일치해야 합니다.
CLASS_LABELS = ["anger", "anxiety", "sadness", "hurt", "embarrassment", "bad"]

# =====================================
# 2. API 엔드포인트
# =====================================

@app.post("/predict")
async def evaluate_response(user_input: dict):

    ai_text = user_input.get("ai_text", "")
    player_input = user_input.get("player_input", "")
    emotion_state = user_input.get("emotion_state", "")

    # 플레이어 텍스트가 비어있을 경우 예외 처리
    if not player_input.strip():
        return {
            "prediction": "bad",
            "stability_delta": -10,
            "anger_delta": 25
        }

    # =====================================
    # 3. 딥러닝 추론 (Inference) 연산 과정
    # =====================================
    with torch.no_grad():
        inputs = tokenizer(
            player_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 소프트맥스 확률 변환
        probs = F.softmax(logits, dim=-1)
        # 차원이 1차원 배열 형태로 이쁘게 풀리도록 확실하게 가공
        probabilities = probs.detach().cpu().numpy().flatten().tolist()

    # [방어 코드] 만약 데이터가 리스트가 아니라 단일 숫자 유형으로 넘어왔을 때의 예외 처리
    if not isinstance(probabilities, list):
        probabilities = [probabilities]

    # 라벨 개수와 확률 데이터 개수 중 작은 것에 맞춰서 안전하게 조립 (IndexError 원천 차단)
    score_dict = {CLASS_LABELS[i]: probabilities[i] for i in range(min(len(CLASS_LABELS), len(probabilities)))}

    # 1. 6개 감정 중 가장 확률이 높은(가장 큰 값을 가진) 감정 키값 찾기
    primary_emotion = max(score_dict, key=score_dict.get)
    primary_score = score_dict.get(primary_emotion, 0.0)

    # 2. 판정 알고리즘 적용 (bad가 아니면 전부 good 처리)
    if primary_emotion == "bad":
        prediction = "bad"
        good_score = 1.0 - primary_score # 굳이 채워넣자면 bad의 여사건 확률
        bad_score = primary_score
    else:
        # bad가 아닌 나머지 감정(anger, anxiety 등)이 가장 높다면? 무조건 good!
        prediction = "good"
        # [핵심 아이디어 반영] 그 중 가장 높은 확률 값을 가진 감정 스코어를 good_score에 대입!
        good_score = primary_score
        bad_score = score_dict.get("bad", 0.0)

    # 만약 모델이 딱 1개의 아웃풋만 주는 세팅이라 bad_score가 누락되었다면 자동 보정
    if len(probabilities) == 1:
        # 모델의 유일한 출력이 good 확률이라면 bad는 1에서 뺀 값
        bad_score = 1.0 - good_score

    # 임계값(Threshold) 판정 알고리즘 적용
    if good_score >= 0.5: 
        prediction = "good"
    else: 
        prediction = "bad"

    # =====================================
    # 4. 언리얼 연동용 게이지 델타 값 연산 (3턴 기획 맞춤)
    # =====================================
    stability_delta = 20
    anger_delta = -15

    if prediction == "bad":
        stability_delta = -10
        anger_delta = 25
    elif prediction == "good":
        # 3턴 만에 결판을 내기 위한 경준님의 메인 한 방 이펙트 (35%)
        stability_delta = 35  
        anger_delta = -25
    
    # 디버깅용 터미널 로그 찍기
    print("========================")
    print("AI TEXT:", ai_text) 
    print("PLAYER INPUT:", player_input) 
    print("Emotion State:", emotion_state)
    print("Prediction (BERT):", prediction) 
    print(f"Probabilities -> Good: {good_score:.2f} / Bad: {bad_score:.2f}")
    print("stability_delta:", stability_delta) 
    print("anger_delta:", anger_delta) 
    print("========================")

    # 언리얼에서 리턴값 Json을 파싱할 때 에러가 나지 않도록 기존 구조의 모든 키값 유지
    return {
        "prediction": prediction,

        # 필요 시 다중 감정 모델로 확장할 때 활용할 수 있도록 키값 유지
        "anger": float(score_dict.get("anger", 0.0)),
        "anxiety": float(score_dict.get("anxiety", 0.0)),
        "sadness": float(score_dict.get("sadness", 0.0)),
        "hurt": float(score_dict.get("hurt", 0.0)),
        "embarrassment": float(score_dict.get("embarrassment", 0.0)),
        "bad": float(bad_score),
        "good": float(good_score),

        "stability_delta": int(stability_delta),
        "anger_delta": int(anger_delta)
    }