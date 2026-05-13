import os
import random
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. 경로 설정
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(current_dir)

data_path = os.path.join(server_dir, "..", "data", "processed")
pkl_save_path = os.path.join(server_dir, "models")

if not os.path.exists(pkl_save_path):
    os.makedirs(pkl_save_path)

# =========================
# 2. 데이터 로드
# =========================
train_df = pd.read_csv(os.path.join(data_path, "train_processed.csv"), encoding="utf-8-sig")
test_df = pd.read_csv(os.path.join(data_path, "valid_processed.csv"), encoding="utf-8-sig")

# =========================
# 3.핵심: 학습 데이터 생성 (상황 인지 + 네거티브 샘플링)
# =========================
def build_dataset(df):
    samples = []

    for idx, row in df.iterrows():
        emotion = row["emotion_state"]
        ai_text = row["ai_text"]
        good_response = row["good_response"]

        # -------------------------
        # [ 전략 1 ] GOOD 샘플: 맨 앞에 [감정] 태그를 붙여서 상황을 각인시킴
        # -------------------------
        good_input = f"[{emotion}] {ai_text} [SEP] {good_response}"
        samples.append({
            "text": good_input,
            "label": "good"
        })

        # -------------------------
        # [ 전략 2 ] BAD 샘플: '현재와 다른 감정'의 정답을 가져와서 오답으로 학습시킴
        # -------------------------
        # 현재 감정과 '다른' 행들만 필터링
        other_emotion_df = df[df["emotion_state"] != emotion]
        
        if not other_emotion_df.empty:
            # 다른 상황에서 쓰인 엉뚱한 대답을 뽑아옴
            bad_response = random.choice(other_emotion_df["good_response"].tolist())
        else:
            # (예외 처리) 혹시 감정이 다 똑같다면 그냥 다른 대답 뽑기
            all_responses = df["good_response"].tolist()
            bad_response = random.choice(all_responses)
            while bad_response == good_response:
                bad_response = random.choice(all_responses)

        # AI의 현재 상황(ai_text)에 엉뚱한 대답(bad_response)을 붙여서 bad로 가르침
        bad_input = f"[{emotion}] {ai_text} [SEP] {bad_response}"
        samples.append({
            "text": bad_input,
            "label": "bad"
        })

    return pd.DataFrame(samples)

print("데이터셋 구축 중...")
train_dataset = build_dataset(train_df)
test_dataset = build_dataset(test_df)

# =========================
# 4. 입력 / 라벨 분리
# =========================
X_train = train_dataset["text"]
y_train = train_dataset["label"]

X_test = test_dataset["text"]
y_test = test_dataset["label"]

# =========================
# 5. TF-IDF 벡터화 (N-gram 확장)
# =========================
# (1, 3)으로 늘려서 단어 3개짜리 묶음(예: "네 잘못이 아니야")도 하나의 특징으로 잡게 함
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

print("TF-IDF 벡터화 중...")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# 6. 로지스틱 회귀 모델 학습
# =========================
model = LogisticRegression(max_iter=1000)
print("모델 학습 중...")
model.fit(X_train_tfidf, y_train)

# =========================
# 7. 평가
# =========================
y_pred = model.predict(X_test_tfidf)

print(f"\n 정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n[ 분류 보고서 ]")
print(classification_report(y_test, y_pred))

# =========================
# 8. 저장
# =========================
joblib.dump(model, os.path.join(pkl_save_path, "response_quality_model.pkl"))
joblib.dump(tfidf, os.path.join(pkl_save_path, "response_tfidf.pkl"))
print(f"\n 모델 저장 완료: {pkl_save_path}")