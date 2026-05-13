import os
import random
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# =========================
# 경로 설정
# =========================

current_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(current_dir)

data_path = os.path.join(
    server_dir,
    "..",
    "data",
    "processed"
)

pkl_save_path = os.path.join(server_dir, "models")


# =========================
# 데이터 로드
# =========================

train_df = pd.read_csv(
    os.path.join(data_path, "train_processed.csv"),
    encoding="utf-8-sig"
)

test_df = pd.read_csv(
    os.path.join(data_path, "valid_processed.csv"),
    encoding="utf-8-sig"
)

# =========================
# 학습 데이터 생성
# =========================

def build_dataset(df):

    samples = []

    responses = df["good_response"].tolist()

    for idx, row in df.iterrows():

        ai_text = row["ai_text"]

        good_response = row["good_response"]

        # -------------------------
        # GOOD 샘플
        # -------------------------

        good_input = f"{ai_text} [SEP] {good_response}"

        samples.append({
            "text": good_input,
            "label": "good"
        })

        # -------------------------
        # BAD 샘플
        # 랜덤 응답 섞기
        # -------------------------

        random_response = random.choice(responses)

        while random_response == good_response:
            random_response = random.choice(responses)

        bad_input = f"{ai_text} [SEP] {random_response}"

        samples.append({
            "text": bad_input,
            "label": "bad"
        })

    return pd.DataFrame(samples)


train_dataset = build_dataset(train_df)
test_dataset = build_dataset(test_df)


# =========================
# 입력 / 라벨
# =========================

X_train = train_dataset["text"]
y_train = train_dataset["label"]

X_test = test_dataset["text"]
y_test = test_dataset["label"]


# =========================
# TF-IDF 벡터화
# =========================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# =========================
# 로지스틱 회귀 모델
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# =========================
# 평가
# =========================

y_pred = model.predict(X_test_tfidf)

print(f"\n정확도: {accuracy_score(y_test, y_pred):.4f}")

print("\n[분류 보고서]")
print(classification_report(y_test, y_pred))


# =========================
# 저장
# =========================

joblib.dump(
    model,
    os.path.join(pkl_save_path, "response_quality_model.pkl")
)

joblib.dump(
    tfidf,
    os.path.join(pkl_save_path, "response_tfidf.pkl")
)

print("\n모델 저장 완료")