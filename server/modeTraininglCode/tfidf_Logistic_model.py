import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =====================================
# 1. 데이터 로드
# =====================================

train_df = pd.read_csv("train_processed.csv")
valid_df = pd.read_csv("valid_processed.csv")

# =====================================
# 2. good / bad 변환
# =====================================

GOOD_LABELS = [
    "sadness",
    "anxiety",
    "hurt",
    "anger",
    "embarrassment"
]

train_df["binary_label"] = train_df["label"].apply(
    lambda x: "good" if x in GOOD_LABELS else "bad"
)

valid_df["binary_label"] = valid_df["label"].apply(
    lambda x: "good" if x in GOOD_LABELS else "bad"
)

# =====================================
# 3. 입력 / 라벨
# =====================================

X_train = train_df["player_input"]
y_train = train_df["binary_label"]

X_valid = valid_df["player_input"]
y_valid = valid_df["binary_label"]

# =====================================
# 4. TF-IDF + Logistic
# =====================================

pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2)
        )
    ),

    (
        "clf",
        LogisticRegression(
            max_iter=1000
        )
    )
])

# =====================================
# 5. 학습
# =====================================

pipeline.fit(X_train, y_train)

# =====================================
# 6. 검증
# =====================================

y_pred = pipeline.predict(X_valid)

print("\n===== VALID RESULT =====")
print(classification_report(y_valid, y_pred))

# =====================================
# 7. 저장
# =====================================

joblib.dump(
    pipeline,
    "tfidf_2class_model.pkl"
)

print("\nTF-IDF 2Class 모델 저장 완료")
