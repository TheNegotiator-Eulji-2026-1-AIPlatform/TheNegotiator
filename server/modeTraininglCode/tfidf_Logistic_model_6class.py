import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =====================================
# 1. 데이터 로드
# =====================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. os.path.join과 '..'을 활용해 최상위 project 폴더를 거쳐 data 폴더로 조인합니다.
#   (current_dir에서 위로 두 번(.., ..) 올라가서 data/processed 폴더와 합치겠다는 뜻!)
data_dir = os.path.join(current_dir, "..", "..", "data", "processed")

# 3. 최종 완성된 절대 경로로 CSV 안전하게 로드!
train_csv_path = os.path.join(data_dir, "train_processed.csv")
valid_csv_path = os.path.join(data_dir, "valid_processed.csv")

train_df = pd.read_csv(train_csv_path)
valid_df = pd.read_csv(valid_csv_path)

# [보정] 데이터셋 내부에 NaN(결측치)이나 숫자가 섞여 있으면 TF-IDF 연산 시 에러가 날 수 있으므로 문스트링 변환 안전장치 추가
train_df["player_input"] = train_df["player_input"].astype(str)
valid_df["player_input"] = valid_df["player_input"].astype(str)

# =====================================
# 2. 6클래스 라벨 그대로 활용 (기존 이진 변환 구역 제거)
# =====================================
# 원본 데이터의 6개 레이블("anger", "anxiety", "sadness", "hurt", "embarrassment", "bad" 등)을 그대로 사용합니다.

# =====================================
# 3. 입력 / 라벨 셋업
# =====================================
X_train = train_df["player_input"]
y_train = train_df["label"]  # 'binary_label' 대신 원본 'label' 주입

X_valid = valid_df["player_input"]
y_valid = valid_df["label"]  # 'binary_label' 대신 원본 'label' 주입

# =====================================
# 4. TF-IDF + Logistic Regression (6클래스 최적화)
# =====================================
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,     # 클래스가 6개로 늘어났으므로, 단어 수용량을 3000에서 5000으로 확장하여 변별력 확보
            ngram_range=(1, 2)
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",  # 6개 다중 클래스 연산을 위해 다항 로지스틱 회귀 옵션 명시
            solver="lbfgs"
        )
    )
])

# =====================================
# 5. 학습
# =====================================
pipeline.fit(X_train, y_train)

# =====================================
# 6. 검증 및 6클래스 성적표 출력
# =====================================
y_pred = pipeline.predict(X_valid)

print("\n===== 6-CLASS VALID RESULT =====")
print(classification_report(y_valid, y_pred))

# =====================================
# 7. 6클래스 전용 파이프라인 저장
# =====================================
joblib.dump(
    pipeline,
    "tfidf_6class_model.pkl"  # 직관적인 구분을 위해 파일명 변경
)

print("\nTF-IDF 6Class 모델 및 파이프라인 저장 완료")