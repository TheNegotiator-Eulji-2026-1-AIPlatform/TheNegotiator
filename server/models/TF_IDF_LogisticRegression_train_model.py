import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# 1. 경로 설정 및 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
# server 폴더 위치 (models의 상위 폴더)
server_dir = os.path.dirname(current_dir)
# data/processed 폴더 위치
data_path = os.path.join(server_dir, "..", "data", "processed")

# pkl 파일을 저장할 폴더 경로 (server/pkls)
pkl_save_path = os.path.join(server_dir, "pkls")

train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

# 2. 데이터 필터링
target_emotions = ['기쁨', '분노', '슬픔']
train_df = train_df[train_df['emotion'].isin(target_emotions)]
test_df = test_df[test_df['emotion'].isin(target_emotions)]

X_train, y_train = train_df['text'], train_df['emotion']
X_test, y_test = test_df['text'], test_df['emotion']

# 3. TF-IDF 벡터화
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. 로지스틱 회귀 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. 성능 평가 및 출력
y_pred = model.predict(X_test_tfidf)
y_score = model.predict_proba(X_test_tfidf) # ROC를 위한 확률값

print(f"테스트 데이터 정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n[ 최종 분류 보고서 ]")
print(classification_report(y_test, y_pred))

# # 6. ROC 곡선 시각화
# classes = sorted(target_emotions)
# y_test_bin = label_binarize(y_test, classes=classes)
# n_classes = len(classes)

# plt.figure(figsize=(8, 6))
# colors = ['blue', 'red', 'green']

# for i in range(n_classes):
#     fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, color=colors[i], lw=2,
#              label=f'ROC curve of {classes[i]} (area = {roc_auc:.2f})')

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) to Multi-class')
# plt.legend(loc="lower right")
# plt.show()

# 7. 모델 저장 (server/pkls 폴더에 저장)
joblib.dump(model, os.path.join(pkl_save_path, 'emotion_model.pkl'))
joblib.dump(tfidf, os.path.join(pkl_save_path, 'tfidf_vectorizer.pkl'))
print(f"\n모델 및 벡터라이저가 '{pkl_save_path}' 에 저장되었습니다 !")