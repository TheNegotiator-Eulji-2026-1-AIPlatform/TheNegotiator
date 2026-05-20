import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. 데이터 로드 (미리 가공된 파일 사용)
train_df = pd.read_csv("train_processed.csv")
valid_df = pd.read_csv("valid_processed.csv")

# 2. 클래스 매핑 설정 (순서 엄수)
classes = ["anger", "anxiety", "sadness", "hurt", "embarrassment", "bad"]
id2label = {i: label for i, label in enumerate(classes)}
label2id = {label: i for i, label in enumerate(classes)}

# 정답 라벨을 숫자 ID로 변환
train_df['label_id'] = train_df['label'].map(label2id)
valid_df['label_id'] = valid_df['label'].map(label2id)

# 3. [핵심] 입력 데이터 분리 및 Hugging Face Dataset 변환
# ai_text를 완전히 배제하고, 오직 player_input만 'text' 컬럼으로 지정하여 모델에 던집니다.
train_df = train_df.rename(columns={'player_input': 'text'})
valid_df = valid_df.rename(columns={'player_input': 'text'})

train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']])
valid_dataset = Dataset.from_pandas(valid_df[['text', 'label_id']])

# 4. 토크나이저 및 모델 로드
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 분류용 모델로 로드하며, 우리가 정한 6개 클래스에 맞게 출력층 세팅
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(classes), id2label=id2label, label2id=label2id
)

# 5. 토크나이징 함수 (글자 수 제한 및 패딩)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True).rename_column("label_id", "labels")
tokenized_valid = valid_dataset.map(tokenize_function, batched=True).rename_column("label_id", "labels")

# 6. 평가 지표 함수 설정 (검증용)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# 7. 학습 세팅 (Training Arguments)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",            # 매 에폭 끝날 때마다 valid.csv로 성능 검사
    save_strategy="epoch",            # 매 에폭마다 가중치 저장
    load_best_model_at_end=True,      # 학습이 끝나면 가장 F1 점수가 높았던 '최적의 모델'로 롤백
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    compute_metrics=compute_metrics
)

# 8. 본격적인 학습 시작
print("모델 학습을 시작합니다...")
trainer.train()

# 9. 최종 모델 및 토크나이저 저장 (pkl 대신 HuggingFace 표준 방식 적용)
# 지정한 폴더 안에 config.json, model.safetensors, 토크나이저 설정 파일들이 쪼개져서 안전하게 저장됩니다.
save_path = "./klue_bert_emotion_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"학습 완료! 최고 성능의 모델이 '{save_path}' 폴더에 저장되었습니다.")