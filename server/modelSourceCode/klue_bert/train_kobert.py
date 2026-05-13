# ================================================================
# The Negotiator — KLUE-BERT 감정 분류 모델
# Google Colab 전용
#
# 실행 순서: 셀을 위에서부터 순서대로 실행하세요
# ================================================================


# ──────────────────────────────────────────────
# 셀 1: GPU 확인
# ──────────────────────────────────────────────
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  GPU가 없습니다. 런타임 → 런타임 유형 변경 → T4 GPU 선택 후 재실행하세요.")


# ──────────────────────────────────────────────
# 셀 2: 라이브러리 설치
# ──────────────────────────────────────────────
# !pip install transformers -q


# ──────────────────────────────────────────────
# 셀 3: CSV 파일 업로드
# ──────────────────────────────────────────────
from google.colab import files

print("train_processed.csv 와 valid_processed.csv 를 업로드하세요.")
uploaded = files.upload()

# 업로드된 파일 확인
import os
print("\n업로드된 파일:", list(uploaded.keys()))


# ──────────────────────────────────────────────
# 셀 4: 라이브러리 임포트 & 상수 정의
# ──────────────────────────────────────────────
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 파일 경로 (업로드한 파일명과 맞춰주세요) ──
TRAIN_PATH = "train_processed.csv"
VALID_PATH = "valid_processed.csv"
MODEL_DIR  = "./models/klue_bert"

# ── 컬럼명 ──
TEXT_COL  = "good_response"
LABEL_COL = "emotion_state"

# ── KLUE-BERT ──
PRETRAINED_MODEL = "klue/bert-base"

# ── 감정 레이블 ──
EMOTION_LABELS = ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]
LABEL2ID       = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL       = {i: label for i, label in enumerate(EMOTION_LABELS)}
NUM_LABELS     = len(EMOTION_LABELS)

# ── 하이퍼파라미터 ──
MAX_LENGTH   = 128   # GPU 있으면 128 권장
BATCH_SIZE   = 32    # GPU 있으면 32 권장
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"설정 완료 — 디바이스: {DEVICE}")


# ──────────────────────────────────────────────
# 셀 5: 데이터 로드
# ──────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

# 결측치 및 유효하지 않은 레이블 제거
train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL])
valid_df = valid_df.dropna(subset=[TEXT_COL, LABEL_COL])
train_df = train_df[train_df[LABEL_COL].isin(EMOTION_LABELS)].reset_index(drop=True)
valid_df = valid_df[valid_df[LABEL_COL].isin(EMOTION_LABELS)].reset_index(drop=True)

# valid → val(50%) / test(50%) 분리
val_df, test_df = train_test_split(
    valid_df, test_size=0.5, stratify=valid_df[LABEL_COL], random_state=RANDOM_SEED
)
val_df  = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 정수 레이블 변환
for df in [train_df, val_df, test_df]:
    df["label"] = df[LABEL_COL].map(LABEL2ID)

print(f"Train: {len(train_df):,}건 | Val: {len(val_df):,}건 | Test: {len(test_df):,}건")
print(f"\n감정 분포 (Train):\n{train_df[LABEL_COL].value_counts().to_string()}")


# ──────────────────────────────────────────────
# 셀 6: Dataset & DataLoader
# ──────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_dataloader(df, tokenizer, shuffle):
    dataset = EmotionDataset(
        texts      = df[TEXT_COL].astype(str).tolist(),
        labels     = df["label"].tolist(),
        tokenizer  = tokenizer,
        max_length = MAX_LENGTH,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


print("Dataset 클래스 정의 완료")


# ──────────────────────────────────────────────
# 셀 7: 모델 & 토크나이저 로드
# ──────────────────────────────────────────────
print(f"KLUE-BERT 로드 중: {PRETRAINED_MODEL}")
print("(첫 실행 시 약 400MB 다운로드 — 잠시 기다려주세요)")

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels = NUM_LABELS,
    id2label   = ID2LABEL,
    label2id   = LABEL2ID,
)
model.to(DEVICE)

train_loader = make_dataloader(train_df, tokenizer, shuffle=True)
val_loader   = make_dataloader(val_df,   tokenizer, shuffle=False)
test_loader  = make_dataloader(test_df,  tokenizer, shuffle=False)

print(f"모델 로드 완료")
print(f"Train 배치 수: {len(train_loader)} | Val 배치 수: {len(val_loader)}")


# ──────────────────────────────────────────────
# 셀 8: 학습 함수 정의
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, macro_f1, all_preds, all_labels


print("학습 함수 정의 완료")


# ──────────────────────────────────────────────
# 셀 9: 학습 실행
# ──────────────────────────────────────────────
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps,
)

print(f"학습 시작 — {EPOCHS} 에포크 | 전체 스텝: {total_steps:,} | Warmup: {warmup_steps:,}\n")

best_val_f1 = 0.0
best_epoch  = 0
history     = []

for epoch in range(1, EPOCHS + 1):
    print(f"{'─'*45}")
    print(f"Epoch {epoch} / {EPOCHS}")
    print(f"{'─'*45}")

    train_loss            = train_epoch(model, train_loader, optimizer, scheduler)
    val_acc, val_f1, _, _ = evaluate_model(model, val_loader)

    print(f"Train Loss  : {train_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val Macro F1: {val_f1:.4f}\n")

    history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "val_f1": val_f1})

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
        print(f"✅ 최고 모델 저장 (Epoch {epoch} | Val F1: {val_f1:.4f})\n")

print(f"학습 완료 — 최고 성능: Epoch {best_epoch} (Val F1: {best_val_f1:.4f})")


# ──────────────────────────────────────────────
# 셀 10: Test 평가
# ──────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
test_acc, test_f1, test_preds, test_labels = evaluate_model(model, test_loader)

label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
report = classification_report(test_labels, test_preds, target_names=label_names, digits=4)
cm     = confusion_matrix(test_labels, test_preds)

print("=" * 55)
print("  📋  최종 Test 결과")
print("=" * 55)
print(f"\nTest Accuracy  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"Test Macro F1  : {test_f1:.4f}")
print(f"\n{report}")


# ──────────────────────────────────────────────
# 셀 11: Confusion Matrix 시각화
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(EMOTION_LABELS)))
ax.set_yticks(range(len(EMOTION_LABELS)))
ax.set_xticklabels(EMOTION_LABELS, fontsize=12)
ax.set_yticklabels(EMOTION_LABELS, fontsize=12)
ax.set_xlabel("예측 감정", fontsize=13)
ax.set_ylabel("실제 감정", fontsize=13)
ax.set_title("Confusion Matrix — KLUE-BERT Fine-tuning", fontsize=13)

thresh = cm.max() / 2.0
for i in range(len(EMOTION_LABELS)):
    for j in range(len(EMOTION_LABELS)):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("Confusion Matrix 저장 완료")


# ──────────────────────────────────────────────
# 셀 12: 학습 곡선 시각화
# ──────────────────────────────────────────────
epochs_list = [h["epoch"] for h in history]
losses      = [h["train_loss"] for h in history]
val_accs    = [h["val_acc"] for h in history]
val_f1s     = [h["val_f1"] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_list, losses, "b-o", label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Train Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_list, val_accs, "g-o", label="Val Accuracy")
ax2.plot(epochs_list, val_f1s,  "r-s", label="Val Macro F1")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.set_title("Validation 성능")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curve.png"), dpi=150)
plt.show()
print("학습 곡선 저장 완료")


# ──────────────────────────────────────────────
# 셀 13: 모델 파일 다운로드
# ──────────────────────────────────────────────
import shutil

# 모델 파일들을 zip으로 묶어서 다운로드
shutil.make_archive("klue_bert_model", "zip", MODEL_DIR)
files.download("klue_bert_model.zip")
print("모델 다운로드 완료!")


# ──────────────────────────────────────────────
# 셀 14: 추론 테스트 (학습 후 직접 테스트)
# ──────────────────────────────────────────────
def predict(text: str) -> dict:
    """
    텍스트 입력 → 감정 예측

    Returns
    -------
    {"emotion": str, "confidence": float, "all_scores": dict}
    """
    model.eval()
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    pred_idx    = int(np.argmax(probs))
    emotion     = ID2LABEL[pred_idx]
    confidence  = float(probs[pred_idx])
    all_scores  = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(NUM_LABELS)}

    return {"emotion": emotion, "confidence": confidence, "all_scores": all_scores}


# 테스트
test_sentences = [
    "너무 화가 나서 참을 수가 없어.",
    "오늘 너무 슬프고 눈물이 나.",
    "정말 기뻐서 너무 좋아!",
]

print("=" * 45)
print("  추론 테스트")
print("=" * 45)
for text in test_sentences:
    result = predict(text)
    print(f"\n입력: {text}")
    print(f"예측: {result['emotion']}  (확신도: {result['confidence']*100:.1f}%)")
    print(f"전체: {result['all_scores']}")
