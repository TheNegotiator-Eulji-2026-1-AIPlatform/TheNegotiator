"""
train_kobert.py
===============
KLUE-BERT 감정 분류 모델 — VSCode용
The Negotiator — 협상가 프로젝트

Author  : 김동훈
Version : 9.0.0

실행 전 설치
-----------
pip install torch transformers joblib scikit-learn pandas matplotlib

실행 방법
---------
python train_kobert.py
python train_kobert.py --train_path ./train_processed.csv --valid_path ./valid_processed.csv

입력 파일
---------
- train_processed.csv : ai_text | player_input | label
- valid_processed.csv : ai_text | player_input | label

출력 파일
---------
- models/klue_bert/model.pkl
- models/klue_bert/tokenizer.pkl
- models/klue_bert/config.json
- models/klue_bert/confusion_matrix.png
- models/klue_bert/training_curve.png
- models/klue_bert/evaluation_report.txt
"""

import os
import json
import joblib
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────
DEFAULT_TRAIN_PATH = "./train_processed.csv"
DEFAULT_VALID_PATH = "./valid_processed.csv"
MODEL_DIR          = "./models/klue_bert"

PRETRAINED_MODEL = "klue/bert-base"

# 명세서 기준 클래스 순서
CLASS_ORDER = ["anger", "anxiety", "sadness", "hurt", "embarrassment", "bad"]
LABEL2ID    = {label: i for i, label in enumerate(CLASS_ORDER)}
ID2LABEL    = {i: label for i, label in enumerate(CLASS_ORDER)}
NUM_LABELS  = len(CLASS_ORDER)

# 게이지 로직 규칙
DELTA_RULES = {
    "bad":           {"stability_delta": -10, "anger_delta":  15},
    "anger":         {"stability_delta":  20, "anger_delta": -15},
    "anxiety":       {"stability_delta":  20, "anger_delta": -15},
    "sadness":       {"stability_delta":  20, "anger_delta": -15},
    "hurt":          {"stability_delta":  20, "anger_delta": -15},
    "embarrassment": {"stability_delta":  20, "anger_delta": -15},
}

# 하이퍼파라미터
MAX_LENGTH   = 128
BATCH_SIZE   = 16    # CPU면 16, GPU면 32로 올리세요
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class NegotiatorDataset(Dataset):
    """입력 형식: ai_text [SEP] player_input"""

    def __init__(self, df, tokenizer, max_length):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            str(row["ai_text"]),
            str(row["player_input"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label":          torch.tensor(row["label_id"], dtype=torch.long),
        }


def make_dataloader(df, tokenizer, shuffle):
    return DataLoader(
        NegotiatorDataset(df, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
def load_data(train_path, valid_path):
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    train_df = train_df[train_df["label"].isin(CLASS_ORDER)].dropna().reset_index(drop=True)
    valid_df = valid_df[valid_df["label"].isin(CLASS_ORDER)].dropna().reset_index(drop=True)

    train_df["label_id"] = train_df["label"].map(LABEL2ID)
    valid_df["label_id"] = valid_df["label"].map(LABEL2ID)

    logger.info(f"Train: {len(train_df):,}건 | Valid: {len(valid_df):,}건")
    logger.info(f"클래스 분포 (Train):\n{train_df['label'].value_counts().to_string()}")

    return train_df, valid_df


# ──────────────────────────────────────────────
# 학습 1 에포크
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            logger.info(f"  Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, macro_f1, all_preds, all_labels, all_probs


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
def plot_confusion_matrix(cm, output_dir):
    font_candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_LABELS))
    ax.set_yticks(range(NUM_LABELS))
    ax.set_xticklabels(CLASS_ORDER, fontsize=11, rotation=30)
    ax.set_yticklabels(CLASS_ORDER, fontsize=11)
    ax.set_xlabel("예측", fontsize=13)
    ax.set_ylabel("실제", fontsize=13)
    ax.set_title("Confusion Matrix — KLUE-BERT", fontsize=13)
    thresh = cm.max() / 2.0
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion Matrix 저장 → {save_path}")


def plot_training_curve(history, output_dir):
    epochs_x = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_x, [h["loss"] for h in history], "b-o")
    ax1.set_title("Train Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True)
    ax2.plot(epochs_x, [h["acc"] for h in history], "g-o", label="Accuracy")
    ax2.plot(epochs_x, [h["f1"]  for h in history], "r-s", label="Macro F1")
    ax2.set_title("Validation 성능"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"학습 곡선 저장 → {save_path}")


# ──────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────
def save_results(model, tokenizer, metrics, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model.state_dict(), os.path.join(output_dir, "model.pkl"))
    joblib.dump(tokenizer,          os.path.join(output_dir, "tokenizer.pkl"))

    config = {
        "class_order": CLASS_ORDER,
        "label2id":    LABEL2ID,
        "id2label":    {str(k): v for k, v in ID2LABEL.items()},
        "max_length":  MAX_LENGTH,
        "delta_rules": DELTA_RULES,
        "pretrained":  PRETRAINED_MODEL,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  KLUE-BERT 성능 평가 — The Negotiator\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1 : {metrics['macro_f1']:.4f}\n\n")
        f.write(metrics["report"])

    logger.info(f"모델 저장 완료 → {output_dir}")


# ──────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────
def print_summary(metrics, history):
    sep = "=" * 55
    print(f"\n{sep}")
    print("  📋  KLUE-BERT 학습 완료 — The Negotiator")
    print(sep)
    print("\n【 에포크별 성능 】")
    for h in history:
        print(f"  Epoch {h['epoch']} | Loss: {h['loss']:.4f} | Acc: {h['acc']:.4f} | F1: {h['f1']:.4f}")
    print(f"\n【 최종 Valid 성능 】")
    print(f"  Accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"\n{metrics['report']}")
    print(f"【 저장 경로 】")
    print(f"  {MODEL_DIR}/model.pkl")
    print(f"  {MODEL_DIR}/tokenizer.pkl")
    print(f"  {MODEL_DIR}/config.json")
    print(f"  {MODEL_DIR}/confusion_matrix.png")
    print(f"  {MODEL_DIR}/training_curve.png")
    print(f"{sep}\n")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main(train_path, valid_path):
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("  The Negotiator — KLUE-BERT 학습 시작")
    logger.info(f"  디바이스: {DEVICE}")
    logger.info(f"  에포크: {EPOCHS} | 배치: {BATCH_SIZE} | 최대 길이: {MAX_LENGTH}")
    logger.info("=" * 55)

    # 1. 데이터 로드
    train_df, valid_df = load_data(train_path, valid_path)

    # 2. 모델 로드
    logger.info(f"KLUE-BERT 로드 중: {PRETRAINED_MODEL}")
    logger.info("(첫 실행 시 약 400MB 자동 다운로드)")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels = NUM_LABELS,
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
    )
    model.to(DEVICE)

    # 3. DataLoader
    train_loader = make_dataloader(train_df, tokenizer, shuffle=True)
    valid_loader = make_dataloader(valid_df, tokenizer, shuffle=False)
    logger.info(f"Train 배치: {len(train_loader)} | Valid 배치: {len(valid_loader)}")

    # 4. Optimizer & Scheduler
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    logger.info(f"전체 스텝: {total_steps:,} | Warmup: {warmup_steps:,}")

    # 5. 학습 루프
    best_val_f1 = 0.0
    best_epoch  = 0
    history     = []

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\n{'─'*45}")
        logger.info(f"Epoch {epoch} / {EPOCHS}")
        logger.info(f"{'─'*45}")

        train_loss               = train_epoch(model, train_loader, optimizer, scheduler)
        val_acc, val_f1, _, _, _ = evaluate_model(model, valid_loader)

        logger.info(f"Train Loss : {train_loss:.4f}")
        logger.info(f"Val Acc    : {val_acc:.4f}")
        logger.info(f"Val F1     : {val_f1:.4f}")

        history.append({"epoch": epoch, "loss": train_loss, "acc": val_acc, "f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            joblib.dump(model.state_dict(), os.path.join(MODEL_DIR, "model.pkl"))
            logger.info(f"✅ 최고 모델 저장 (Epoch {epoch} | F1: {val_f1:.4f})")

    logger.info(f"\n학습 완료 — 최고: Epoch {best_epoch} (F1: {best_val_f1:.4f})")

    # 6. 최고 모델 평가
    model.load_state_dict(joblib.load(os.path.join(MODEL_DIR, "model.pkl")))
    val_acc, val_f1, val_preds, val_labels, _ = evaluate_model(model, valid_loader)

    report  = classification_report(val_labels, val_preds, target_names=CLASS_ORDER, digits=4)
    cm      = confusion_matrix(val_labels, val_preds)
    metrics = {"accuracy": val_acc, "macro_f1": val_f1, "report": report}

    # 7. 시각화
    plot_confusion_matrix(cm, MODEL_DIR)
    plot_training_curve(history, MODEL_DIR)

    # 8. 저장
    save_results(model, tokenizer, metrics, MODEL_DIR)

    # 9. 요약
    print_summary(metrics, history)


# ──────────────────────────────────────────────
# 엔트리 포인트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLUE-BERT 감정 분류 — The Negotiator")
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid_path", type=str, default=DEFAULT_VALID_PATH)
    args = parser.parse_args()
    main(train_path=args.train_path, valid_path=args.valid_path)
