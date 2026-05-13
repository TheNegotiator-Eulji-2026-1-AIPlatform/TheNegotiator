"""
preprocess.py
=============
AIHUB 감성대화말뭉치 전처리 파이프라인
The Negotiator — 협상가 프로젝트

Author  : 김동훈
Version : 7.0.0

변경사항
--------
- system_response 복구 (NPC 대사 생성에 활용)
- 감정 3종만 사용: 분노 / 슬픔 / 기쁨

E코드 → 감정 매핑
-----------------
E10~E19 → 분노
E20~E29 → 슬픔
E60~E69 → 기쁨
나머지   → 제거

저장 컬럼
---------
text | system_response | emotion
"""

import os
import re
import json
import glob
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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

# E코드 → 감정 매핑 (3종만)
EMOTION_CODE_MAP = {
    **{f"E{i}": "분노" for i in range(10, 20)},   # E10~E19
    **{f"E{i}": "슬픔" for i in range(20, 30)},   # E20~E29
    **{f"E{i}": "기쁨" for i in range(60, 70)},   # E60~E69
}

VALID_EMOTIONS        = {"분노", "슬픔", "기쁨"}
MAX_SAMPLES_PER_CLASS = 5_000
MIN_TEXT_LENGTH       = 5
RANDOM_SEED           = 42


# ──────────────────────────────────────────────
# STEP 1: JSON 로드
# ──────────────────────────────────────────────

def extract_from_item(item: dict) -> list[dict]:
    """
    단일 항목에서 (text, system_response, emotion) 레코드를 추출한다.

    HS01,HS02,HS03을 각각 개별 샘플로 반환한다.
    - 분노/슬픔/기쁨 외 감정은 스킵
    - 빈 문자열 턴은 스킵
    """
    records = []

    try:
        e_code  = item["profile"]["emotion"]["type"].strip()
        emotion = EMOTION_CODE_MAP.get(e_code)

        # 분노/슬픔/기쁨 외 감정은 스킵
        if not emotion:
            return records

        content = item["talk"]["content"]

        for i in range(1, 4):
            text            = content.get(f"HS0{i}", "").strip()
            system_response = content.get(f"SS0{i}", "").strip()

            # 사람 발화가 없으면 스킵
            if not text:
                continue

            records.append({
                "text":            text,
                "system_response": system_response,
                "emotion":         emotion,
            })

    except (KeyError, TypeError, AttributeError):
        pass

    return records


def load_single_json(file_path: str) -> list[dict]:
    """단일 JSON 파일에서 레코드 리스트를 반환한다."""
    records = []

    try:
        with open(file_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"JSON 파싱 실패 — {os.path.basename(file_path)}: {e}")
        return records

    data_list = raw if isinstance(raw, list) else raw.get("data", [])

    for item in data_list:
        result = extract_from_item(item)
        if result:
            records.extend(result)

    return records


def load_all_json(data_dir: str) -> pd.DataFrame:
    """디렉터리 내 모든 JSON 파일을 재귀 탐색하여 DataFrame으로 합친다."""
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)

    if not json_files:
        raise FileNotFoundError(
            f"JSON 파일을 찾을 수 없습니다: {data_dir}\n"
            "AIHUB에서 다운받은 라벨링데이터 폴더를 data/raw/ 안에 넣어주세요."
        )

    logger.info(f"총 {len(json_files)}개 JSON 파일 발견")

    all_records = []
    for i, path in enumerate(json_files, 1):
        records = load_single_json(path)
        all_records.extend(records)
        if i % 50 == 0 or i == len(json_files):
            logger.info(f"  [{i}/{len(json_files)}] 누적 {len(all_records):,}건")

    if not all_records:
        raise ValueError("유효한 데이터가 없습니다. JSON 구조를 확인해주세요.")

    df = pd.DataFrame(all_records)
    logger.info(f"원본 총 발화 수: {len(df):,}건 (분노/슬픔/기쁨만)")
    return df


# ──────────────────────────────────────────────
# STEP 2: 결측치 처리
# ──────────────────────────────────────────────

def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    """빈 발화 및 5자 미만 초단문을 제거한다."""
    before = len(df)
    df = df.dropna(subset=["text", "emotion"])
    df = df[df["text"].str.strip() != ""]
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    after = len(df)
    logger.info(f"결측치/초단문 제거: {before:,} → {after:,}건 ({before - after:,}건 삭제)")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# STEP 3: 텍스트 정규화
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,!?~…]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(normalize_text)
    before = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH].reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.info(f"정규화 후 재제거: {before - after:,}건")
    logger.info("텍스트 정규화 완료")
    return df


# ──────────────────────────────────────────────
# STEP 4: 레이블 검증
# ──────────────────────────────────────────────

def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """VALID_EMOTIONS 외 레이블 행 제거 및 분포 출력."""
    before = len(df)
    df = df[df["emotion"].isin(VALID_EMOTIONS)].reset_index(drop=True)
    after = len(df)
    logger.info(f"레이블 검증 후 유효 데이터: {after:,}건 ({before - after:,}건 제거)")
    logger.info(f"감정 분포:\n{df['emotion'].value_counts().to_string()}")
    return df


# ──────────────────────────────────────────────
# STEP 5: 클래스 불균형 처리 (다운샘플링)
# ──────────────────────────────────────────────

def downsample(df: pd.DataFrame, max_per_class: int = MAX_SAMPLES_PER_CLASS) -> pd.DataFrame:
    """각 감정 클래스를 max_per_class 이하로 다운샘플링."""
    frames = []
    for emotion, group in df.groupby("emotion"):
        frames.append(group.sample(min(len(group), max_per_class), random_state=RANDOM_SEED))
    sampled = pd.concat(frames).reset_index(drop=True)
    logger.info(f"다운샘플링 후 총 데이터: {len(sampled):,}건")
    logger.info(f"클래스 분포:\n{sampled['emotion'].value_counts().to_string()}")
    return sampled


# ──────────────────────────────────────────────
# STEP 6: 데이터 분리 (Stratified Split)
# ──────────────────────────────────────────────

def split_dataset(df: pd.DataFrame):
    """Stratified Split으로 Train / Validation / Test = 8:1:1 분리."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["emotion"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=RANDOM_SEED
    )
    logger.info(
        f"데이터 분리 완료 — "
        f"Train: {len(train_df):,}건 | Val: {len(val_df):,}건 | Test: {len(test_df):,}건"
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ──────────────────────────────────────────────
# 전처리 요약 출력
# ──────────────────────────────────────────────

def print_summary(raw_count, final_df, train_df, val_df, test_df, output_dir):
    sep = "=" * 55
    print(f"\n{sep}")
    print("  📋  전처리 요약 리포트  —  The Negotiator")
    print(sep)

    print("\n【 사용한 컬럼 】")
    print("  ✅  HS01, HS02, HS03     → 학습 텍스트 (text)")
    print("  ✅  SS01, SS02, SS03     → NPC 응답 (system_response)")
    print("  ✅  E코드 → 감정 3종     → 분노 / 슬픔 / 기쁨")
    print("  ❌  불안 / 당황 / 상처   → 제거")
    print("  ❌  연령 / 성별 / 상황키워드 / 신체질환 → 제거")

    print("\n【 전처리 단계별 결과 】")
    print(f"  STEP 1  JSON 로드 & HS01,02,03 개별 추출  {raw_count:>8,} 건")
    print(f"  STEP 2  결측치 / 5자 미만 제거")
    print(f"  STEP 3  텍스트 정규화 (HTML·특수문자·공백)")
    print(f"  STEP 4  레이블 검증 (분노/슬픔/기쁨만 유지)")
    print(f"  STEP 5  다운샘플링 (클래스당 최대 {MAX_SAMPLES_PER_CLASS:,}건)")
    print(f"          → 최종 {len(final_df):,} 건")

    print("\n【 감정 클래스 분포 (다운샘플링 후) 】")
    for emotion, cnt in final_df["emotion"].value_counts().items():
        bar = "█" * (cnt // 200)
        print(f"  {emotion:4s}  {cnt:,}건  {bar}")

    print("\n【 데이터 분리 결과 (8:1:1 Stratified Split) 】")
    print(f"  Train  :  {len(train_df):,} 건  ({len(train_df)/len(final_df)*100:.0f}%)")
    print(f"  Val    :  {len(val_df):,} 건  ({len(val_df)/len(final_df)*100:.0f}%)")
    print(f"  Test   :  {len(test_df):,} 건  ({len(test_df)/len(final_df)*100:.0f}%)")

    print("\n【 저장 경로 】")
    print(f"  {output_dir}/train.csv")
    print(f"  {output_dir}/val.csv")
    print(f"  {output_dir}/test.csv")
    print(f"\n  저장 컬럼: text  |  system_response  |  emotion")
    print(f"{sep}\n")


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────

def run_pipeline(data_dir: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("  The Negotiator — 데이터 전처리 파이프라인 시작")
    logger.info("=" * 55)

    logger.info("[STEP 1] JSON 로드 — HS01,HS02,HS03 개별 샘플 추출")
    df = load_all_json(data_dir)
    raw_count = len(df)

    logger.info("[STEP 2] 결측치 / 초단문 제거")
    df = remove_missing(df)

    logger.info("[STEP 3] 텍스트 정규화")
    df = apply_normalization(df)

    logger.info("[STEP 4] 레이블 검증 (분노/슬픔/기쁨만 유지)")
    df = validate_labels(df)

    logger.info("[STEP 5] 클래스 불균형 처리 (다운샘플링)")
    df = downsample(df)

    logger.info("[STEP 6] Stratified Split (8:1:1)")
    train_df, val_df, test_df = split_dataset(df)

    # text, system_response, emotion 3개 컬럼 저장
    cols = ["text", "system_response", "emotion"]
    train_df[cols].to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8-sig")
    val_df[cols].to_csv(  os.path.join(output_dir, "val.csv"),   index=False, encoding="utf-8-sig")
    test_df[cols].to_csv( os.path.join(output_dir, "test.csv"),  index=False, encoding="utf-8-sig")

    print_summary(raw_count, df, train_df, val_df, test_df, output_dir)


# ──────────────────────────────────────────────
# 엔트리 포인트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AIHUB 감성대화말뭉치 전처리 파이프라인 — The Negotiator"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/raw",
        help="AIHUB JSON 파일이 있는 디렉터리 (기본값: ./data/raw)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="전처리 결과 CSV 저장 디렉터리 (기본값: ./data/processed)",
    )

    args = parser.parse_args()
    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)
