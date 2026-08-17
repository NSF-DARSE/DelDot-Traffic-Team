#!/usr/bin/env python3
"""
Validate a traffic forecast output file.

Usage:
    python validate_submission.py my_submission.csv

Optional:
    python validate_submission.py my_submission.csv --template submission_template.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ["target_id", "station_key", "direction_code", "timestamp"]
OUTPUT_COLUMNS = ["forecast_volume", "lower_90", "upper_90", "reliability_score"]
REQUIRED_COLUMNS = ID_COLUMNS + OUTPUT_COLUMNS


def normalize_direction(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")


def validate(submission_path: Path, template_path: Path) -> int:
    errors = []

    if not submission_path.exists():
        print(f"ERROR: file not found: {submission_path}")
        return 2
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}")
        return 2

    try:
        sub = pd.read_csv(submission_path, low_memory=False)
        template = pd.read_csv(template_path, low_memory=False)
    except Exception as exc:
        print(f"ERROR: could not read CSV file: {exc}")
        return 2

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in sub.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        print("PASS: False")
        return 1

    if len(sub) != len(template):
        errors.append(
            f"Row count mismatch: file={len(sub):,}, template={len(template):,}"
        )

    if sub["target_id"].isna().any():
        errors.append("target_id contains missing values.")

    duplicate_count = int(sub["target_id"].duplicated().sum())
    if duplicate_count:
        errors.append(f"Duplicate target_id rows: {duplicate_count}")

    template_ids = set(template["target_id"].astype(str))
    submission_ids = set(sub["target_id"].astype(str))

    missing_ids = template_ids - submission_ids
    extra_ids = submission_ids - template_ids

    if missing_ids:
        errors.append(f"Missing template target IDs: {len(missing_ids)}")
    if extra_ids:
        errors.append(f"Unexpected target IDs: {len(extra_ids)}")

    if not duplicate_count:
        left = template[ID_COLUMNS].copy()
        right = sub[ID_COLUMNS].copy()

        left["target_id"] = left["target_id"].astype(str)
        right["target_id"] = right["target_id"].astype(str)

        for df in (left, right):
            df["station_key"] = df["station_key"].astype(str).str.strip()
            df["direction_code"] = normalize_direction(df["direction_code"])
            df["timestamp"] = normalize_timestamp(df["timestamp"])

        joined = left.merge(
            right,
            on="target_id",
            how="left",
            suffixes=("_template", "_file"),
            validate="one_to_one",
        )

        for col in ["station_key", "direction_code", "timestamp"]:
            a = joined[f"{col}_template"]
            b = joined[f"{col}_file"]
            mismatch = (a != b) & ~(a.isna() & b.isna())
            if mismatch.any():
                errors.append(
                    f"{col} was altered for {int(mismatch.sum())} target rows."
                )

    numeric = {}
    for col in OUTPUT_COLUMNS:
        numeric[col] = pd.to_numeric(sub[col], errors="coerce")
        bad = int(numeric[col].isna().sum())
        if bad:
            errors.append(f"{col} has {bad} missing or non-numeric values.")
        finite = numeric[col].notna()
        if finite.any():
            nonfinite = int((~np.isfinite(numeric[col][finite])).sum())
            if nonfinite:
                errors.append(f"{col} has {nonfinite} non-finite values.")

    pred = numeric["forecast_volume"]
    lo = numeric["lower_90"]
    hi = numeric["upper_90"]
    rel = numeric["reliability_score"]

    if (pred < 0).any():
        errors.append(f"forecast_volume has {int((pred < 0).sum())} negative values.")
    if (lo < 0).any():
        errors.append(f"lower_90 has {int((lo < 0).sum())} negative values.")

    invalid_interval = (lo > pred) | (pred > hi)
    if invalid_interval.any():
        errors.append(
            f"Invalid interval ordering for {int(invalid_interval.sum())} rows."
        )

    invalid_reliability = ~rel.between(0, 1, inclusive="both")
    if invalid_reliability.any():
        errors.append(
            f"reliability_score is outside [0,1] for "
            f"{int(invalid_reliability.sum())} rows."
        )

    print("\nForecast file validation")
    print("------------------------")
    print(f"Rows: {len(sub):,}")
    print(f"Template rows: {len(template):,}")
    print(f"Unique target IDs: {sub['target_id'].nunique():,}")

    for error in errors:
        print("ERROR:", error)

    passed = len(errors) == 0
    print(f"PASS: {passed}")
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a traffic forecast file.")
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent / "submission_template.csv",
    )
    args = parser.parse_args()
    sys.exit(validate(args.submission, args.template))


if __name__ == "__main__":
    main()
