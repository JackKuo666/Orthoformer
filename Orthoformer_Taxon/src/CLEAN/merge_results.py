import os
import pandas as pd
import numpy as np

# Root folder containing subfolders like:
# <root>/<dataset>/{bacformer,our}/{xgb_multiclass_summary.csv | xgboost_summary.csv}
root_dir = "/mnt/disk_c/user_data/liuping/bacformer/results/"
out_csv = "/mnt/disk_c/user_data/liuping/bacformer/merged_summary.csv"

# ---- Binary summary (xgboost_summary.csv) column rename: Chinese -> English ----
BINARY_RENAME = {
    "总样本数":       "total_samples",
    "成功处理文件数": "processed_samples",
    "缺失文件数":     "missing_files",
    "阳性比例":       "positive_ratio",
    "类别分布":       "label_distribution",
    "准确率":         "accuracy",
    "精确率":         "precision",
    "召回率":         "recall",
    "F1分数":         "f1",
    "ROC_AUC":       "roc_auc",
    "模型":           "model",
}

# ---- Multiclass summary expected columns ----
MULTICLASS_EXPECTED = {"accuracy", "macro_precision", "macro_recall", "macro_f1", "n_classes"}

# ---- Unified schema (metadata + metrics) ----
META_COLS = ["parent_folder", "method", "summary_file", "task_type"]
METRIC_COLS = [
    # common metrics
    "accuracy", "precision", "recall", "f1", "roc_auc",
    # multiclass extras
    "macro_precision", "macro_recall", "macro_f1", "n_classes",
    # binary extras
    "total_samples", "processed_samples", "missing_files",
    "positive_ratio", "label_distribution", "model",
]
UNIFIED_COLS = META_COLS + METRIC_COLS

def read_any_csv(path: str) -> pd.DataFrame:
    """
    Read CSV with automatic delimiter detection (comma, tab, etc.).
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)

def normalize_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a binary xgboost_summary.csv to the metric columns only (no metadata here).
    """
    df = df.rename(columns=BINARY_RENAME)

    # Coerce numeric types where applicable
    for col in ["total_samples", "processed_samples", "missing_files"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    for col in ["positive_ratio", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build an output frame containing only METRIC_COLS, creating missing ones as NaN
    out = pd.DataFrame(index=df.index)
    for c in METRIC_COLS:
        out[c] = df[c] if c in df.columns else np.nan
    return out

def normalize_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a multiclass xgb_multiclass_summary.csv to the metric columns only (no metadata here).
    """
    missing = MULTICLASS_EXPECTED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in multiclass summary: {missing}")

    # Coerce numerics
    for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "n_classes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build an output frame containing only METRIC_COLS, creating missing ones as NaN
    out = pd.DataFrame(index=df.index)
    for c in METRIC_COLS:
        out[c] = np.nan
    out["accuracy"]        = df["accuracy"]
    out["macro_precision"] = df["macro_precision"]
    out["macro_recall"]    = df["macro_recall"]
    out["macro_f1"]        = df["macro_f1"]
    out["n_classes"]       = df["n_classes"]
    return out

all_rows = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname not in {"xgboost_summary.csv", "xgb_multiclass_summary.csv"}:
            continue

        fpath = os.path.join(dirpath, fname)

        try:
            df = read_any_csv(fpath)
        except Exception as e:
            print(f"Skip (read error): {fpath} -> {e}")
            continue

        # Parse metadata from path: .../<parent>/<method>/<file>
        parts = os.path.normpath(fpath).split(os.sep)
        parent = parts[-3] if len(parts) >= 3 else "unknown"
        method = parts[-2] if len(parts) >= 2 else "unknown"

        # Normalize metrics only
        if fname == "xgboost_summary.csv":
            task_type = "binary"
            try:
                ndf = normalize_binary(df)
            except Exception as e:
                print(f"Skip (binary normalize error): {fpath} -> {e}")
                continue
        else:  # xgb_multiclass_summary.csv
            task_type = "multiclass"
            try:
                ndf = normalize_multiclass(df)
            except Exception as e:
                print(f"Skip (multiclass normalize error): {fpath} -> {e}")
                continue

        # Remove any pre-existing metadata that might be present
        for col in META_COLS:
            if col in ndf.columns:
                ndf = ndf.drop(columns=[col])

        # Insert fresh metadata
        ndf.insert(0, "task_type", task_type)
        ndf.insert(0, "summary_file", fname)
        ndf.insert(0, "method", method)
        ndf.insert(0, "parent_folder", parent)

        all_rows.append(ndf)

if all_rows:
    merged = pd.concat(all_rows, ignore_index=True, sort=False)

    # Ensure column order (metadata first, then metrics). If any extra columns exist, append them.
    ordered = [c for c in UNIFIED_COLS if c in merged.columns]
    extra = [c for c in merged.columns if c not in ordered]
    merged = merged[ordered + extra]

    merged.to_csv(out_csv, index=False)
    print(f"✅ Saved merged CSV with {len(merged)} rows to {out_csv}")
else:
    print("⚠️ No summary files found.")
    