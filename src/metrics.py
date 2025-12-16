import pandas as pd
from typing import Dict, List

# Triage ordering for over/under-triage analysis
TRIAGE_ORDER = {
    "Self-care": 0,
    "Non-urgent": 1,
    "Urgent": 2,
    "Emergency": 3,
}

def add_error_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_correct_triage"] = df["triage_pred"] == df["correct_triage"]

    df["triage_pred_level"] = df["triage_pred"].map(TRIAGE_ORDER)
    df["correct_triage_level"] = df["correct_triage"].map(TRIAGE_ORDER)

    # Unknown or unmapped labels become NaN; avoid comparisons on them
    valid_mask = df["triage_pred_level"].notna() & df["correct_triage_level"].notna()

    df["over_triage"] = False
    df["under_triage"] = False

    df.loc[valid_mask & (df["triage_pred_level"] > df["correct_triage_level"]), "over_triage"] = True
    df.loc[valid_mask & (df["triage_pred_level"] < df["correct_triage_level"]), "under_triage"] = True

    return df

def group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute accuracy, over-triage rate, and under-triage rate by group.
    """
    df = add_error_flags(df)
    grouped = (
        df.groupby(group_col)
        .agg(
            accuracy=("is_correct_triage", "mean"),
            over_triage_rate=("over_triage", "mean"),
            under_triage_rate=("under_triage", "mean"),
        )
        .reset_index()
    )
    return grouped

def parity_difference(series: pd.Series) -> float:
    values = series.values
    return float(values.max() - values.min())

def summarize_bias(df: pd.DataFrame, group_cols: List[str]) -> Dict[str, Dict]:
    """
    For each group column, compute group metrics and parity differences
    in accuracy, over-triage, and under-triage.
    """
    summary: Dict[str, Dict] = {}
    for col in group_cols:
        stats = group_metrics(df, col)
        summary[col] = {
            "stats": stats,
            "accuracy_parity_diff": parity_difference(stats["accuracy"]),
            "over_triage_parity_diff": parity_difference(stats["over_triage_rate"]),
            "under_triage_parity_diff": parity_difference(stats["under_triage_rate"]),
        }
    return summary

def print_bias_summary(summary: Dict[str, Dict], model_name: str = ""):
    header = f"=== Bias summary for model: {model_name} ===" if model_name else "=== Bias summary ==="
    print("\n" + header)
    for col, info in summary.items():
        print(f"\n--- Group: {col} ---")
        stats = info["stats"]
        print(stats.to_string(index=False))
        print(
            f"Accuracy parity diff: {info['accuracy_parity_diff'] * 100:.1f}% | "
            f"Over-triage parity diff: {info['over_triage_parity_diff'] * 100:.1f}% | "
            f"Under-triage parity diff: {info['under_triage_parity_diff'] * 100:.1f}%"
        )
