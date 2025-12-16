from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from .config import RESULTS_DIR

def plot_group_accuracy_bar(
    df: pd.DataFrame,
    group_col: str,
    model_name: str,
    out_path: Path,
):
    grouped = (
        df.groupby(group_col)["is_correct_triage"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct_triage": "accuracy"})
    )

    plt.figure()
    plt.bar(grouped[group_col].astype(str), grouped["accuracy"])
    plt.xlabel(group_col)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by {group_col} for {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_model_comparison_bar(
    combined_df: pd.DataFrame,
    group_col: str,
    out_path: Path,
):
    """
    combined_df columns expected:
    [model, group, accuracy]
    where 'group' is the value of group_col.
    """
    # Pivot for plotting: rows = group, columns = model, values = accuracy
    pivot = combined_df.pivot(index="group", columns="model", values="accuracy")

    plt.figure()
    ax = pivot.plot(kind="bar")
    plt.xlabel(group_col)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by {group_col} across models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
