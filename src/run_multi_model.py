import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

from .config import RESULTS_DIR, LLM_CONFIGS
from .llm_runner_multi import run_llm_for_model
from .metrics import add_error_flags, group_metrics, summarize_bias, print_bias_summary
from .charts import plot_group_accuracy_bar, plot_model_comparison_bar

GROUP_COLS = ["gender", "ethnicity", "english_level", "age"]

def load_or_run_model(model_name: str) -> pd.DataFrame:
    out_path = RESULTS_DIR / f"{model_name}_outputs.csv"
    if out_path.exists():
        print(f"Loading existing outputs for {model_name} from {out_path}")
        df = pd.read_csv(out_path)
    else:
        df = run_llm_for_model(model_name)
    return df

def build_combined_metrics(
    model_dfs: Dict[str, pd.DataFrame],
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Build a combined metrics table across models and groups.
    Columns: [model, group_col, group_value, accuracy, over_triage_rate, under_triage_rate]
    """
    records = []
    for model_name, df in model_dfs.items():
        df_flags = add_error_flags(df)
        for col in group_cols:
            gm = group_metrics(df_flags, col)
            for _, row in gm.iterrows():
                records.append({
                    "model": model_name,
                    "group_col": col,
                    "group": row[col],
                    "accuracy": row["accuracy"],
                    "over_triage_rate": row["over_triage_rate"],
                    "under_triage_rate": row["under_triage_rate"],
                })
    return pd.DataFrame.from_records(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Short names of models to run (default: all). "
             "Options: " + ", ".join(LLM_CONFIGS.keys())
    )
    args = parser.parse_args()

    if args.models is None:
        models_to_run = list(LLM_CONFIGS.keys())
    else:
        if "all" in args.models:
            models_to_run = list(LLM_CONFIGS.keys())
        else:
            models_to_run = args.models

    print(f"Models to run: {models_to_run}")

    model_dfs: Dict[str, pd.DataFrame] = {}

    # 1. Run or load each model
    for model_name in models_to_run:
        df = load_or_run_model(model_name)
        model_dfs[model_name] = df

        # Print bias summary for each model
        bias_summary = summarize_bias(df, GROUP_COLS)
        print_bias_summary(bias_summary, model_name=model_name)

        # Per-model charts
        charts_dir = RESULTS_DIR / "charts"
        df_flags = add_error_flags(df)
        for col in GROUP_COLS:
            out_path = charts_dir / f"{model_name}_accuracy_by_{col}.png"
            plot_group_accuracy_bar(
                df_flags,
                group_col=col,
                model_name=model_name,
                out_path=out_path,
            )

    # 2. Combined metrics across models
    combined_metrics = build_combined_metrics(model_dfs, GROUP_COLS)
    combined_path = RESULTS_DIR / "combined_metrics.csv"
    combined_metrics.to_csv(combined_path, index=False)
    print(f"\nSaved combined metrics to {combined_path}")

    # 3. Cross-model comparison charts
    charts_dir = RESULTS_DIR / "charts"
    for col in GROUP_COLS:
        subset = combined_metrics[combined_metrics["group_col"] == col]
        out_path = charts_dir / f"models_comparison_accuracy_by_{col}.png"
        plot_model_comparison_bar(subset[["model", "group", "accuracy"]], col, out_path)

if __name__ == "__main__":
    main()
