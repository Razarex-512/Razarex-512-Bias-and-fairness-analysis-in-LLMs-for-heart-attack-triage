from pathlib import Path
import pandas as pd

from . import config
from .llm_runner import run_llm_on_vignettes
from .metrics import summarize_bias, print_bias_summary
from .sdweat import simple_sdweat_example


def main():
    # 1. Run LLM on all vignette variants (or load cached results)
    outputs_path = config.RESULTS_DIR / "llm_outputs.csv"
    if outputs_path.exists():
        print(f"Loading existing outputs from {outputs_path}")
        df = pd.read_csv(outputs_path)
    else:
        df = run_llm_on_vignettes(max_new_tokens=256, num_samples=1)

    # 2. Compute basic group fairness metrics
    bias_summary = summarize_bias(
        df,
        group_cols=["gender", "ethnicity", "english_level", "age"]
    )
    print_bias_summary(bias_summary)

    # 3. Run a simple SD-WEAT-style embedding analysis
    simple_sdweat_example()


if __name__ == "__main__":
    main()
