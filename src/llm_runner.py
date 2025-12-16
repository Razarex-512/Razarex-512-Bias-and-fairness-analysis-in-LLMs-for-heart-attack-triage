import re
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .config import (
    PRIMARY_LLM_MODEL,
    DEVICE,
    TRIAGE_LABELS,
    RESULTS_DIR,
)
from .vignettes import generate_all_variants

def load_llm_pipeline():
    """
    Load a lightweight chat LLM for text generation.
    """
    print(f"Loading LLM model: {PRIMARY_LLM_MODEL} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(PRIMARY_LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PRIMARY_LLM_MODEL,
        torch_dtype=None,  # use default
        device_map="auto" if DEVICE == "cuda" else None
    )
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1
    )
    return text_gen

def parse_triage_from_output(output_text: str) -> str:
    """
    Extract triage category from model output.
    Expects a line like 'Triage: Emergency'.
    Falls back to keyword search if needed.
    """
    # Try explicit "Triage:" line
    triage_pattern = r"Triage:\s*(.+)"
    match = re.search(triage_pattern, output_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        # Standardize to known labels
        for lbl in TRIAGE_LABELS:
            if lbl.lower() in candidate.lower():
                return lbl

    # Fallback: look for labels anywhere
    for lbl in TRIAGE_LABELS:
        if re.search(lbl, output_text, re.IGNORECASE):
            return lbl

    return "Unknown"

def run_llm_on_vignettes(max_new_tokens: int = 256, num_samples: int = 1) -> pd.DataFrame:
    """
    Run the LLM on all generated vignette variants and collect outputs.
    """
    variants = generate_all_variants()
    text_gen = load_llm_pipeline()

    records: List[Dict] = []

    for var in tqdm(variants, desc="Running LLM on vignettes"):
        for sample_idx in range(num_samples):
            gen = text_gen(
                var["prompt"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )[0]["generated_text"]

            # For some models, generated_text includes the prompt at the start.
            # Keep only the completion by removing the original prompt prefix if present.
            completion = gen.replace(var["prompt"], "").strip()

            triage_pred = parse_triage_from_output(completion)

            records.append({
                "base_id": var["base_id"],
                "variant_id": var["variant_id"],
                "gender": var["gender"],
                "age": var["age"],
                "ethnicity": var["ethnicity"],
                "english_level": var["english_level"],
                "prompt": var["prompt"],
                "raw_output": completion,
                "triage_pred": triage_pred,
                "correct_triage": var["correct_triage"],
                "correct_diagnosis": var["correct_diagnosis"],
                "sample_idx": sample_idx,
            })

    df = pd.DataFrame.from_records(records)
    out_path = RESULTS_DIR / "llm_outputs.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved LLM outputs to {out_path}")
    return df
