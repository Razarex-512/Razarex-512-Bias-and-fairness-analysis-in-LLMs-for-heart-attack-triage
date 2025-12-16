import re
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .config import DEVICE, TRIAGE_LABELS, RESULTS_DIR, LLM_CONFIGS
from .vignettes import generate_all_variants

import torch

def parse_triage_from_output(output_text: str) -> str:
    """
    Extract triage category from model output.
    Expects a line like 'Triage: Emergency'.
    Falls back to keyword search if needed.
    """
    triage_pattern = r"Triage:\s*(.+)"
    match = re.search(triage_pattern, output_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        for lbl in TRIAGE_LABELS:
            if lbl.lower() in candidate.lower():
                return lbl

    for lbl in TRIAGE_LABELS:
        if re.search(lbl, output_text, re.IGNORECASE):
            return lbl

    return "Unknown"

def load_llm_pipeline(model_id: str):
    """
    Load a given causal LLM on CPU or GPU.
    When using device_map='auto' (accelerate), we must NOT pass a device
    argument to the pipeline.
    """
    print(f"\nLoading LLM model: {model_id} on {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if DEVICE == "cuda":
        # Let accelerate handle device placement
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # half precision for speed / memory
            device_map="auto",
            trust_remote_code=True,
        )
        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # no device arg here when using accelerate
        )
    else:
        # Pure CPU case
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # force CPU
        )

    return text_gen

def run_llm_for_model(
    model_short_name: str,
    max_new_tokens: int = 256,
    num_samples: int = 1,
) -> pd.DataFrame:
    """
    Run the specified model (by short name from LLM_CONFIGS) on all vignette variants.
    Save results to results/{model_short_name}_outputs.csv.
    """
    if model_short_name not in LLM_CONFIGS:
        raise ValueError(f"Unknown model_short_name '{model_short_name}'. "
                         f"Available: {list(LLM_CONFIGS.keys())}")

    model_id = LLM_CONFIGS[model_short_name]
    variants = generate_all_variants()
    text_gen = load_llm_pipeline(model_id)

    records: List[Dict] = []

    for var in tqdm(variants, desc=f"Running {model_short_name} on vignettes"):
        for sample_idx in range(num_samples):
            gen = text_gen(
                var["prompt"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )[0]["generated_text"]

            completion = gen.replace(var["prompt"], "").strip()
            triage_pred = parse_triage_from_output(completion)

            records.append({
                "model": model_short_name,
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
    out_path = RESULTS_DIR / f"{model_short_name}_outputs.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved outputs for {model_short_name} to {out_path}")
    return df
