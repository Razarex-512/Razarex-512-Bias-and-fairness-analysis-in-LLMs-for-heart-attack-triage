from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model names
PRIMARY_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BIO_ENCODER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DNA_ENCODER_MODEL = "zhihan1996/DNABERT-2-117M"
BASELINE_ENCODER_MODEL = "distilbert-base-uncased"

TRIAGE_LABELS = ["Emergency", "Urgent", "Non-urgent", "Self-care"]

GENDERS = ["male", "female"]
AGES = ["45", "65"]
ETHNICITIES = ["White", "Black", "South Asian", "East Asian"]
ENGLISH_LEVELS = ["native", "limited"]

# LLM configurations for multi-model experiments
LLM_CONFIGS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi2": "microsoft/phi-2",
    "stablelm-3b": "stabilityai/stablelm-3b-4e1t",
}