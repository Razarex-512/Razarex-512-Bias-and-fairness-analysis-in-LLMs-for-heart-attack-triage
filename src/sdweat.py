from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from .config import BIO_ENCODER_MODEL, DEVICE

class EncoderWrapper:
    def __init__(self, model_name: str):
        print(f"Loading encoder: {model_name} on {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        outputs = self.model(**encoded)
        # Use [CLS] token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu().numpy()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def average_association(
    encoder: EncoderWrapper,
    targets: List[str],
    attributes: List[str]
) -> float:
    target_embs = encoder.encode(targets)
    attr_embs = encoder.encode(attributes)

    sims = []
    for t in target_embs:
        for a in attr_embs:
            sims.append(cosine_similarity(t, a))
    return float(np.mean(sims))

def simple_sdweat_example() -> Dict[str, float]:
    """
    Very small example of comparing association of:
    - female terms vs 'heart attack' terms
    - male terms vs 'heart attack' terms
    This is just to demonstrate the pipeline.
    """
    encoder = EncoderWrapper(BIO_ENCODER_MODEL)

    female_terms = ["woman", "female", "she", "mother"]
    male_terms = ["man", "male", "he", "father"]

    heart_attack_terms = [
        "heart attack", "chest pain", "myocardial infarction", "ischemia"
    ]

    female_assoc = average_association(encoder, female_terms, heart_attack_terms)
    male_assoc = average_association(encoder, male_terms, heart_attack_terms)

    result = {
        "female_assoc": female_assoc,
        "male_assoc": male_assoc,
        "difference": male_assoc - female_assoc,
    }

    print("\n=== Simple SD-WEAT-style result (Bio_ClinicalBERT) ===")
    print(result)
    return result
