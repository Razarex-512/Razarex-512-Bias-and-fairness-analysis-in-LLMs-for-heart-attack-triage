# Evaluating Bias in Lightweight Language Models for Chest Pain Triage

This repository contains the code and experimental setup used to evaluate fairness in lightweight open-source language models applied to a clinical triage task. The project focuses on chest pain urgency classification and examines whether identical clinical cases receive different triage decisions when only demographic attributes are varied.

The work was completed as part of a course project for CPSC 571/671 and emphasizes reproducible, local experimentation using open-source tools.

---

## Project Overview

Lightweight language models are increasingly accessible and can run on consumer hardware, making them attractive for decision-support applications. However, their behavior across different patient demographics is not well understood.

This project evaluates three lightweight models:
- Phi-2
- StableLM-3B
- TinyLlama

Using a counterfactual vignette-based design, the study measures accuracy, subgroup performance differences, and error patterns such as over-triage and under-triage across demographic groups.

---

## Research Question

Do identical clinical cases receive different triage decisions when only patient demographic attributes change?

---

## Methodology

The evaluation follows a task-based fairness approach using controlled clinical vignettes.

Key methods include:
- Counterfactual fairness testing
- Task-based triage evaluation
- Subgroup performance analysis
- Over-triage and under-triage error analysis

Each vignette describes the same chest pain scenario, while one demographic attribute is varied at a time:
- Gender
- Age
- Ethnicity
- English proficiency

All models receive identical prompts and inputs.

---

## Experimental Setup

- Task: Chest pain urgency classification
- Labels: Emergency, Urgent, Non-urgent, Self-care
- Number of vignettes: 96
- Execution: Local, consumer-grade hardware
- Prompting: Fixed prompt structure across all models
- Output: Logged and stored for analysis

---

## Tools and Technologies

**Programming Language**
- Python

**Libraries and Frameworks**
- Hugging Face Transformers
- PyTorch
- Standard Python data processing libraries

**Models**
- Phi-2
- StableLM-3B
- TinyLlama

All tools and models used are open-source and free to run locally.

---

## Results

The experiments show that:
- All evaluated models exhibit some level of demographic disparity
- Ethnicity produces the largest performance gaps across models
- Higher overall accuracy does not guarantee fairer behavior
- Smaller models tend to show greater variability and instability

Results are visualized using subgroup accuracy charts and model comparison plots.

---

## Reproducibility

The project is designed to be reproducible:
- Fixed prompt templates
- Identical inputs across models
- Logged outputs saved as CSV files
- Local execution without reliance on external APIs

---

## Limitations and Future Work

- The vignette set is limited in size and scope
- Only one clinical symptom was evaluated
- No bias mitigation strategies were applied

Future work could include expanding clinical scenarios, testing additional models, and exploring mitigation techniques such as prompt refinement or post-processing adjustments.

---

## Disclaimer

This project is for academic research purposes only and is not intended for clinical use. The models evaluated here should not be used for real medical decision-making without rigorous validation and oversight.
