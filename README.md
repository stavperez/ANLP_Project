**Baseline Model Notebook**
This notebook provides the baseline evaluation for the Manager-Worker architecture in structured multi-hop question answering. It performs two main tasks:
1. Compare Baseline Models
Evaluates a pretrained model (zero-shot) versus a fine-tuned model trained on the enriched Break dataset, using Semantic Textual Similarity (STS) to assess performance.
2. Integrate with Manager-Worker Results
Loads the .pkl output file from the Manager-Worker script, adds the baseline model predictions, and analyzes both outputs using STS against the ground truth.

Requirements:
GPU (e.g., Google Colab or local CUDA environment)
Python libraries: transformers, torch, pandas, sentence-transformers

How to Run:
Enable GPU in your runtime environment. Open the notebook Baseline_model.ipynb. Provide the .pkl file from the Manager-Worker script when prompted.
Run all cells in order to:
Load the dataset and models. Evaluate pretrained and fine-tuned baseline models. Add baseline predictions to the Manager-Worker results. Compute STS scores and compare performance

Outputs:
STS comparison between Pretrained baseline and Manager-Worker pipeline
Combined results table with all model predictions and scores

**Manager Model Notebook**
A fine-tuned Flan-T5-small model for question decomposition. It takes a complex question and breaks it into step-by-step sub-questions.

Features:
- Uses the Break dataset (QDMR).
- Converts questions → structured decompositions.
- Fine-tunes and saves models in Google Drive.
- Ready for training + inference in Google Colab.

Usage:
Via Google Colab - GPU, with the needed libraries able to be installed via pip (code already exists for it, if opened via colab)
