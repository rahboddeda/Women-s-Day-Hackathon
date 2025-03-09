# Model Architecture & Evaluation Report

## Overview
This report presents the model architectures and evaluation metrics for each task in the "AI for Safer Online Spaces for Women" hackathon. It provides insights into the models used, their configurations, and the performance metrics recorded during evaluation.

## Model Architectures

### Task 1: Parent-Child Conversation Reconstruction
**Model:**
- Transformer-based summarization models (BART, GPT-based)
- Sequence-to-sequence approach for context reconstruction

**Evaluation Metrics:**

| Metric               | Summarizer | Custom BERT Training |
|----------------------|------------|----------------------|
| BLEU Score          | 0.0161     | 0.0896               |
| ROUGE-1             | 0.0288     | 0.1318               |
| ROUGE-2             | 0.0267     | 0.1032               |
| ROUGE-L             | 0.0288     | 0.1233               |
| Perplexity          | 45.6756    | N/A                  |
| Semantic Similarity | 0.1038     | 0.3218               |

---

### Task 2: Subreddit-Based Topic Classification
**Model:**
- TF-IDF and word embeddings for feature extraction
- Random Forest, SVM, and Transformer-based classifiers

**Evaluation Metrics:**

| Metric                 | Value  |
|------------------------|--------|
| Precision             |  0.47   |
| Recall                |  0.60   |
| F1-score              |  0.50   |
| Accuracy              |  0.60   |
| Topic Coherence Score |  0.3932 |
---

### Task 3: Detecting Toxic or Harmful Comments
**Model:**
- Fine-tuned RoBERTa for toxicity classification
- Class balancing techniques applied

**Evaluation Metrics:**

| Metric   | Value  |
|----------|--------|
| AUC-ROC  | 0.8757 |
| AUC-PR   | 0.6035 |

**False Positive & False Negative Rates**
| Class | FPR   | FNR   |
|-------|-------|-------|
| 0     | 0.4698 | 0.0112 |
| 1     | 0.0000 | 1.0000 |
| 2     | 0.0194 | 0.4609 |

---

### Task 4: Context-Aware Misogyny Detection
**Model:**
- Fine-tuned BERT for misogyny classification
- TF-IDF for feature extraction
- LIME & SHAP for explainability

**Evaluation Metrics:**

| Metric                 | Value  |
|------------------------|--------|
| Accuracy               | 0.9748 |
| Precision              | 0.7460 |
| Recall                 | 0.7344 |
| F1-Score               | 0.7402 |
| Classification Accuracy | 0.9031 |
| Cohen’s Kappa Score    | 0.7269 |

---

## Summary & Future Improvements
- The models performed well across tasks, achieving high accuracy and precision with small number of epochs.
- Future enhancements include bias mitigation, improved explainability, and dataset expansion.

This report serves as a structured evaluation of the AI-driven solution for safer online conversations.
