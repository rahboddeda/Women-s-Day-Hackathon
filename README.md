# AI for Safer Online Spaces for Women

## Overview

This project was developed for the "AI for Safer Online Spaces for Women" hackathon, aimed at leveraging AI and NLP techniques to foster safer digital conversations. Our solution focuses on reconstructing conversations, classifying discussions, and detecting toxicity to create a more inclusive online environment.

## Dataset

Reference Paper: "An Expert-Annotated Dataset for the Detection of Online Misogyny."

Dataset link: https://github.com/ellamguest/online-misogyny-eacl2021

We used a dataset containing Reddit discussions with a parent-child conversational structure. The dataset includes:

- **Text Data**: Online discussions extracted from Reddit.
- **Annotations**: Labels for toxicity, misogyny, and contextual information.
- **Metadata**: Includes subreddit, author, timestamps, and classification labels.

## Tasks and Approach

We implemented solutions for four major tasks:

### Task 1: Parent-Child Conversation Reconstruction

**Objective**: Reconstruct fragmented online discussions to maintain context and improve moderation.

**Approach**:

- Used NLP techniques to analyze conversation flow.
- Implemented sequence-based models to predict missing context.
- Summarized discussions while preserving intent.
- Also trained a custom BERT model for the summarization task.

**Model**:

- Utilized transformer-based models (BART, GPT-based) for summarization.
- Evaluated using BLEU, ROUGE, Perplexity, and Semantic Similarity.

### Task 2: Subreddit-Based Topic Classification

**Objective**: Classify discussions based on their respective subreddits.

**Approach**:

- Preprocessed text data by removing stop words, stemming, and tokenization.
- Implemented TF-IDF and word embeddings for feature extraction.
- Trained classifiers such as Random Forest, SVM, and Transformer-based models for subreddit prediction.

**Evaluation**:

- Measured accuracy, precision, recall, and F1-score.
- Developed an interactive visualization to track topic trends.
- Calculated topic coherence scores to evaluate topic modeling performance.

### Task 3: Detecting Toxic or Harmful Comments

**Objective**: Identify and mitigate toxic and misogynistic content in online discussions.

**Approach**:

- Employed deep learning models for text classification.
- Fine-tuned pre-trained models such as RoBERTa.
- Used AUC-ROC, AUC-PR, and confusion matrix for evaluation.
- Applied class balancing techniques.

**Results**:

- Achieved high classification accuracy on the test set.
- Evaluated model fairness using false positive/negative rates.

### Task 4: Context-Aware Misogyny Detection

**Objective**: Detect misogynistic language while considering context.

**Approach**:

- Fine-tuned **BERT for Sequence Classification**.
- Used **TF-IDF** for textual feature extraction.
- Performed **gendered word bias checks** to analyze potential biases.
- Used **LIME and SHAP** to enhance explainability and identify key words influencing model decisions.

**Evaluation**:

- Accuracy: **0.9748**
- Precision, Recall, F1-score
- Cohen’s Kappa Score: **0.7269**
- Bias analysis with **gendered word checks**
- **Highlighted key words** influencing predictions using LIME and SHAP.

**Potential Use in Monitoring Systems**:

- The highlighted words from LIME and SHAP can be integrated into real-time monitoring systems to flag users for review.
- These flagged words provide **transparent insights** into why a comment was marked as misogynistic.
- Such a system could assist **moderators and automated tools** in identifying and acting upon problematic content while ensuring **explainability and fairness**.

## Model Training & Evaluation

Each model was trained using the following pipeline:

1. **Data Preprocessing**: Tokenization, lowercasing, and removal of stop words.
2. **Feature Extraction**: TF-IDF, word embeddings, and contextual embeddings.
3. **Model Selection**: Experimented with traditional ML and deep learning models.
4. **Hyperparameter Tuning**: Optimized parameters using grid search and Bayesian optimization.
5. **Evaluation Metrics**:
   - Accuracy
   - Precision, Recall, and F1-score
   - Topic Coherence Score (for topic modeling)
   - AUC-ROC, AUC-PR, and confusion matrix (for toxicity detection)
   - Cohen’s Kappa Score (for misogyny detection)

## Model & Evaluation Report

### Model Architecture and Methodology

- **Task 1**: Transformer-based sequence models (BART, GPT-based) for summarization and conversation reconstruction.
- **Task 2**: TF-IDF, word embeddings, and deep learning classifiers for topic classification.
- **Task 3**: Fine-tuned RoBERTa model for toxicity detection, with class balancing techniques applied.
- **Task 4**: Fine-tuned BERT model with bias analysis and explainability techniques (LIME, SHAP).

### Evaluation Metrics

- **Task 1**: BLEU, ROUGE, Perplexity, Semantic Similarity.
- **Task 2**: Accuracy, Precision, Recall, F1-score, Topic Coherence Score.
- **Task 3**: AUC-ROC, AUC-PR, Confusion Matrix.
- **Task 4**: Accuracy, Precision, Recall, F1-score, Cohen’s Kappa Score.

### Challenges Faced and Solutions Implemented

#### **1. Data Imbalance in Toxicity Detection**

- **Challenge**: The dataset contained significantly fewer misogynistic/toxic samples.
- **Solution**: Applied class weighting and oversampling techniques to improve model fairness.

#### **2. Maintaining Context in Conversation Reconstruction**

- **Challenge**: Parent-child relationships in Reddit discussions were sometimes ambiguous.
- **Solution**: Used BART’s sequence-to-sequence modeling to improve contextual consistency.

#### **3. Overfitting in Topic Classification**

- **Challenge**: Some subreddits had very distinct vocabularies, leading to model overfitting.
- **Solution**: Regularized models using dropout, data augmentation, and cross-validation.

#### **4. Bias in Misogyny Detection**

- **Challenge**: Potential bias in classifying gender-related words.
- **Solution**: Implemented **gendered word bias checks** and **explainability techniques (LIME, SHAP)** to ensure fairness.

## How to Run the Code

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required libraries

### Steps

1. Clone the repository:
   ```sh
   git clone <repo_link>
   cd <repo_folder>
   ```
2. Install dependencies:
   ```
   pip install transformers nltk torch datasets pandas numpy matplotlib seaborn scikit-learn gensim plotly wordcloud shap lime tqdm
   ```
3. Run jupyter notebook

   ```sh
    jupyter notebook

   ```

4. Open and execute the following notebooks:
   - Que1.ipynb (Conversation Reconstruction)
   - Que2.ipynb (Topic Classification)
   - Que3.ipynb (Toxicity Detection)
   - Que4.ipynb (Misogyny Detection)

## Future Enhancements

- Integrate bias mitigation techniques to reduce false positives/negatives.

- Improve explainability by enhancing visualization of toxic elements.

- Extend the model to support multilingual text analysis.

## Contributors

- Rahul Boddeda
- Lahari Kethu

## License

This project is licensed under [MIT License].
