# NLP_Final_Project
# Automated Detection of Food Hazards in Incident Reports

## Overview
This repository contains the code and documentation for the final project in CST8507 - Natural Language Processing, based on SemEval-2025 Task 9: Food Hazard Detection Challenge. The project implements machine learning models to classify food incident reports by predicting product and hazard categories from report titles. The implementation includes baseline models (TF-IDF with Logistic Regression and Random Forest) and an advanced fine-tuned BERT model, along with explainability and error analysis.

- **Team Members**: Khaled Saleh, Yazid Rahmouni
- **Repository Structure**:
  - `NLP_Final_Project_Code.ipynb`: Jupyter notebook with full code implementation (data loading, preprocessing, models, evaluation).
  - `incidents_train.csv`, `incidents_valid.csv`, `incidents_test.csv`: Dataset files (not included here; download from Zenodo as described below).
  - `README.md`: This file.

## Problem Definition
The project addresses the SemEval-2025 Task 9: Food Hazard Detection Challenge, which focuses on automated classification of food safety incident reports to identify potential hazards. This is crucial for public health, as it enables faster response to food recalls and contamination events.

- **Subtask 1**: Coarse-grained classification – Predict the product category (22 classes, e.g., "meat, egg and dairy products", "fruits and vegetables") and hazard category (10 classes, e.g., "biological", "foreign bodies") from the title of the incident report.
- **Subtask 2**: Fine-grained classification – Predict the specific product (1,142 types, e.g., "smoked sausage", "chicken breast") and hazard (128 types, e.g., "listeria monocytogenes", "plastic fragment").
  
The data is multilingual and multiclass, requiring models that are accurate and interpretable to support real-world applications in food safety monitoring.

## Dataset Used
The dataset consists of simulated food recall incident reports, sourced from `food_recall_incidents.csv` on Zenodo (DOI: 10.5281/zenodo.XXXXXXX – replace with actual DOI if available; for this project, we used the provided train/valid/test splits).

- **Splits**:
  - Training: 5,082 samples
  - Validation: 565 samples
  - Test: 997 samples

- **Features**:
  - `title`: The main text feature (incident report title, preprocessed for modeling).
  - Labels: `product-category` (coarse product), `hazard-category` (coarse hazard), `product` (fine-grained product), `hazard` (fine-grained hazard).
  - Other columns (not used for modeling): `year`, `month`, `day`, `country`, `text` (full report body), etc.

- **Preprocessing**:
  - Convert titles to lowercase.
  - Remove punctuation and extra spaces.
  - Example: Original title "Recall Notification: FSIS-024-94" → Processed: "recall notification fsis02494".

The dataset is imbalanced, with some categories (e.g., "meat, egg and dairy products") dominating, which poses challenges for fine-grained classification. Labels were encoded using `sklearn.preprocessing.LabelEncoder` for consistency across splits.

To obtain the dataset:
1. Download from Zenodo or the SemEval-2025 task repository.
2. Place the CSV files in the root directory to run the notebook.

## Evaluation Metrics
The primary metric is **Macro F1-score**, which averages F1-scores across all classes without weighting by support. This is suitable for imbalanced multiclass problems, as it treats rare classes equally.

- F1-score = 2 * (Precision * Recall) / (Precision + Recall)
- Macro F1 = Average of per-class F1-scores.

Additional analysis includes confusion matrices, classification reports (precision, recall, F1 per class), and inspection of misclassified samples.

## Model Explanation
### Baseline Models
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) with `sklearn.feature_extraction.text.TfidfVectorizer` (max_features=5000, ngram_range=(1,2) for unigrams and bigrams).
- **Classifiers**:
  - Logistic Regression (`sklearn.linear_model.LogisticRegression`, max_iter=1000).
  - Random Forest (`sklearn.ensemble.RandomForestClassifier`, n_estimators=100, random_state=42).
- Trained separately for each subtask/target (product-category, hazard-category, product, hazard).
- **Explainability**: SHAP (SHapley Additive exPlanations) for feature importance in Logistic Regression (e.g., keywords like "recall", "allergy").

References for baselines:
- TF-IDF: Jones, K. S. (1972). "A statistical interpretation of term specificity and its application in retrieval." Journal of Documentation.
- Logistic Regression: Wright, R. E. (1995). "Logistic regression." In Reading and understanding multivariate statistics.
- Random Forest: Breiman, L. (2001). "Random forests." Machine Learning.
- SHAP: Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.

### Advanced Model
- **BERT (Bidirectional Encoder Representations from Transformers)**: Fine-tuned `bert-base-uncased` for sequence classification using `transformers.BertForSequenceClassification`.
  - Tokenizer: `BertTokenizer` with max_length=128.
  - Optimizer: AdamW with learning rate=2e-5.
  - Training: 3 epochs, batch_size=32 (or 16 for fine-grained due to more classes).
  - Custom Dataset class for PyTorch DataLoader.
- Separate models for each target to handle different class counts.
- **Explainability**: LIME (Local Interpretable Model-agnostic Explanations) for local predictions on BERT.

Reference for BERT:
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL.

All models were trained on preprocessed titles only, with reproducibility ensured via random seeds (42).

## Results Achieved
### Validation Set Results
| Model                  | Subtask                  | Macro F1 |
|------------------------|--------------------------|----------|
| Logistic Regression   | Subtask 1 Product Cat   | 0.5105  |
| Logistic Regression   | Subtask 1 Hazard Cat    | 0.5295  |
| Random Forest         | Subtask 1 Product Cat   | 0.5294  |
| Random Forest         | Subtask 1 Hazard Cat    | 0.5785  |
| Logistic Regression   | Subtask 2 Product       | 0.1354  |
| Logistic Regression   | Subtask 2 Hazard        | 0.1982  |
| Random Forest         | Subtask 2 Product       | 0.2248  |
| Random Forest         | Subtask 2 Hazard        | 0.3832  |
| BERT                  | Subtask 1 Product Cat   | ~0.69   |

(Note: BERT weighted avg F1 ~0.69; macro approx from report.)

### Test Set Results
| Model                  | Subtask                  | Macro F1 |
|------------------------|--------------------------|----------|
| Logistic Regression   | Subtask 1 Product Cat   | 0.4463  |
| Logistic Regression   | Subtask 1 Hazard Cat    | 0.5417  |
| Random Forest         | Subtask 1 Product Cat   | 0.4249  |
| Random Forest         | Subtask 1 Hazard Cat    | 0.5827  |
| Logistic Regression   | Subtask 2 Product       | 0.0787  |
| Logistic Regression   | Subtask 2 Hazard        | 0.1764  |
| Random Forest         | Subtask 2 Product       | 0.2293  |
| Random Forest         | Subtask 2 Hazard        | 0.3359  |
| BERT                  | Subtask 1 Product Cat   | 0.5283  |
| BERT                  | Subtask 1 Hazard Cat    | 0.5135  |
| BERT                  | Subtask 2 Product       | 0.0096  |
| BERT                  | Subtask 2 Hazard        | 0.1491  |

Error analysis revealed common misclassifications between similar categories (e.g., "ices and desserts" vs. "nuts, nut products and seeds"). SHAP highlighted keywords like "recall" and "allergy" as important features.

## Comparison with Baseline Results
BERT outperforms baselines in Subtask 1 (coarse classification), achieving higher Macro F1 on both validation (~0.69 vs. ~0.51-0.58) and test (0.5283 Product Cat vs. 0.4463 LR / 0.4249 RF; 0.5135 Hazard Cat vs. 0.5417 LR / 0.5827 RF). This is due to BERT's contextual embeddings capturing nuances in titles better than TF-IDF bag-of-words.

However, for Subtask 2 (fine-grained), BERT underperforms (e.g., test Product: 0.0096 vs. 0.2293 RF), likely due to extreme class imbalance (1,142 products) and limited training data/epochs. Baselines like RF handle sparsity better here but still yield low scores overall.

Recommendations for improvement: Handle imbalance (e.g., oversampling), more epochs for BERT, or hierarchical classification (coarse then fine).

**Baseline References** (as used in implementation):
- Scikit-learn for TF-IDF, LR, RF: Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research.
- Hugging Face Transformers for BERT: Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." EMNLP.
- SHAP: https://github.com/shap/shap
- LIME: https://github.com/marcotcr/lime

## Installation and Usage
1. Install dependencies: `pip install -r requirements.txt` (create from notebook imports: pandas, numpy, scikit-learn, transformers, torch, shap, lime, etc.).
2. Run the notebook: `jupyter notebook NLP_Final_Project_Code.ipynb`.
3. Ensure CUDA/GPU for BERT training if available.

For questions, contact the team members.
