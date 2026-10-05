# Explainable Heart Disease Classification Dashboard

An interactive machine learning dashboard that classifies historical heart disease dataset records and explains which features contributed to each prediction.

Built with **Python, scikit-learn, and Streamlit**.

> Educational portfolio project. This application is not a diagnosis tool or a validated estimate of future disease risk.

## Features

- Interactive form for 13 dataset features.
- Predictions using a trained Logistic Regression model.
- Model probability displayed alongside the predicted label.
- Individual explanations using feature contributions to log-odds.
- Readable feature labels and contribution charts.
- Support for unknown vessel-count and thal-test values through saved imputation.
- Dashboard displaying held-out evaluation metrics.

## Dataset

This project uses the **Cleveland subset of the UCI Heart Disease dataset**:

- **303 records**
- **13 input features**
- **164 records** with label absent
- **139 records** with label present

The original target is converted into a binary label:

| Label | Meaning |
|---|---|
| 0 | Original dataset target equals 0 |
| 1 | Original dataset target is greater than 0 |

The model classifies the recorded dataset label; it does not predict when someone will develop heart disease.

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

## Machine Learning Workflow

1. Fetch the dataset directly using `ucimlrepo`.
2. Inspect missing values, class counts, and duplicate records.
3. Create a stratified training/test split.
4. Build preprocessing pipelines.
5. Compare Logistic Regression and Random Forest using five-fold cross-validation on the training set.
6. Select Logistic Regression.
7. Evaluate the selected model on the held-out test set.
8. Export the fitted pipeline and metadata.
9. Load the artifacts in Streamlit for predictions and explanations.

### Preprocessing

**Numeric features**
- Median imputation.
- Standard scaling.

**Categorical features**
- Most-frequent-value imputation.
- One-hot encoding.

Preprocessing is fitted inside the model pipeline during cross-validation to prevent leakage between training and validation folds.

## Model Comparison

Five-fold cross-validation results on the training set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.847 | 0.878 | 0.784 | 0.825 | 0.906 |
| Random Forest | 0.826 | 0.834 | 0.783 | 0.804 | 0.900 |

Logistic Regression was selected for its slightly higher mean ROC-AUC and straightforward explanations. The small difference does not establish statistical superiority.

## Held-Out Test Results

The stratified split used **242 training records** and **61 test records**, with `random_state=42`.

| Metric | Score |
|---|---:|
| Accuracy | 86.9% |
| Precision | 81.2% |
| Recall | 92.9% |
| F1 | 0.867 |
| ROC-AUC | 0.958 |

### Confusion Matrix

| Actual / Predicted | Label absent | Label present |
|---|---:|---:|
| Label absent | 27 | 6 |
| Label present | 2 | 26 |

These results describe performance on a small held-out sample from this dataset. They do not establish clinical reliability or generalization to other populations.

## How the Explanations Work

For Logistic Regression:

```text
log-odds = intercept + sum(prepared feature × coefficient)
```

The dashboard calculates each prepared feature's contribution and displays the ten largest contributions by absolute magnitude.

- Positive contributions push the model score toward label 1.
- Negative contributions push the model score toward label 0.
- The full calculation includes all prepared features and the intercept.

These are exact coefficient contributions to the model's log-odds, not SHAP values or explanations of medical causation.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Matplotlib
- joblib
- Google Colab for training

## Project Structure

```text
heart-disease-dashboard/
├── app.py
├── requirements.txt
├── .gitignore
└── models/
    ├── heart_pipeline.joblib
    └── metadata.json
```

## Run Locally

### 1. Clone the repository

```powershell
git clone https://github.com/sanznyk/heart-disease-dashboard.git
cd heart-disease-dashboard
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The exported model requires **scikit-learn 1.6.1**. The application checks the installed version against its saved metadata.

### 4. Start the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

### 5. Explore a prediction

Enter an example record and select **Classify and explain**.

Use **Unknown** for vessel-count or thal-test values when needed. The saved preprocessing pipeline will impute them using training-data statistics.

Only load model files from trusted sources.

## Limitations

- The dataset contains only 303 historical records.
- Evaluation uses a single held-out split of 61 records.
- Some inputs require clinical test results.
- Imputed inputs introduce uncertainty.
- Model probabilities have not been clinically calibrated.
- Feature contributions describe model associations, not causal effects.
- The application has not been validated for clinical use.

## What This Project Demonstrates

- Exploratory data analysis.
- Missing-value handling.
- Categorical encoding and numeric scaling.
- Leakage-aware preprocessing pipelines.
- Stratified splitting and cross-validation.
- Model comparison and classification metrics.
- Individual prediction explanations.
- Model serialization and an interactive Streamlit interface.

## Dataset Attribution

Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989).
*Heart Disease*. UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C52P4X

Dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
