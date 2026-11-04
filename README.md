# 🩺 Diabetes Risk Screening Tool

A machine-learning screening tool that flags people who should get a confirmatory diabetes test, based on 8 simple clinic measurements. Built for the NextGen Institute ML final project (Idea #11) by **Abdelrhman Sherif Mekkawy**.

**Live app:** _add your Streamlit Community Cloud / Hugging Face link here after deploying_

![evaluation](images/evaluation.png)

## Problem
Clinics screening many people want to catch as many diabetic cases as possible without testing everyone. Missing a sick person costs more than an extra lab test, so the model is tuned for **recall** (and checked with precision and ROC-AUC), not accuracy.

## Data
Pima Indians Diabetes Database (NIDDK / UCI): 768 women aged 21+, 8 features, 34.9% diabetic. Copy used: [plotly/datasets](https://github.com/plotly/datasets/blob/master/diabetes.csv). About 49% of `Insulin` and 30% of `SkinThickness` values are impossible zeros (missing), handled inside the pipeline.

## Results (untouched test set, n = 154)

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always "no diabetes") | 0.649 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression, threshold 0.50 | 0.727 | 0.685 | 0.597 | 0.638 | 0.816 |
| **Logistic Regression, tuned threshold 0.43** | 0.727 | **0.815** | 0.579 | 0.677 | 0.816 |

Three tuned algorithms were compared (Logistic Regression, Random Forest, Gradient Boosting) with 5-fold cross-validation; Logistic Regression had the best CV ROC-AUC (0.841). The threshold was chosen on cross-validated training predictions only. The test set is small, so expect a few points of noise.

## Project layout
```
app.py                          Streamlit app (English + Arabic toggle)
diabetes_risk_screening.ipynb   Full analysis (Restart & Run All)
utils.py                        Shared preprocessing helper
data/diabetes.csv               Dataset (small, 23 KB)
models/                         Saved pipeline (joblib) + metrics
images/                         Charts used in this README
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py        # language switch (English / العربية) is at the top of the page
```
Use the same scikit-learn version as in `requirements.txt` so the saved pipeline loads correctly.

## Deploy
Push this folder to GitHub, then on [share.streamlit.io](https://share.streamlit.io) create an app pointing to `app.py`.

## Limitations
Small, single-population dataset; not validated for other groups. Educational project, **not a medical device**.
