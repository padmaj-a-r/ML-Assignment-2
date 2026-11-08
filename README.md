# Spam Email Classification: ML Assignment 2

## a. Problem Statement

The goal is to build and compare five machine learning classification models that decide whether an email is **spam** or **not spam (ham)**, based on the content statistics of the email (word frequencies, character frequencies and capital-letter run lengths). The models are evaluated with six metrics, and the best one for this dataset is identified. The trained models are deployed in an interactive Streamlit web app.

## b. Dataset Description

- **Dataset:** Spambase
- **Source:** UCI Machine Learning Repository (Hopkins et al., Hewlett-Packard Labs)
- **Task:** Binary classification (1 = spam, 0 = not spam)
- **Instances:** 4,601 emails
- **Features:** 57 numeric input features, plus 1 target column (shape: 4601 x 58)
  - 48 features: percentage of words in the email matching a given word (e.g. "free", "money")
  - 6 features: percentage of characters matching a given character (e.g. `!`, `$`)
  - 3 features: capital-letter run lengths (average, longest, total)
- **Missing values:** None
- **Class distribution:** 2,788 not spam (60.6%) and 1,813 spam (39.4%)
- **Train/test split:** 80% / 20% (3,680 training rows, 921 test rows), stratified, `random_state=42`
- **Preprocessing:** `StandardScaler` for Logistic Regression, KNN and SVM (distance- and gradient-based models). Decision Tree and Naive Bayes use the raw features. Each scaler is saved inside its model pipeline, so the app applies the same scaling automatically.

## c. GitHub Repository Link

https://github.com/YOUR-USERNAME/ml-assignment-2

**Live Streamlit app:** https://YOUR-APP-NAME.streamlit.app

### Repository structure

```
ml-assignment-2/
|-- app.py              (Streamlit app)
|-- train.py            (trains the 5 models, computes metrics, saves models)
|-- requirements.txt
|-- README.md
|-- test_data.csv       (held-out test data: features + target)
|-- results.csv         (metrics table)
|-- spambase.csv        (original dataset)
|-- model/              (5 saved models: .joblib)
```

## d. Models Used

All five models were trained on the same training split and evaluated on the same test split.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9078 | 0.8828 | 0.8926 | 0.8877 | 0.8140 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest accuracy, AUC, F1 and MCC. Spam is largely linearly separable once the features are scaled, so this simple model is hard to beat. |
| Decision Tree | Good accuracy (0.9110) but the lowest AUC (0.9078). A single tree gives coarse probability estimates and tends to overfit, so it ranks emails less reliably. |
| KNN | Slightly lower accuracy than the tree, but a clearly better AUC (0.9506). It is sensitive to scaling and to the 57-dimensional space, which limits precision and recall. |
| Naive Bayes | Weakest overall (accuracy 0.8339, MCC 0.6941) but the highest recall (0.9532), so it catches the most spam. Low precision (0.7178) means many genuine emails are flagged as spam. The feature-independence assumption does not hold well for correlated word frequencies. |
| Support Vector Machine | Very close to Logistic Regression, with the highest precision (0.9277). Strong, but slower to train and no better than the simpler linear model here. |
| **Overall winner** | **Logistic Regression.** It leads on accuracy, AUC, F1 and MCC, and is also the simplest and fastest model. SVM is a close second. |

## How to Run Locally

```
pip install -r requirements.txt
python train.py          # trains models and saves them in model/
streamlit run app.py     # launches the web app
```

## Streamlit App Features

- Upload a test CSV (features + `target` column)
- Dropdown to select any of the 5 trained models
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Side-by-side metrics for all models on the uploaded data
- Models are loaded from disk and **not retrained** on the uploaded data