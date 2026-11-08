# Spam Email Classification: ML Assignment 2

## a. Problem Statement

The goal is to build and compare five machine learning classification models that are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbour
4. Naice Bayes
5. Support Vector Machines

and then decide whether an email is **spam** or **not spam (ham)**, based on the contents of the email (word repetation frequency, character repetation frequencies and capital-letter lengths). 

The models are evaluated with six metrics they are

1. Accuracy
2. AUC
3. Precision
4. Recall
5. F1 Score
6. MCC

 After getting this analysis the best one for this dataset is identified. The trained models are deployed in the interactive Streamlit web app. 

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
- **Preprocessing:** `StandardScaler` for Logistic Regression 
KNN and SVM (distance- and gradient-based models). 
Decision Tree and Naive Bayes use the raw features. 
Each scaler is saved inside its model pipeline, so the app applies the same scaling automatically.

## c. GitHub Repository Link

https://github.com/padmaj-a-r/ML-Assignment-2

**Live Streamlit app:** https://inbox-guard-spam-lab-ml.streamlit.app/

### Repository structure

```
ML-Assignment-2/
|-- model/
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- logistic_regression.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
|-- .gitignore
|-- app.py                (Streamlit app)
|-- README.md
|-- requirements.txt
|-- results.csv           (metrics table)
|-- spambase.csv          (original dataset)
|-- test_data.csv         (held-out test data: features + target)
|-- train.py              (trains the 5 models, computes metrics, saves models)
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
| Logistic Regression | Logistic Regression Came out on top for this dataset, with 0.9294 as the accuracy, 0.9702 AUC and 0.8518 MCC. After standardizing the given features, a straight-line boundary is observed that separates spam from genuine mail accurately, so the additional complexity of the other models brought no significant gain here. |
| Decision Tree | Decision Tree Reached 0.9110 accuracy, which is okay, but its AUC of 0.9078 was the lowest of all five models. Because a tree outputs the probabilities (each leaf gives a nearly fixed value), it is less able to rank emails from "likely spam" to "unlikely spam", and it can overfit the training data. Hence this is not considered as the best model |
| KNN | KNN gave the Accuracy of 0.9077 which was the second lowest, yet its AUC of 0.9506 was clearly higher than the decision tree's. It depends on distances between emails, hence it needs scaling and it struggles a little in a space with 57 features, which held precision (0.8861) and recall (0.8788) less. |
| Naive Bayes | Naive Bayes is The weakest model overall (accuracy 0.8339, MCC 0.6941), but it had the highest recall at 0.9532, so it let the fewest spam emails through which is good. The cost was a precision of only 0.7178, meaning many genuine emails were wrongly flagged which is not the quality of a good model for this scenario. It assumes the features are independent, and also the word frequencies in emails are clearly correlated. |
| Support Vector Machine | SVM is Nearly tied along with Logistic Regression (accuracy 0.9273, AUC 0.9667, MCC 0.8471) and it had the best precision, at 0.9277. Anyhow, it takes longer to train the model and gave no real improvement over the simpler linear model on this data. |
| **Overall winner** | **Logistic Regression.** This Ranks first on accuracy, AUC, F1 and MCC, and it is also the simplest and quickest to train. SVM is a close but a runner-up, and Naive Bayes is the right pick only if catching every spam email matters more than avoiding false alarms something like infectious disease screening. |

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