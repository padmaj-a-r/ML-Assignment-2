import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# ---------- Load data ----------
df = pd.read_csv("spambase.csv", header=None)
try:
    float(df.iloc[0, 0])          # first cell is a number -> no header row
    df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
except ValueError:                # first cell is text -> file has a header
    df = pd.read_csv("spambase.csv")
    df = df.rename(columns={df.columns[-1]: "target"})

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Class counts:\n", df["target"].value_counts())

X = df.drop(columns="target")
y = df["target"]

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- Models ----------
models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=2000))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
    "KNN": Pipeline([("scale", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": Pipeline([("clf", GaussianNB())]),
    "Support Vector Machine": Pipeline([("scale", StandardScaler()),
                                        ("clf", SVC(probability=True, random_state=42))]),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(model, f"model/{name.replace(' ', '_').lower()}.joblib")

results = pd.DataFrame(rows).round(4)
print("\n", results.to_string(index=False))
results.to_csv("results.csv", index=False)

# ---------- Save test data (features + target) ----------
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)
print("\nSaved models to model/ and test data to test_data.csv")