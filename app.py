import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="InboxGuard | Spam Model Lab", page_icon="🛡️", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    padding: 28px 32px; border-radius: 16px; margin-bottom: 18px;
}
.hero h1 { color: #ffffff; margin: 0; font-size: 2.2rem; }
.hero p  { color: #bcd6e0; margin: 6px 0 0 0; font-size: 1.05rem; }
.card {
    border-radius: 14px; padding: 16px 12px; text-align: center;
    color: #ffffff; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.card .label { font-size: 0.85rem; letter-spacing: 1px; opacity: 0.9; }
.card .value { font-size: 1.9rem; font-weight: 700; }
.badge {
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    background: #f9a826; color: #1b1b1b; font-weight: 700; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛡️ InboxGuard: Spam Model Lab</h1>
  <p>Five pre-trained classifiers, one spam dataset. Upload test data and see which model guards the inbox best.</p>
</div>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
CARD_COLORS = {
    "Accuracy": "#1e88e5", "AUC": "#8e24aa", "Precision": "#00897b",
    "Recall": "#f4511e", "F1": "#3949ab", "MCC": "#6d4c41",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Control Panel")
    uploaded = st.file_uploader("1. Upload test data (CSV)", type="csv")
    choice = st.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
    st.caption("The CSV needs the 57 feature columns plus a `target` column "
               "(1 = spam, 0 = not spam). Models are loaded from disk and never retrained.")

if uploaded is None:
    st.info("👈 Start by uploading `test_data.csv` in the control panel.")
    st.stop()

data = pd.read_csv(uploaded)
if "target" not in data.columns:
    st.error("The CSV must contain a column named 'target'.")
    st.stop()

y = data["target"]
model = load_model(MODEL_FILES[choice])
missing = [c for c in model.feature_names_in_ if c not in data.columns]
if missing:
    st.error(f"Missing feature columns, for example: {missing[:3]}")
    st.stop()

pred, metrics = evaluate(model, data[list(model.feature_names_in_)], y)

# ---------- Dataset summary ----------
c1, c2, c3 = st.columns(3)
c1.metric("Emails uploaded", f"{len(data):,}")
c2.metric("Spam emails", f"{int((y == 1).sum()):,}")
c3.metric("Not spam", f"{int((y == 0).sum()):,}")

st.subheader(f"Results for: {choice}")
cols = st.columns(6)
for col, (name, val) in zip(cols, metrics.items()):
    col.markdown(
        f'<div class="card" style="background:{CARD_COLORS[name]}">'
        f'<div class="label">{name.upper()}</div>'
        f'<div class="value">{val:.4f}</div></div>',
        unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🧩 Confusion Matrix", "📋 Classification Report", "🏆 Compare All Models", "🔍 Data Preview"])

with tab1:
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=["Not spam", "Spam"], yticklabels=["Not spam", "Spam"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    left, right = st.columns([1, 1])
    left.pyplot(fig)
    tn, fp, fn, tp = cm.ravel()
    right.markdown(f"""
**How to read this**
- ✅ **{tp}** spam emails correctly caught
- ✅ **{tn}** genuine emails correctly let through
- ⚠️ **{fp}** genuine emails wrongly flagged as spam
- ❌ **{fn}** spam emails that slipped through
""")

with tab2:
    report = classification_report(y, pred, target_names=["Not spam", "Spam"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

with tab3:
    rows = []
    for name, path in MODEL_FILES.items():
        m = load_model(path)
        _, mt = evaluate(m, data[list(m.feature_names_in_)], y)
        rows.append({"Model": name, **mt})
    comp = pd.DataFrame(rows).set_index("Model").round(4)
    best = comp["MCC"].idxmax()
    st.markdown(f'<span class="badge">🏆 Best on this data (by MCC): {best}</span>',
                unsafe_allow_html=True)
    st.dataframe(comp.style.highlight_max(axis=0, color="#2e7d32"))
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])

with tab4:
    st.write(f"{data.shape[0]} rows, {data.shape[1]} columns")
    st.dataframe(data.head(20))