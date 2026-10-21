# app.py
import json
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Spam Classifier", page_icon="🛡️", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_classifier.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None

st.title("🛡️ AI Message Guard")
st.caption("SMS Spam / Ham Classifier — TF-IDF + Naive Bayes")
st.warning("⚠️ Educational tool. HAM does NOT guarantee safety.")

tab1, tab2 = st.tabs(["🔍 Classify Message", "📊 Model Performance"])

with tab1:
    message = st.text_area("Paste your message:", height=150)

    if st.button("Classify", type="primary"):
        if not message.strip():
            st.error("Please enter a message.")
        else:
            pred = model.predict([message])[0]
            probs = dict(zip(model.classes_, model.predict_proba([message])[0]))
            spam_prob = float(probs.get("spam", 0.0))

            if pred == "spam":
                st.error(f"### 🚨 SPAM (confidence: {spam_prob:.1%})")
            else:
                st.success(f"### ✅ HAM (confidence: {1 - spam_prob:.1%})")

            st.progress(spam_prob, text=f"Spam: {spam_prob:.1%}")
            st.progress(1 - spam_prob, text=f"Ham: {(1 - spam_prob):.1%}")

with tab2:
    if metrics:
        st.subheader("Evaluation Metrics (Test Set)")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("Spam F1", f"{metrics['f1_spam']:.1%}")
        c3, c4 = st.columns(2)
        c3.metric("Spam Precision", f"{metrics['precision_spam']:.1%}")
        c4.metric("Spam Recall", f"{metrics['recall_spam']:.1%}")

        cm = metrics["confusion_matrix"]
        st.dataframe(
            {
                "Actual \\ Predicted": ["Ham", "Spam"],
                "Predicted Ham": [cm[0][0], cm[1][0]],
                "Predicted Spam": [cm[0][1], cm[1][1]],
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("metrics.json not found.")