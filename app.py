"""
Streamlit app for IMDB Movie Review Sentiment Analysis.

Run:
    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from preprocessing import clean_text

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/sentiment_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return model, vectorizer


@st.cache_data
def load_sample_data():
    df = pd.read_csv("data/IMDB Dataset.csv")
    return df


model, vectorizer = load_artifacts()

st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "A classic NLP project on the **IMDB 50k Movie Reviews** dataset — "
    "TF-IDF features + Logistic Regression, ~90% test accuracy."
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Explore Dataset", "ℹ️ About the Model"])

# ---------------- Tab 1: Prediction ----------------
with tab1:
    st.subheader("Try it out")
    example = st.selectbox(
        "Load an example (optional):",
        [
            "",
            "This movie was an absolute masterpiece, the acting and story blew me away!",
            "Waste of two hours. Boring plot, wooden acting, I want my time back.",
        ],
    )
    user_input = st.text_area(
        "Enter a movie review:",
        value=example,
        height=150,
        placeholder="Type or paste a movie review here...",
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = proba[pred]
            else:
                score = model.decision_function(vec)[0]
                confidence = 1 / (1 + pow(2.718281828, -score))  # sigmoid approx for display
                if pred == 0:
                    confidence = 1 - confidence

            label = "Positive 😊" if pred == 1 else "Negative 😞"
            color = "green" if pred == 1 else "red"

            st.markdown(f"### Prediction: :{color}[{label}]")
            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence * 100:.1f}%")

# ---------------- Tab 2: Explore Dataset ----------------
with tab2:
    st.subheader("Dataset Overview")
    df = load_sample_data()
    col1, col2 = st.columns(2)
    col1.metric("Total reviews", f"{len(df):,}")
    col2.metric("Classes", "Positive / Negative (balanced)")

    st.write("**Sentiment distribution:**")
    fig, ax = plt.subplots(figsize=(4, 3))
    df["sentiment"].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.write("**Sample reviews:**")
    st.dataframe(df.sample(5, random_state=1)[["review", "sentiment"]], use_container_width=True)

# ---------------- Tab 3: About ----------------
with tab3:
    st.subheader("Model Details")
    try:
        with open("models/model_info.txt") as f:
            st.text(f.read())
    except FileNotFoundError:
        st.write("Run `train.py` first to generate model info.")

    st.write("**Pipeline:**")
    st.markdown(
        "- Clean text (lowercase, strip HTML tags, remove punctuation/numbers)\n"
        "- TF-IDF vectorization (unigrams + bigrams, 40,000 features)\n"
        "- Compared Logistic Regression, Linear SVM, Naive Bayes\n"
        "- Best model selected by F1 score and saved for inference"
    )

    col1, col2 = st.columns(2)
    try:
        col1.image("models/confusion_matrix.png", caption="Confusion Matrix")
    except Exception:
        pass
    try:
        col2.image("models/roc_curve.png", caption="ROC Curve")
    except Exception:
        pass
