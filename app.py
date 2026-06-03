import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.write("Enter a news article below and click Predict.")

news_text = st.text_area(
    "News Content",
    height=200
)

if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:

        news_vector = vectorizer.transform([news_text])

        prediction = model.predict(news_vector)

        confidence = max(
            model.predict_proba(news_vector)[0]
        ) * 100

        if prediction[0] == 0:
            st.error(
                f"❌ Fake News\n\nConfidence: {confidence:.2f}%"
            )
        else:
            st.success(
                f"✅ Real News\n\nConfidence: {confidence:.2f}%"
            )

st.sidebar.title("About")
st.sidebar.info(
    """
    Fake News Detection Project

    Technologies Used:
    - Python
    - Scikit-Learn
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """
)