import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# 💾 Load saved model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 🎨 Streamlit Page Configuration
st.set_page_config(page_title="News Classifier", layout="centered")

st.title("📰 News Article Classification App")
st.write("Type or paste a news headline or article content below to classify it into a category.")

# ✍️ User Input
user_input = st.text_area("Enter the news article text here:")

# 🔍 Predict Button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform the input text
        X_input = vectorizer.transform([user_input])
        
        # Predict the category
        prediction = model.predict(X_input)[0]
        
        # Display Result
        st.success(f"Predicted Category: **{prediction}**")
