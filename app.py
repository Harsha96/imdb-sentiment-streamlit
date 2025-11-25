import streamlit as st
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


# Load Model 

model = tf.keras.models.load_model("model.h5")

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

max_len = 1000 


# Text Preprocessing

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()

    encoded = [word_index.get(w, 2) + 3 for w in words]
    padded = pad_sequences([encoded], maxlen=max_len)
    return padded



# Prediction 

def predict_sentiment(review):
    processed = preprocess_text(review)
    prob = model.predict(processed)[0][0]
    sentiment = "Positive 😊" if prob >= 0.5 else "Negative 😞"
    return sentiment, prob



# Streamlit 

st.set_page_config(page_title="IMDB Movie Review Classifier by Hirushan", page_icon="🎬")

st.title("🎬 IMDB Movie Review Classifier by Hirushan")
st.write("Analyze the emotional tone of any movie review using a trained neural network model.")

st.markdown(
    """
    ---
    ### **Enter your movie review below**
    Our model will read your text and classify it as **Positive** or **Negative**.
    """
)

user_input = st.text_area(
    "Movie Review",
    placeholder="Type your movie review here...",
    height=150
)

if st.button("🔍 Classify Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review before classifying.")
    else:
        sentiment, score = predict_sentiment(user_input)

        st.markdown("---")
        st.subheader("🔎 Prediction Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** `{score:.4f}`")
        st.markdown("---")

     
        if score > 0.8:
            st.success("Model is highly confident in this prediction.")
        elif score > 0.5:
            st.info("Model is moderately confident.")
        else:
            st.warning("Model is uncertain. The review may contain mixed expressions.")

st.markdown(
    """
    ---
    **Developed by:** *Harsha*  
    *(LSTM Based Sentiment Classification Project)*
    """
)
