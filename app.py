import streamlit as st
import tensorflow as tf
import re
from keras.layers import TFSMLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# -----------------------------
# Load SavedModel using TFSMLayer
# -----------------------------
# Make sure "model_saved" folder is next to this app.py
# Check the input name from SavedModel signature (here we assume 'input_1')
model = TFSMLayer("model_saved", call_endpoint="serving_default")

# -----------------------------
# Load IMDB word index
# -----------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

max_len = 1000

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()

    # Encode words using IMDB word_index
    encoded = [word_index.get(w, 2) + 3 for w in words]  # 2 = OOV token
    padded = pad_sequences([encoded], maxlen=max_len)
    return padded

# -----------------------------
# Prediction using TFSMLayer
# -----------------------------
def predict_sentiment(review):
    processed = preprocess_text(review)
    processed = tf.convert_to_tensor(processed, dtype=tf.int32)

    # ⚡ IMPORTANT: pass input as dict with key from SavedModel signature
    # Change "input_1" if your SavedModel input has a different name
    output = model({"input_1": processed})

    # Convert to float
    prob = float(list(output.values())[0].numpy()[0][0])

    sentiment = "Positive 😊" if prob >= 0.5 else "Negative 😞"
    return sentiment, prob

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IMDB Movie Review Classifier", page_icon="🎬")

st.title("🎬 IMDB Movie Review Classifier by Hirushan")
st.write("Analyze the emotional tone of any movie review using a trained neural network model.")

st.markdown(
    """
    ---
    ### **Enter your movie review below**
    My model will read your text and classify it as **Positive** or **Negative**.
    """
)

user_input = st.text_area(
    "Movie Review",
    placeholder="Type your movie review here...",
    height=150,
    max_chars=1000
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
