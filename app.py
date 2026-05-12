import streamlit as st
import pickle
import os

# Page settings
st.set_page_config(
    page_title="News Category Predictor",
    page_icon="📰",
    layout="centered"
)

# Load trained model
model = pickle.load(open("news_model.pkl", "rb"))

# Create uploads folder if not exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Title
st.title("📰 News Category Prediction App")

st.write(
    "Predict whether news belongs to Sports, Film, Beauty, Games, Introduction etc."
)

# -------------------------------
# OPTION 1 : Manual Text Input
# -------------------------------

st.subheader("✍️ Enter News Text")

news_text = st.text_area(
    "Type your news here",
    height=150
)

# -------------------------------
# OPTION 2 : Upload Text File
# -------------------------------

st.subheader("📁 Upload Text File")

uploaded_file = st.file_uploader(
    "Upload a .txt file",
    type=["txt"]
)

file_text = ""

if uploaded_file is not None:

    # Save uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read file content
    file_text = uploaded_file.read().decode("utf-8")

    st.success("File uploaded successfully!")

    st.text_area(
        "File Content",
        file_text,
        height=200
    )

# -------------------------------
# Prediction Logic
# -------------------------------

st.subheader("🔍 Prediction")

if st.button("Predict Category"):

    # Use uploaded file text if available
    final_text = file_text if file_text else news_text

    if final_text.strip() == "":
        st.warning("Please enter text or upload a file.")

    else:

        # Prediction
        prediction = model.predict([final_text])[0]

        # Confidence score
        probability = model.predict_proba([final_text]).max() * 100

        # Results
        st.success(f"Predicted Category: {prediction}")

        st.info(f"Confidence Score: {probability:.2f}%")

# Footer
st.markdown("---")
st.caption("Built using Streamlit + NLP + Machine Learning")