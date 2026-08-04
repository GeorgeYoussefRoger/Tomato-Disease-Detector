import streamlit as st
import requests

st.title("🍅 Tomato Disease Detector")
st.markdown("Upload a tomato leaf image to detect diseases.")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    with st.spinner("Analyzing..."):
        response = requests.post("http://localhost:8000/predict", files={"image": uploaded})
        result = response.json()
        pred_name = result["prediction"]
        confidence = result["confidence"]

    st.image(uploaded, caption="Uploaded image", width=300)
    col1, col2 = st.columns(2)
    col1.metric("Prediction", pred_name)
    col2.metric("Confidence", f"{confidence:.1%}")