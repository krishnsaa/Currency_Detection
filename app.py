import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fake Currency Detector", page_icon="💵")
st.title("Fake Currency Note Detector")
st.write("Upload an image of an Indian currency note to verify if it is real or fake.")

@st.cache_resource
def load_currency_model():
    return tf.keras.models.load_model("currency_model.keras")

model = load_currency_model()
class_names = ["Fake Note", "Real Note"]

# Automatically read expected (height, width) from the loaded model
input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]
target_height, target_width = input_shape[1], input_shape[2]

uploaded_file = st.file_uploader("Choose a currency note image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Currency Note", use_container_width=True)

    # Resize to the exact dimensions expected by the model
    img_resized = image.resize((target_width, target_height))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing note..."):
        predictions = model.predict(img_batch)
        score = float(predictions[0][0])

        # Handle both binary sigmoid (1 output) and 2-class softmax outputs
        if predictions.shape[-1] == 1:
            predicted_index = 1 if score >= 0.5 else 0
            confidence = score if predicted_index == 1 else 1.0 - score
        else:
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index])

        label = class_names[predicted_index]

    if label == "Real Note":
        st.success(f"Prediction: **{label}** ({confidence * 100:.2f}% confidence)")
    else:
        st.error(f"Prediction: **{label}** ({confidence * 100:.2f}% confidence)")
