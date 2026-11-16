import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("currency_model.keras")
class_names = ["Fake Note", "Real Note"]

def check_currency(image):
    # Resize and prepare the image
    img = image.resize((160, 160))
    arr = np.expand_dims(img_to_array(img), axis=0)
    
    # Predict
    pred = model.predict(arr, verbose=0)[0][0]
    
    # Return probabilities for the UI bars
    if pred >= 0.5:
        return {class_names[1]: float(pred), class_names[0]: float(1 - pred)}
    else:
        return {class_names[0]: float(1 - pred), class_names[1]: float(pred)}

# Create the web interface
demo = gr.Interface(
    fn=check_currency,
    inputs=gr.Image(type="pil", label="Upload Currency Photo"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Fake Indian Currency Detector",
    description="Upload or take a photo of an Indian currency note to verify its authenticity."
)

demo.launch()

