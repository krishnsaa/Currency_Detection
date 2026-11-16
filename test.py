import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("currency_model.keras")
class_names = ["fake", "real"]

def check_note(img_path):
    img = load_img(img_path, target_size=(160, 160))
    arr = np.expand_dims(img_to_array(img), axis=0)
    pred = model.predict(arr, verbose=0)[0][0]

    label = class_names[1] if pred >= 0.5 else class_names[0]
    conf = pred if pred >= 0.5 else (1 - pred)

    plt.imshow(img)
    plt.title(f"{label.upper()} ({conf * 100:.2f}%)")
    plt.axis("off")
    plt.show()

check_note("aug_3_WhatsApp Image 2025-04-05 at 10.55.41 (1).jpeg")

