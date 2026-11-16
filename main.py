import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

data_dir = "dataset"
img_size = (160, 160)
batch_size = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

augment = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1)
])

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))
x = augment(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=6)

model.save("currency_model.keras")

plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Fake Currency Detection")
plt.legend()
plt.grid(True)
plt.show()

def check_note(img_path):
    img = load_img(img_path, target_size=img_size)
    arr = np.expand_dims(img_to_array(img), axis=0)
    pred = model.predict(arr, verbose=0)[0][0]

    label = class_names[1] if pred >= 0.5 else class_names[0]
    conf = pred if pred >= 0.5 else (1 - pred)

    plt.imshow(img)
    plt.title(f"{label.upper()} ({conf * 100:.2f}%)")
    plt.axis("off")
    plt.show()

