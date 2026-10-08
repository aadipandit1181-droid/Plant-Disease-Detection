import os
import sys
import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/plant_disease_model.keras"
CLASS_PATH = "models/class_names.json"
IMG_SIZE = (128, 128)

if len(sys.argv) < 2:
    print("Usage: python predict.py predictions/leaf.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("Image not found:", image_path)
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print("Trained model not found. Run train.py first.")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array, verbose=0)[0]
index = int(np.argmax(predictions))
confidence = float(predictions[index]) * 100

print("\n🌿 Plant Disease Detection Result")
print("--------------------------------")
print("Prediction :", class_names[index])
print(f"Confidence : {confidence:.2f}%")
