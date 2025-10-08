import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import requests
from PIL import Image
import io

# -----------------------------
# LOAD MODEL (.h5)
# -----------------------------
MODEL_PATH = "cat_dog_classifier.h5"  # Change to your model path
model = tf.keras.models.load_model(MODEL_PATH)

# Set image size
img_width, img_height = 100, 100

# -----------------------------
# FUNCTION TO PROCESS IMAGE
# -----------------------------
def process_image(img, img_width, img_height):
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, img_width, img_height, 3))
    img_array = img_array / 255.0
    return img_array

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🐶🐱 Pet Predictor - Dog or Cat Classifier")
st.write("Upload an image or paste an image URL to check if it’s a **Dog or Cat**!")

option = st.radio("Choose Input Type:", ["Upload from Device", "Use Image URL", "Random from Folder"])

# --- 1️⃣ Upload from Device ---
if option == "Upload from Device":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        img_array = process_image(img, img_width, img_height)
        pred = model.predict(img_array)
        pred_class = "🐶 Dog" if pred > 0.5 else "🐱 Cat"
        st.success(f"**Prediction:** {pred_class}")

# --- 2️⃣ Use Image URL ---
elif option == "Use Image URL":
    image_url = st.text_input("Enter Image URL:")
    if st.button("Predict from URL"):
        if image_url:
            try:
                response = requests.get(image_url)
                img = Image.open(io.BytesIO(response.content))
                st.image(img, caption="Image from URL", use_container_width=True)
                img_array = process_image(img, img_width, img_height)
                pred = model.predict(img_array)
                pred_class = "🐶 Dog" if pred > 0.5 else "🐱 Cat"
                st.success(f"**Prediction:** {pred_class}")
            except Exception as e:
                st.error(f"Error loading image: {e}")

# --- 3️⃣ Random from Folder ---
elif option == "Random from Folder":
    test_folder_path = "D:\\cat_dog_dataset\\test_set_1"  # Change as needed
    test_image_files = [f for f in os.listdir(test_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if st.button("Pick Random Image"):
        random_image_file = random.choice(test_image_files)
        random_image_path = os.path.join(test_folder_path, random_image_file)
        img = Image.open(random_image_path)
        st.image(img, caption=f"Random Image: {random_image_file}", use_container_width=True)
        img_array = process_image(img, img_width, img_height)
        pred = model.predict(img_array)
        pred_class = "🐶 Dog" if pred > 0.5 else "🐱 Cat"
        st.success(f"**Prediction:** {pred_class}")
