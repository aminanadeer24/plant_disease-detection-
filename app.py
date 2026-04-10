import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# ⚠ Replace this with your actual class mapping
class_names = {
    0: 'Potato___Early_blight',
    1: 'Potato___Late_blight',
    2: 'Tomato___Healthy'
}

st.title("🌱 Plant Disease Detection System")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png"])

if uploaded_file is not None:
    
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    disease_name = class_names[predicted_class]

    # Display Result
    st.success(f"🌿 Disease Detected: {disease_name}")
    st.info(f"🔍 Confidence: {round(confidence*100, 2)} %")