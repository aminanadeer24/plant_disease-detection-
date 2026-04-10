import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ==========================
# 1️⃣ Load Trained Model
# ==========================
model = tf.keras.models.load_model("plant_disease_model.h5")

# ==========================
# 2️⃣ Class Names
# IMPORTANT:
# Replace these with YOUR actual class names
# ==========================
class_names = {
    0: "Potato___Early_blight",
    1: "Potato___Late_blight",
    2: "Potato___Healthy",
    3: "Tomato___Early_blight",
    4: "Tomato___Late_blight",
    5: "Tomato___Healthy"
}

# ==========================
# 3️⃣ Load Test Image
# ==========================
img_path = "test_leaf.jpg"   # Make sure this file exists
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ==========================
# 4️⃣ Make Prediction
# ==========================
prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

# ==========================
# 5️⃣ Show Result
# ==========================
print("🌿 Predicted Disease:", class_names[predicted_class])
print("🔍 Confidence:", round(confidence * 100, 2), "%")