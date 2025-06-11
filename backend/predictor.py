import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype('float32') / 255.0
    return image_array
# Make a prediction using the model
def predict(model, processed_image):
    prediction = model.predict(processed_image)
    label = "PNEUMONIA" if prediction[0][0] >= 0.5 else "NORMAL"
    confidence = float(prediction[0][0]) if label == "PNEUMONIA" else 1 - float(prediction[0][0])
    return label, round(confidence * 100, 2)
