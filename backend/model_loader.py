import tensorflow as tf
import os

def load_model():
    model_path = os.path.join("assets", "pneumonia_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

