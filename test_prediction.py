from backend.model_loader import load_model
from backend.predictor import preprocess_image, predict

# Load the model
model = load_model()
print("✅ Model loaded successfully.")

# Load and preprocess an image
image_path = "sample_test1.jpg"  # <- Change this to a test image path
with open(image_path, "rb") as f:
    processed = preprocess_image(f)

# Predict
label, confidence = predict(model, processed)
print(f"🩺 Prediction: {label} ({confidence}%)")
