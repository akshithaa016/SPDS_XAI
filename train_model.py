import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Set dataset path relative to your project folder
dataset_dir = "chest_xray"

# Hyperparameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 10

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'  # pneumonia = 1, normal = 0
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "val"),
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

# Normalize pixel values [0,1]
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Define a simple CNN model
model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Create assets folder if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Save the model
model.save("assets/pneumonia_model.h5")

print("Model trained and saved at assets/pneumonia_model.h5")
