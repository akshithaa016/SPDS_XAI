import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import cv2
import os
import uuid

def generate_saliency_map(model, image, save_dir="backend/saliency_outputs"):
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    image = tf.convert_to_tensor(image)

    # Get the prediction class
    preds = model(image)
    pred_class = tf.argmax(preds[0])

    # Watch the input
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = predictions[:, pred_class]

    # Calculate gradient of the output with respect to input image
    grads = tape.gradient(loss, image)
    grads = tf.reduce_max(tf.abs(grads), axis=-1)[0]  # Reduce across color channels

    # Normalize gradients
    grads = (grads - tf.reduce_min(grads)) / (tf.reduce_max(grads) - tf.reduce_min(grads) + K.epsilon())

    # Convert to numpy
    saliency_map = grads.numpy()

    # Save image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saliency_filename = f"{uuid.uuid4().hex}_saliency.png"
    saliency_path = os.path.join(save_dir, saliency_filename)

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(image[0], cmap='gray')
    plt.imshow(saliency_map, cmap='hot', alpha=0.5)
    plt.savefig(saliency_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return saliency_path
