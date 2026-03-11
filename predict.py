import cv2
import numpy as np
from config import IMG_SIZE, CLASS_LABELS


def predict_pneumonia(image_path, model):
    """
    Takes an image file path and loaded model.
    Returns: (class_label: str, confidence: float)
    """
    # Load image
    img = cv2.imread(image_path)

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to 0-1
    img = img / 255.0

    # Add batch dimension: (224,224,3) -> (1,224,224,3)
    img = np.expand_dims(img, axis=0)

    # Get prediction
    prediction = model.predict(img)[0][0]

    # Determine class and confidence
    if prediction >= 0.5:
        class_label = CLASS_LABELS[1]  # PNEUMONIA
        confidence = prediction * 100
    else:
        class_label = CLASS_LABELS[0]  # NORMAL
        confidence = (1 - prediction) * 100

    return class_label, round(confidence, 2)
