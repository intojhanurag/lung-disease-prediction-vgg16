import numpy as np
import cv2
import tensorflow as tf
from config import IMG_SIZE


def generate_gradcam(image_path, model, last_conv_layer_name="block5_conv3"):
    """
    Generate Grad-CAM heatmap for the given image.
    Returns the path to the saved overlay image.
    """
    # Load and preprocess image
    original_img = cv2.imread(image_path)
    if len(original_img.shape) == 2 or original_img.shape[2] == 1:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    resized_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    preprocessed = resized_img / 255.0
    preprocessed = np.expand_dims(preprocessed, axis=0)

    # Build a model that outputs the last conv layer activations and the final prediction
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output],
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed)
        predicted_class = predictions[0][0]

    # Gradients of the predicted class w.r.t. the last conv layer output
    grads = tape.gradient(predicted_class, conv_outputs)

    # Global average pooling of gradients — importance weight per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each filter by its importance and sum
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU — keep only positive influences
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to 0-1
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert to color map (0-255, apply JET colormap)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # Overlay heatmap on original image with transparency
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    # Save overlay image
    overlay_path = image_path.rsplit(".", 1)[0] + "_gradcam.jpg"
    cv2.imwrite(overlay_path, overlay)

    return overlay_path
