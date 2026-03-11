import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from config import MODEL_PATH, UPLOAD_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS
from predict import predict_pneumonia
from gradcam import generate_gradcam

app = Flask(__name__)
app.secret_key = "lung-disease-prediction-secret-key"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file uploaded. Please select an X-ray image.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose an X-ray image.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a PNG, JPG, or JPEG image.")
        return redirect(url_for("index"))

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Run prediction
    class_label, confidence = predict_pneumonia(filepath, model)

    # Generate Grad-CAM heatmap
    gradcam_path = generate_gradcam(filepath, model)
    gradcam_filename = os.path.basename(gradcam_path)

    # Pass image paths relative to static folder for display
    image_url = url_for("static", filename=f"uploads/{filename}")
    gradcam_url = url_for("static", filename=f"uploads/{gradcam_filename}")

    return render_template(
        "result.html",
        prediction=class_label,
        confidence=confidence,
        image_url=image_url,
        gradcam_url=gradcam_url,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
