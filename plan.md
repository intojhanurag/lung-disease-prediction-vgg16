# LUNG DISEASE PREDICTION USING MACHINE LEARNING — COMPLETE BUILD PLAN

## PROJECT IDENTITY

- **Project Title**: Lung Diseases Prediction Using Machine Learning
- **Subject Code**: BIT-753 (Major Project)
- **College**: Galgotias College of Engineering & Technology
- **Department**: Information Technology
- **Team Members**:
  - Kanhaiya Kumar Sahani (2300970130067)
  - Anurag Kumar Ojha (2300970130027)
  - Asad Khan (2300970130033)
- **Guide**: Mrs. Shanu Verma

---

## GOAL

Build a complete end-to-end web application that:
1. Takes a chest X-ray image as input
2. Preprocesses it
3. Runs it through a trained VGG16 deep learning model
4. Predicts whether the lung is **NORMAL** or has **PNEUMONIA**
5. Shows the prediction result with confidence percentage on a web UI
6. Deploys on a publicly accessible URL

---

## TECH STACK (FINAL — NO ALTERNATIVES)

| Component          | Technology                  | Version        |
|--------------------|-----------------------------|----------------|
| Language           | Python                      | 3.10+          |
| Deep Learning      | TensorFlow                  | 2.15.0         |
| Model              | VGG16 (transfer learning)   | Keras built-in |
| Image Processing   | OpenCV (cv2)                | 4.9+           |
| Data Augmentation  | tf.keras.preprocessing      | built-in       |
| Web Framework      | Flask                       | 3.0+           |
| Frontend           | HTML + CSS + JavaScript     | vanilla (no framework) |
| Charts             | Chart.js (CDN)              | 4.x            |
| Deployment         | Render.com (free tier)      | -              |
| Version Control    | Git + GitHub                | -              |
| Dataset            | Kaggle Chest X-Ray Pneumonia| Paul Mooney    |

---

## DATASET DETAILS

- **Source**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Name**: Chest X-Ray Images (Pneumonia)
- **Total Images**: 5,863 JPEG images
- **Structure**:
  ```
  chest_xray/
  ├── train/
  │   ├── NORMAL/       (1,341 images)
  │   └── PNEUMONIA/    (3,875 images)
  ├── val/
  │   ├── NORMAL/       (8 images)
  │   └── PNEUMONIA/    (8 images)
  └── test/
      ├── NORMAL/       (234 images)
      └── PNEUMONIA/    (390 images)
  ```
- **Classes**: 2 (Binary Classification)
  - Class 0: NORMAL
  - Class 1: PNEUMONIA
- **Image Format**: JPEG, grayscale/RGB, varying sizes
- **Download Method**: Use `kagglehub` Python package OR manual download via Kaggle CLI

---

## PROJECT FOLDER STRUCTURE (EXACT)

```
lung-disease-prediction/
├── app.py                      # Flask application (main entry point)
├── train.py                    # Model training script
├── predict.py                  # Prediction utility functions
├── config.py                   # All hyperparameters and paths
├── requirements.txt            # Python dependencies
├── Procfile                    # For Render deployment
├── render.yaml                 # Render config
├── .gitignore
├── README.md
├── model/
│   └── vgg16_pneumonia.keras   # Saved trained model (Keras format)
├── static/
│   ├── css/
│   │   └── style.css           # All styles
│   ├── js/
│   │   └── main.js             # Frontend JavaScript
│   └── uploads/                # Temporary upload directory (gitignored)
├── templates/
│   ├── index.html              # Upload page
│   └── result.html             # Prediction result page
├── data/                       # Dataset directory (gitignored)
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
└── notebooks/
    └── training.ipynb          # Google Colab training notebook (backup)
```

---

## PHASE 1: PROJECT SETUP

### Step 1.1: Initialize Project
```bash
mkdir lung-disease-prediction
cd lung-disease-prediction
git init
```

### Step 1.2: Create requirements.txt
```
tensorflow==2.15.0
flask==3.0.0
opencv-python-headless==4.9.0.80
numpy==1.26.4
Pillow==10.2.0
matplotlib==3.8.3
scikit-learn==1.4.0
gunicorn==21.2.0
```

### Step 1.3: Create .gitignore
```
data/
model/*.keras
model/*.h5
static/uploads/*
__pycache__/
*.pyc
.env
venv/
*.zip
```

### Step 1.4: Create config.py
```python
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "model", "vgg16_pneumonia.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# Image settings
IMG_SIZE = 224          # VGG16 requires 224x224
IMG_CHANNELS = 3        # RGB
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
DENSE_UNITS = 512

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT = 0.1
HEIGHT_SHIFT = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.15
HORIZONTAL_FLIP = True
FILL_MODE = "nearest"

# Class labels
CLASS_LABELS = {0: "NORMAL", 1: "PNEUMONIA"}

# Flask settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
```

---

## PHASE 2: DATASET DOWNLOAD AND PREPARATION

### Step 2.1: Download Dataset
Use ONE of these methods (try in order):

**Method A — kagglehub (preferred)**:
```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(path)
# Then copy/symlink to data/chest_xray/
```

**Method B — Kaggle CLI**:
```bash
pip install kaggle
# Ensure ~/.kaggle/kaggle.json exists with your API key
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

**Method C — Manual**:
- Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Click Download → Unzip into `data/chest_xray/`

### Step 2.2: Verify Dataset Structure
After download, confirm this structure exists:
```
data/chest_xray/train/NORMAL/     → ~1341 .jpeg files
data/chest_xray/train/PNEUMONIA/  → ~3875 .jpeg files
data/chest_xray/test/NORMAL/      → ~234 .jpeg files
data/chest_xray/test/PNEUMONIA/   → ~390 .jpeg files
```
If `val/` has only 8+8=16 images (it does), we will create a proper validation split from training data in the code. Do NOT rely on the default val folder.

---

## PHASE 3: MODEL TRAINING (train.py)

### Step 3.1: Build train.py with these EXACT specifications

**Data Loading**:
- Use `tf.keras.preprocessing.image.ImageDataGenerator` for train data WITH augmentation
- Use `tf.keras.preprocessing.image.ImageDataGenerator` for test data with ONLY `rescale=1.0/255`
- Create validation split: use `validation_split=0.15` in the training generator (takes 15% of training data as validation)
- Training generator: `subset="training"`
- Validation generator: `subset="validation"`
- All images resized to 224×224 RGB
- `class_mode="binary"`
- `shuffle=True` for training, `shuffle=False` for test

**Data Augmentation (training data only)**:
- `rescale=1.0/255`
- `rotation_range=20`
- `width_shift_range=0.1`
- `height_shift_range=0.1`
- `shear_range=0.1`
- `zoom_range=0.15`
- `horizontal_flip=True`
- `fill_mode="nearest"`
- `validation_split=0.15`

**Model Architecture (EXACT)**:
```
1. Base Model: VGG16 (pretrained on ImageNet, include_top=False, input_shape=(224,224,3))
2. Freeze ALL layers of VGG16 base (layer.trainable = False)
3. Add on top:
   - GlobalAveragePooling2D()
   - Dense(512, activation="relu")
   - Dropout(0.5)
   - Dense(1, activation="sigmoid")    # Binary classification
```

**Compilation**:
- Optimizer: `Adam(learning_rate=0.0001)`
- Loss: `binary_crossentropy`
- Metrics: `["accuracy"]`

**Callbacks**:
- `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)`
- `ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)`

**Training**:
- `model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[...])`

**After Training — Generate and Save These Outputs**:
1. Save model to `model/vgg16_pneumonia.keras`
2. Print final test accuracy, precision, recall, F1-score
3. Generate and save confusion matrix plot → `static/confusion_matrix.png`
4. Generate and save training history plot (accuracy + loss curves) → `static/training_history.png`
5. Print classification report using `sklearn.metrics.classification_report`

**Expected Results** (approximate, based on literature):
- Training Accuracy: ~95-97%
- Validation Accuracy: ~90-93%
- Test Accuracy: ~88-92%
- These are realistic for VGG16 on this dataset

---

## PHASE 4: PREDICTION UTILITY (predict.py)

### Step 4.1: Build predict.py

This file provides a single function that the Flask app will call:

```python
def predict_pneumonia(image_path, model):
    """
    Takes an image file path and loaded model.
    Returns: (class_label: str, confidence: float, preprocessed_image: numpy_array)

    Steps:
    1. Load image using cv2.imread(image_path)
    2. If grayscale (single channel), convert to RGB using cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    3. Resize to (224, 224) using cv2.resize
    4. Normalize: img = img / 255.0
    5. Expand dims: img = np.expand_dims(img, axis=0)
    6. prediction = model.predict(img)[0][0]
    7. If prediction >= 0.5 → "PNEUMONIA" with confidence = prediction * 100
       If prediction < 0.5 → "NORMAL" with confidence = (1 - prediction) * 100
    8. Return class_label, confidence_percentage
    """
```

---

## PHASE 5: FLASK WEB APPLICATION (app.py)

### Step 5.1: Build app.py with these EXACT routes

**Global Setup**:
- Load the trained model ONCE at startup using `tf.keras.models.load_model(MODEL_PATH)`
- Create `static/uploads/` directory if it doesn't exist
- Set `MAX_CONTENT_LENGTH` for upload limit

**Routes**:

| Route              | Method    | Purpose                                    |
|--------------------|-----------|--------------------------------------------|
| `/`                | GET       | Show upload page (index.html)              |
| `/predict`         | POST      | Accept image upload, run prediction, show result |
| `/about`           | GET       | Show project info page (about the team, model info) |

**`/predict` Route Logic**:
1. Receive uploaded file from form
2. Validate: check file exists, has allowed extension (png/jpg/jpeg)
3. Save to `static/uploads/` with secure filename
4. Call `predict_pneumonia(filepath, model)`
5. Render `result.html` with: prediction label, confidence %, uploaded image path

**Error Handling**:
- No file uploaded → flash error, redirect to `/`
- Invalid file type → flash error, redirect to `/`
- Model loading failure → show error page

---

## PHASE 6: FRONTEND TEMPLATES AND STYLING

### Step 6.1: index.html (Upload Page)
**Design Requirements**:
- Clean medical-themed design
- Color scheme: white background, blue (#1a73e8) primary, light gray (#f8f9fa) sections
- Centered card layout with:
  - Project title: "Lung Disease Prediction"
  - Subtitle: "Upload a Chest X-Ray for AI-Powered Analysis"
  - Drag-and-drop area OR file input button
  - Image preview (show selected image before upload using JavaScript)
  - "Analyze X-Ray" submit button (blue, prominent)
  - Footer with team names and college name
- Mobile responsive (use flexbox)
- Use Google Fonts: "Inter" for body, "Poppins" for headings (load from CDN)

### Step 6.2: result.html (Result Page)
**Design Requirements**:
- Show the uploaded X-ray image (left side or top)
- Show prediction result in a large card:
  - If NORMAL: green background card, checkmark icon, "NORMAL" text
  - If PNEUMONIA: red background card, warning icon, "PNEUMONIA DETECTED" text
  - Confidence percentage shown as a circular progress indicator OR large number
- "Upload Another Image" button
- Disclaimer text: "This tool is for educational purposes only and should not be used as a substitute for professional medical advice."

### Step 6.3: style.css
**Specifications**:
- Max width for main content: 900px, centered
- Card: white background, border-radius 12px, box-shadow: 0 2px 20px rgba(0,0,0,0.08)
- Upload area: dashed 2px border, border-radius 12px, padding 40px, text centered
- Button: background #1a73e8, color white, padding 14px 40px, border-radius 8px, no border, cursor pointer, font-size 16px
- Hover on button: background #1557b0
- Result card: padding 30px, border-radius 12px
- Normal result: background #e8f5e9, border-left 5px solid #4caf50
- Pneumonia result: background #ffebee, border-left 5px solid #f44336
- Font sizes: h1=28px, h2=22px, body=16px

### Step 6.4: main.js
**Features**:
- Image preview when user selects a file
- Drag and drop file handling
- Loading spinner when form is submitted
- File size validation (max 16MB)
- File type validation (only jpg/jpeg/png)

---

## PHASE 7: TRAINING NOTEBOOK FOR GOOGLE COLAB (notebooks/training.ipynb)

Create a Jupyter notebook version of train.py that can be run on Google Colab for GPU training. This is the RECOMMENDED way to train since Colab provides free GPU.

**Notebook Structure**:
1. Cell 1: Install dependencies (`!pip install tensorflow opencv-python-headless scikit-learn matplotlib`)
2. Cell 2: Mount Google Drive OR download dataset from Kaggle
3. Cell 3: Import all libraries
4. Cell 4: Set all hyperparameters (same as config.py)
5. Cell 5: Create data generators
6. Cell 6: Build VGG16 model (same architecture as train.py)
7. Cell 7: Train model with callbacks
8. Cell 8: Evaluate on test set — print accuracy, classification report
9. Cell 9: Plot confusion matrix and save
10. Cell 10: Plot training history (accuracy + loss curves) and save
11. Cell 11: Save model as `vgg16_pneumonia.keras`
12. Cell 12: Download model file to local machine

---

## PHASE 8: DEPLOYMENT ON RENDER.COM

### Step 8.1: Prepare for Deployment

**Procfile** (root of project):
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

**render.yaml**:
```yaml
services:
  - type: web
    name: lung-disease-prediction
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.13
```

**Important for deployment**:
- The trained model file (`model/vgg16_pneumonia.keras`) MUST be committed to the repo (do NOT gitignore it) OR hosted on Google Drive/Hugging Face and downloaded at startup
- Recommended: Upload model to GitHub using Git LFS (since model will be ~60-100MB)
- Alternative: Store model on Google Drive with public link, download on app startup if not found locally

### Step 8.2: Deploy Steps
1. Push code to GitHub repository
2. Go to https://render.com → New → Web Service
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
6. Set Python version: 3.10
7. Deploy

### Step 8.3: Alternative Deployment — Streamlit Cloud (backup option)
If Render doesn't work (model too large for free tier), create an alternate `streamlit_app.py`:
- Use Streamlit instead of Flask
- Deploy on Streamlit Cloud (free, supports larger apps)
- Same prediction logic, different UI framework

---

## PHASE 9: TESTING AND VERIFICATION

### Step 9.1: Test Locally
```bash
# Start Flask app
python app.py

# Open browser to http://localhost:5000
# Upload a test X-ray image from data/chest_xray/test/NORMAL/ → should predict NORMAL
# Upload a test X-ray image from data/chest_xray/test/PNEUMONIA/ → should predict PNEUMONIA
```

### Step 9.2: Verify These Outputs Exist After Training
- [ ] `model/vgg16_pneumonia.keras` — trained model file
- [ ] `static/confusion_matrix.png` — confusion matrix visualization
- [ ] `static/training_history.png` — accuracy/loss curves
- [ ] Terminal output showing: test accuracy, precision, recall, F1-score, classification report

### Step 9.3: Verify Web App Works
- [ ] Homepage loads with upload form
- [ ] Can upload JPEG/PNG X-ray image
- [ ] Prediction shows correct label (NORMAL or PNEUMONIA)
- [ ] Confidence percentage is displayed
- [ ] "Upload Another" button works
- [ ] Invalid file type shows error message
- [ ] No file selected shows error message

---

## EXECUTION ORDER (DO THIS EXACTLY IN THIS SEQUENCE)

```
Step 1:  Create project folder structure and all config files
Step 2:  Install all dependencies from requirements.txt
Step 3:  Download dataset to data/chest_xray/
Step 4:  Create config.py with all settings
Step 5:  Create train.py with exact model architecture
Step 6:  Create predict.py with prediction function
Step 7:  Create notebooks/training.ipynb for Colab
Step 8:  Train model (locally OR on Colab with GPU)
         → This produces: model/vgg16_pneumonia.keras
         → This produces: confusion_matrix.png, training_history.png
         → This prints: accuracy, precision, recall, F1, classification report
Step 9:  Create Flask app (app.py)
Step 10: Create all HTML templates (index.html, result.html)
Step 11: Create CSS (style.css)
Step 12: Create JS (main.js)
Step 13: Test locally — upload test images, verify predictions
Step 14: Create Procfile and render.yaml
Step 15: Push to GitHub
Step 16: Deploy on Render.com
Step 17: Test deployed URL with sample X-ray images
```

---

## IMPORTANT NOTES FOR CLAUDE

1. **Do NOT use any deprecated Keras imports.** Use `tf.keras` everywhere, not standalone `keras`.
2. **Save model in `.keras` format**, not `.h5`. Use `model.save("model/vgg16_pneumonia.keras")`.
3. **VGG16 expects RGB input.** If X-ray is grayscale, convert to RGB before feeding to model.
4. **The dataset is imbalanced** (3875 pneumonia vs 1341 normal). Use `class_weight` parameter in `model.fit()` to handle this:
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   class_weights = compute_class_weight("balanced", classes=np.unique(train_generator.classes), y=train_generator.classes)
   class_weight_dict = dict(enumerate(class_weights))
   # Pass class_weight=class_weight_dict to model.fit()
   ```
5. **The val/ folder in dataset has only 16 images.** Do NOT use it as-is. Use `validation_split=0.15` from the training data instead.
6. **For deployment**, if model file is too large for GitHub (>100MB), use Git LFS:
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   ```
7. **All file paths must use `os.path.join()`** for cross-platform compatibility.
8. **Flask `static/uploads/` folder**: Create it at startup if it doesn't exist. Clean old uploads periodically.
9. **The web UI must have a disclaimer** stating this is for educational purposes only.
10. **Do not skip generating confusion_matrix.png and training_history.png** — these are needed for the PPT and report later.
