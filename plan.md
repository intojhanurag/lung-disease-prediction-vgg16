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
6. Shows Grad-CAM heatmap highlighting the region model focused on
7. Deploys on a publicly accessible URL

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
major-project/
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
│   ├── uploads/                # Temporary upload directory (gitignored)
│   ├── confusion_matrix.png    # Evaluation plot
│   └── training_history.png    # Training curves plot
├── templates/
│   ├── index.html              # Upload page
│   ├── result.html             # Prediction result page
│   └── about.html              # Project info page
├── data/                       # Dataset directory (gitignored)
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
└── notebooks/
    └── training.ipynb          # Google Colab training notebook
```

---

---

# COMPLETED PHASES (RECORD)

---

## PHASE 1: PROJECT SETUP — COMPLETED

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
*:Zone.Identifier
plan.md
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

## PHASE 2: DATASET DOWNLOAD AND PREPARATION — COMPLETED

### Step 2.1: Download Dataset
Used `kagglehub` inside Google Colab notebook to auto-download the dataset.

### Step 2.2: Verify Dataset Structure
Dataset verified with correct structure:
```
train/NORMAL/     → 1341 images
train/PNEUMONIA/  → 3875 images
test/NORMAL/      → 234 images
test/PNEUMONIA/   → 390 images
```
Validation split created from training data (15%) instead of using the default 16-image val folder.

---

## PHASE 3: MODEL TRAINING (train.py) — COMPLETED

### Training Details:
- **Trained on**: Google Colab with GPU (T4)
- **Training notebook**: `notebooks/training.ipynb`
- **Model architecture**: VGG16 (frozen) + GlobalAveragePooling2D + Dense(512) + Dropout(0.5) + Dense(1, sigmoid)
- **Total Parameters**: 14,977,857
- **Trainable Parameters**: 263,169
- **Model file size**: ~60 MB
- **Class imbalance handled**: Yes, using `compute_class_weight("balanced")`

### Outputs Generated:
- [x] `model/vgg16_pneumonia.keras` — trained model file
- [x] `static/confusion_matrix.png` — confusion matrix visualization
- [x] `static/training_history.png` — accuracy/loss curves

---

## PHASE 4: PREDICTION UTILITY (predict.py) — COMPLETED

- predict_pneumonia() function built
- Handles grayscale → RGB conversion
- Resizes to 224x224, normalizes, returns class label + confidence %

---

## PHASE 5: FLASK WEB APPLICATION (app.py) — COMPLETED

### Routes Built:
| Route      | Method | Purpose                           |
|------------|--------|-----------------------------------|
| `/`        | GET    | Upload page (index.html)          |
| `/predict` | POST   | Run prediction, show result       |
| `/about`   | GET    | Project info page (about.html)    |

- Model loads once at startup
- File validation (type + size)
- Flash error messages for invalid uploads

---

## PHASE 6: FRONTEND TEMPLATES AND STYLING — COMPLETED

- [x] `templates/index.html` — Upload page with drag-and-drop
- [x] `templates/result.html` — Result page with color-coded prediction card + confidence bar
- [x] `templates/about.html` — Project info page with team details
- [x] `static/css/style.css` — Full responsive styling, medical theme
- [x] `static/js/main.js` — Drag-drop, image preview, file validation, loading spinner
- [x] Mobile responsive design with flexbox

---

## PHASE 7: TRAINING NOTEBOOK FOR GOOGLE COLAB — COMPLETED

- [x] `notebooks/training.ipynb` — Fully self-contained notebook
- Auto-downloads dataset via kagglehub
- Trains model, evaluates, generates plots
- Auto-downloads 3 files (model + 2 plots) to user's browser

---

## PHASE 8: DEPLOYMENT CONFIG — COMPLETED

- [x] `Procfile` created
- [x] `render.yaml` created
- [x] `Dockerfile` created for Hugging Face Spaces
- [x] `.python-version` added for deployment
- [x] Push to GitHub — DONE
- [x] Deploy on Hugging Face Spaces — DONE (Render free tier had insufficient RAM for TensorFlow)

---

## PHASE 9: TESTING AND VERIFICATION — COMPLETED

### Local Testing:
- [x] Flask app runs on localhost:5000
- [x] Homepage loads with upload form
- [x] Can upload JPEG/PNG X-ray image
- [x] Prediction shows correct label (NORMAL or PNEUMONIA)
- [x] Confidence percentage is displayed (tested: 64.69% Pneumonia)
- [x] "Upload Another Image" button works

### Deployed Testing:
- [x] Test deployed URL with sample X-ray images — WORKING

---

---

# COMPLETED UPGRADE PHASES

---

## PHASE 10: GRAD-CAM HEATMAP VISUALIZATION — COMPLETED

- [x] `gradcam.py` created — generates Grad-CAM heatmap from VGG16 last conv layer (block5_conv3)
- [x] `/predict` route updated to generate Grad-CAM after each prediction
- [x] `result.html` updated — shows original X-ray and Grad-CAM heatmap side by side
- [x] Color legend added explaining red/yellow/green/blue regions
- [x] Context-aware explanation text (different for Normal vs Pneumonia)

---

## PHASE 11: MODEL PERFORMANCE DASHBOARD — COMPLETED

- [x] `/performance` route added
- [x] `templates/performance.html` created with:
  - Model architecture flow diagram
  - Parameters table (14.9M total, 263K trainable)
  - Training configuration grid
  - Dataset distribution with visual bar chart
  - Data augmentation details
  - Confusion matrix and training history plots
  - Training callbacks explanation

---

## PHASE 12: PREDICTION HISTORY — COMPLETED

- [x] In-memory list storing last 20 predictions
- [x] `/history` route added
- [x] `templates/history.html` created with:
  - Color-coded table (green/red rows)
  - Image thumbnails, prediction badges, confidence, timestamps
  - Clear history button
  - Empty state with upload link
- [x] `/history/clear` POST route for clearing history

---

## PHASE 13: PUSH TO GITHUB — COMPLETED

- [x] Repository: https://github.com/intojhanurag/college_major_project
- [x] Git LFS set up for `.keras`, `.png`, `.jpg`, `.jpeg` files
- [x] All files committed and pushed
- [x] Navigation bar added across all pages (Upload, History, Performance, About)

---

## PHASE 14: DEPLOY ON HUGGING FACE SPACES — COMPLETED

- [x] Render.com attempted first — failed due to 512MB RAM limit (insufficient for TensorFlow + VGG16)
- [x] Switched to Hugging Face Spaces (Docker SDK) — 16GB RAM, free
- [x] `Dockerfile` created with Python 3.10, OpenCV dependencies, Gunicorn
- [x] `.python-version` file added
- [x] Deployed successfully at: https://huggingface.co/spaces/anurag2004/lung-disease-prediction

---

## PHASE 15: FINAL TESTING — COMPLETED

- [x] Homepage loads correctly on public URL
- [x] Upload X-ray → get prediction with confidence
- [x] Grad-CAM heatmap shows correctly
- [x] Performance page displays all metrics and plots
- [x] History page tracks recent predictions
- [x] About page shows team info
- [x] Disclaimer is visible
- [x] All pages have navigation bar

---

## IMPORTANT NOTES

1. **Do NOT use any deprecated Keras imports.** Use `tf.keras` everywhere, not standalone `keras`.
2. **Save model in `.keras` format**, not `.h5`. Use `model.save("model/vgg16_pneumonia.keras")`.
3. **VGG16 expects RGB input.** If X-ray is grayscale, convert to RGB before feeding to model.
4. **The dataset is imbalanced** (3875 pneumonia vs 1341 normal). Handled using `class_weight` in `model.fit()`.
5. **The val/ folder in dataset has only 16 images.** Used `validation_split=0.15` from training data instead.
6. **For deployment**, model file is ~60MB. Use Git LFS if pushing to GitHub.
7. **All file paths must use `os.path.join()`** for cross-platform compatibility.
8. **Flask `static/uploads/` folder**: Created at startup if it doesn't exist.
9. **The web UI must have a disclaimer** stating this is for educational purposes only.
10. **Do not skip generating confusion_matrix.png and training_history.png** — needed for PPT and report.
