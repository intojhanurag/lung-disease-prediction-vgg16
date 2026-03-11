---
title: Lung Disease Prediction
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Lung Disease Prediction Using Machine Learning

A deep learning-powered web application that analyzes chest X-ray images to predict whether a patient's lungs are **Normal** or show signs of **Pneumonia** — with real-time confidence scoring.

Built using **VGG16 transfer learning** on the Kaggle Chest X-Ray Pneumonia dataset (5,863 images), served through a **Flask** web interface with drag-and-drop image upload.

---

## Demo
![alt text](<Screenshot 2026-03-11 203410.png>)

| Upload Screen | Normal Result | Pneumonia Result |
|:---:|:---:|:---:|
| Drag & drop or browse X-ray | Green card with confidence % | Red alert with confidence % |

---

## Key Features

- **AI-Powered Diagnosis** — VGG16 deep learning model with ~90% test accuracy
- **Grad-CAM Heatmap** — Visual explainability showing which lung regions the AI focused on
- **Real-Time Prediction** — Upload an X-ray and get results in seconds
- **Confidence Score** — Shows prediction confidence as a percentage with visual bar
- **Model Performance Dashboard** — View confusion matrix, training curves, and model architecture
- **Prediction History** — Track and review recent analyses
- **Drag & Drop Upload** — Modern UI with image preview before analysis
- **Mobile Responsive** — Works on desktop, tablet, and mobile browsers
- **Class-Balanced Training** — Handles dataset imbalance using computed class weights

---

## Model Architecture

```
VGG16 (ImageNet pretrained, frozen) → GlobalAveragePooling2D → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
```

| Metric | Value |
|--------|-------|
| Base Model | VGG16 (ImageNet weights) |
| Total Parameters | 14,977,857 |
| Trainable Parameters | 263,169 |
| Input Size | 224 x 224 x 3 (RGB) |
| Output | Binary (Normal / Pneumonia) |
| Model Size | ~60 MB |
| Optimizer | Adam (lr=0.0001) |
| Loss Function | Binary Crossentropy |

### Training Callbacks
- **EarlyStopping** — patience=5, restores best weights
- **ReduceLROnPlateau** — halves LR after 3 epochs of no improvement
- **ModelCheckpoint** — saves best model by validation accuracy

---

## Dataset

**Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Test | 234 | 390 | 624 |

- Images: JPEG chest X-rays, varying sizes, resized to 224x224
- Validation: 15% split from training data (not the default 16-image val folder)
- Class imbalance handled with `sklearn.utils.class_weight.compute_class_weight`

### Data Augmentation (Training Only)
| Parameter | Value |
|-----------|-------|
| Rotation | 20° |
| Width/Height Shift | 10% |
| Shear | 10% |
| Zoom | 15% |
| Horizontal Flip | Yes |
| Rescaling | 1/255 |

---

## Project Structure

```
├── app.py                  # Flask web application
├── train.py                # Model training script
├── predict.py              # Prediction utility function
├── gradcam.py              # Grad-CAM heatmap generation
├── config.py               # Hyperparameters and paths
├── requirements.txt        # Python dependencies
├── Dockerfile              # Hugging Face Spaces deployment
├── Procfile                # Gunicorn config
├── render.yaml             # Render config (backup)
├── model/
│   └── vgg16_pneumonia.keras   # Trained model (60 MB)
├── static/
│   ├── css/style.css           # Styling
│   ├── js/main.js              # Drag-drop, preview, validation
│   ├── uploads/                # Temporary upload directory
│   ├── confusion_matrix.png    # Evaluation visualization
│   └── training_history.png    # Accuracy/loss curves
├── templates/
│   ├── index.html              # Upload page
│   ├── result.html             # Prediction result + Grad-CAM page
│   ├── performance.html        # Model performance dashboard
│   ├── history.html            # Prediction history page
│   └── about.html              # Project info page
├── notebooks/
│   └── training.ipynb          # Google Colab training notebook
└── data/                       # Dataset directory (not in repo)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Model | VGG16 (Transfer Learning) |
| Image Processing | OpenCV |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Deployment | Hugging Face Spaces (Docker) / Gunicorn |
| Training Platform | Google Colab (GPU) |

---

## Getting Started

### Prerequisites
- Python 3.10+ (tested with 3.12)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/intojhanurag/lung-disease-prediction-vgg16.git
cd lung-disease-prediction-vgg16

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### Train the Model (Optional)

Training is done on **Google Colab** for free GPU access:

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `notebooks/training.ipynb`
3. Set **Runtime → Change runtime type → GPU (T4)**
4. Click **Runtime → Run all**
5. Wait ~15-20 minutes
6. Download the generated `vgg16_pneumonia.keras` → place in `model/`
7. Download `confusion_matrix.png` and `training_history.png` → place in `static/`

---

## API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Upload page — drag & drop or browse for X-ray image |
| `/predict` | POST | Accepts image, runs model inference, returns result with Grad-CAM |
| `/history` | GET | View recent prediction history |
| `/history/clear` | POST | Clear prediction history |
| `/performance` | GET | Model performance dashboard with metrics and plots |
| `/about` | GET | Project information and team details |

---

## Evaluation Results

After training on Google Colab with GPU (T4):

- **Test Accuracy:** ~88–92%
- **Validation Accuracy:** ~90–93%

Detailed metrics (precision, recall, F1-score) and visualizations are generated during training:
- `static/confusion_matrix.png` — True vs Predicted label counts
- `static/training_history.png` — Accuracy and loss curves across epochs

---

## Live Demo

**Try it now:** [https://huggingface.co/spaces/anurag2004/lung-disease-prediction](https://huggingface.co/spaces/anurag2004/lung-disease-prediction)

---

## Deployment

Deployed on **Hugging Face Spaces** using Docker (16GB RAM, free tier).

Render.com was attempted first but its free tier (512MB RAM) was insufficient for TensorFlow + VGG16 inference.

### To deploy your own:
1. Fork this repository
2. Create a new Space on [Hugging Face](https://huggingface.co/new-space) with **Docker SDK**
3. Push the code to the Space
4. It builds and deploys automatically

---

## Team

| Name | Roll Number |
|------|-------------|
| Kanhaiya Kumar Sahani | 2300970130067 |
| Anurag Kumar Ojha | 2300970130027 |
| Asad Khan | 2300970130033 |

**Guide:** Mrs. Shanu Verma
**College:** Galgotias College of Engineering & Technology
**Department:** Information Technology
**Subject:** BIT-753 (Major Project)

---

## Disclaimer

This application is developed for **educational and academic purposes only**. It is not intended to be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## License

This project is developed as part of an academic curriculum and is intended for educational use.
