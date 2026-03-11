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
