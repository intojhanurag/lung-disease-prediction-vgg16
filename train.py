import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *


def create_data_generators():
    """Create training, validation, and test data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT,
        height_shift_range=HEIGHT_SHIFT,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE,
        validation_split=0.15,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def build_model():
    """Build VGG16 transfer learning model."""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    # Freeze all VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history):
    """Save training accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "static", "training_history.png"), dpi=150)
    plt.close()
    print("Training history plot saved to static/training_history.png")


def plot_confusion_matrix(y_true, y_pred):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)

    classes = ["NORMAL", "PNEUMONIA"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=20)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "static", "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Confusion matrix saved to static/confusion_matrix.png")


def main():
    print("=" * 60)
    print("LUNG DISEASE PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # Create model directory
    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

    # Step 1: Create data generators
    print("\n[1/5] Loading dataset...")
    train_generator, val_generator, test_generator = create_data_generators()
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")

    # Step 2: Compute class weights for imbalanced data
    print("\n[2/5] Computing class weights...")
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Step 3: Build model
    print("\n[3/5] Building VGG16 model...")
    model = build_model()
    model.summary()

    # Step 4: Train
    print("\n[4/5] Training model...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    # Step 5: Evaluate
    print("\n[5/5] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Generate predictions for classification report
    y_pred_probs = model.predict(test_generator)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    y_true = test_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

    # Save plots
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)

    # Save model
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
