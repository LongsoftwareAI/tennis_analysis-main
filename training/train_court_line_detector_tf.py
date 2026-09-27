import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
import torch
import sys

sys.path.append(".")
from court_line_detector import CourtLineDetector

def load_court_data(img_dir, annotation_file, max_samples=None):
    """
    Load image paths and ground-truth 14 keypoint coordinates from JSON annotation.
    """
    with open(annotation_file, "r") as f:
        data = json.load(f)

    images = []
    keypoints_list = []
    raw_images = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for item in data:
        if max_samples and len(images) >= max_samples:
            break
        img_name = item.get("id", "") or item.get("image", "")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            if not img_name.endswith(".png") and not img_name.endswith(".jpg"):
                for ext in [".png", ".jpg", ".jpeg"]:
                    if os.path.exists(img_path + ext):
                        img_path = img_path + ext
                        break
        if not os.path.exists(img_path):
            continue

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue

        orig_h, orig_w = raw_img.shape[:2]
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (224, 224))
        norm_img = ((resized.astype(np.float32) / 255.0) - mean) / std

        kps = np.array(item["kps"], dtype=np.float32)
        # Normalize keypoints to 224x224 coordinate scale
        if np.max(kps) <= 1.05:  # Already normalized [0, 1]
            kps[::2] *= 224.0
            kps[1::2] *= 224.0
        else:  # Absolute pixel coordinates
            kps[::2] *= 224.0 / orig_w
            kps[1::2] *= 224.0 / orig_h

        images.append(norm_img)
        keypoints_list.append(kps)
        raw_images.append(img_rgb)

    return np.array(images, dtype=np.float32), np.array(keypoints_list, dtype=np.float32), raw_images

def plot_court_training_results(history, model, val_images, val_targets, raw_val_images, save_dir="reports/court_detector_plots"):
    """
    Plot training loss/MAE curves and sample visual keypoint predictions for the report.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n=== Generating Court Detector Training & Evaluation Charts ===")

    epochs = range(1, len(history.history["loss"]) + 1)

    # 1. Loss & MAE Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TensorFlow Court Line Detector (ResNet50) - Training Progress", fontsize=14, fontweight="bold")

    # Loss (MSE)
    axes[0].plot(epochs, history.history["loss"], label="Train Loss (MSE)", color="#1f77b4", linewidth=2)
    if "val_loss" in history.history:
        axes[0].plot(epochs, history.history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Loss Curve (Mean Squared Error)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # MAE (Mean Absolute Error)
    if "mae" in history.history:
        axes[1].plot(epochs, history.history["mae"], label="Train MAE (Pixels)", color="#2ca02c", linewidth=2)
    if "val_mae" in history.history:
        axes[1].plot(epochs, history.history["val_mae"], label="Val MAE (Pixels)", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy Metric (Mean Absolute Error)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (Coordinate Pixels)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    loss_chart_path = os.path.join(save_dir, "court_detector_loss_mae_curves.png")
    plt.savefig(loss_chart_path, dpi=300)
    print(f" -> Saved Loss/MAE Curves: {loss_chart_path}")
    plt.show(block=False)
    plt.close()

    # 2. Visual Prediction Overlay Comparison on Validation Samples
    if len(val_images) > 0:
        num_samples = min(4, len(val_images))
        preds = model.predict(val_images[:num_samples], verbose=0)

        fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
        if num_samples == 1:
            axes = [axes]
        fig.suptitle("Court Line Keypoints: Ground Truth (Green) vs Predicted (Red)", fontsize=13, fontweight="bold")

        for idx in range(num_samples):
            sample_img = cv2.resize(raw_val_images[idx], (224, 224)).copy()
            gt_kps = val_targets[idx]
            pred_kps = preds[idx]

            # Draw Ground Truth in Green
            for i in range(0, len(gt_kps), 2):
                gx, gy = int(gt_kps[i]), int(gt_kps[i + 1])
                cv2.circle(sample_img, (gx, gy), 4, (0, 255, 0), -1)

            # Draw Predicted in Red
            for i in range(0, len(pred_kps), 2):
                px, py = int(pred_kps[i]), int(pred_kps[i + 1])
                cv2.circle(sample_img, (px, py), 3, (255, 0, 0), -1)

            axes[idx].imshow(sample_img)
            axes[idx].set_title(f"Sample #{idx + 1}")
            axes[idx].axis("off")

        plt.tight_layout()
        pred_chart_path = os.path.join(save_dir, "court_keypoints_visual_predictions.png")
        plt.savefig(pred_chart_path, dpi=300)
        print(f" -> Saved Visual Predictions Chart: {pred_chart_path}")
        plt.show(block=False)
        plt.close()

    print(f"=== All court detector plots saved to: {os.path.abspath(save_dir)} ===\n")

def train_court_detector(img_dir, annotation_file, epochs=50, batch_size=16, lr=1e-4, save_path="models/keypoints_model.keras", max_samples=None):
    """
    Train ResNet50 court keypoints model using TensorFlow / Keras.
    """
    print("=== Training Court Line Keypoints Detector (TensorFlow / Keras) ===")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(annotation_file):
        print(f"Warning: Annotation file {annotation_file} not found. Creating initialized model at {save_path}...")
        detector = CourtLineDetector()
        detector.save(save_path)
        print(f"Model saved to {save_path}")
        return

    X, y, raw_images = load_court_data(img_dir, annotation_file, max_samples=max_samples)
    print(f"Loaded {len(X)} samples for training.")

    # Train / Val Split
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    raw_val = raw_images[split_idx:]

    detector = CourtLineDetector()
    model = detector.model

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    model.save(save_path)
    # Also save .h5 copy for compatibility
    h5_path = save_path.replace(".keras", ".h5")
    model.save(h5_path)
    print(f"\nTraining Complete! Model saved to {save_path} and {h5_path}")

    # Generate and display charts for the report
    plot_court_training_results(history, model, X_val, y_val, raw_val, save_dir="reports/court_detector_plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Court Line Detector using TensorFlow.")
    parser.add_argument("--img_dir", type=str, default="training/datasets/court/tennis_court_keypoints", help="Path to court image directory.")
    parser.add_argument("--annotation_file", type=str, default="training/datasets/court/tennis_court_keypoints/data.json", help="Path to annotations json.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max_samples", type=int, default=2500, help="Max samples to load (default: 2500).")
    parser.add_argument("--save_path", type=str, default="models/keypoints_model.keras", help="Model destination path.")

    args = parser.parse_args()
    train_court_detector(args.img_dir, args.annotation_file, args.epochs, args.batch_size, args.lr, args.save_path, args.max_samples)
