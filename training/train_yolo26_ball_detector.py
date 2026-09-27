import os
import argparse
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

def plot_and_display_ball_results(run_dir, save_dir="reports/ball_detector_plots"):
    """
    Parse results.csv from ball detector training, generate publication-ready
    loss and metric curves, copy confusion matrices, and display them on screen.
    """
    os.makedirs(save_dir, exist_ok=True)
    results_csv = os.path.join(run_dir, "results.csv")

    if not os.path.exists(results_csv):
        print(f"[!] Warning: results.csv not found at {results_csv}")
        return

    print(f"\n=== Generating Tennis Ball Training & Evaluation Charts for Report ===")
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    # 1. Plot Training & Validation Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tennis Ball Detector (YOLO26 Large) - Loss Curves", fontsize=14, fontweight="bold")

    # Box Loss
    if "train/box_loss" in df.columns:
        axes[0].plot(epochs, df["train/box_loss"], label="Train Box Loss", color="#1f77b4", linewidth=2)
    if "val/box_loss" in df.columns:
        axes[0].plot(epochs, df["val/box_loss"], label="Val Box Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Bounding Box Loss (Ball Localization)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Classification / Progression / DFL Loss
    cls_col = "train/cls_loss" if "train/cls_loss" in df.columns else None
    val_cls_col = "val/cls_loss" if "val/cls_loss" in df.columns else None
    dfl_col = "train/dfl_loss" if "train/dfl_loss" in df.columns else ("train/prog_loss" if "train/prog_loss" in df.columns else None)

    if cls_col:
        axes[1].plot(epochs, df[cls_col], label="Train Class Loss", color="#2ca02c", linewidth=2)
    if val_cls_col:
        axes[1].plot(epochs, df[val_cls_col], label="Val Class Loss", color="#d62728", linewidth=2, linestyle="--")
    if dfl_col:
        axes[1].plot(epochs, df[dfl_col], label=f"Train {dfl_col.split('/')[-1].upper()}", color="#9467bd", linewidth=1.5, linestyle=":")
    axes[1].set_title("Classification & Head Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    loss_chart_path = os.path.join(save_dir, "ball_detector_loss_curves.png")
    plt.savefig(loss_chart_path, dpi=300)
    print(f" -> Saved Loss Chart: {loss_chart_path}")
    plt.show(block=False)
    plt.close()

    # 2. Plot Evaluation Metrics (mAP50, mAP50-95, Precision, Recall)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tennis Ball Detector (YOLO26 Large) - Evaluation Metrics", fontsize=14, fontweight="bold")

    map50_col = [c for c in df.columns if "mAP50(" in c or c == "metrics/mAP50"]
    map95_col = [c for c in df.columns if "mAP50-95(" in c or c == "metrics/mAP50-95"]
    p_col = [c for c in df.columns if "precision(" in c or c == "metrics/precision"]
    r_col = [c for c in df.columns if "recall(" in c or c == "metrics/recall"]

    if map50_col:
        axes[0].plot(epochs, df[map50_col[0]], label="mAP @ 0.50", color="#e377c2", linewidth=2.5)
    if map95_col:
        axes[0].plot(epochs, df[map95_col[0]], label="mAP @ 0.50:0.95", color="#17becf", linewidth=2)
    axes[0].set_title("Mean Average Precision (mAP)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    if p_col:
        axes[1].plot(epochs, df[p_col[0]], label="Precision", color="#2ca02c", linewidth=2)
    if r_col:
        axes[1].plot(epochs, df[r_col[0]], label="Recall", color="#ff7f0e", linewidth=2)
    axes[1].set_title("Precision & Recall Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    metrics_chart_path = os.path.join(save_dir, "ball_detector_metrics_curves.png")
    plt.savefig(metrics_chart_path, dpi=300)
    print(f" -> Saved Metrics Chart: {metrics_chart_path}")
    plt.show(block=False)
    plt.close()

    # 3. Copy official Ultralytics result figures to report folder
    copy_patterns = [
        ("results.png", "ball_detector_training_summary_all.png"),
        ("confusion_matrix.png", "ball_detector_confusion_matrix.png"),
        ("confusion_matrix_normalized.png", "ball_detector_confusion_matrix_norm.png"),
        ("PR_curve.png", "ball_detector_precision_recall_curve.png"),
        ("F1_curve.png", "ball_detector_f1_score_curve.png"),
        ("val_batch0_pred.jpg", "ball_detector_sample_predictions.jpg"),
        ("val_batch1_pred.jpg", "ball_detector_sample_predictions_batch2.jpg")
    ]

    for src_name, dst_name in copy_patterns:
        src_file = os.path.join(run_dir, src_name)
        if os.path.exists(src_file):
            dst_file = os.path.join(save_dir, dst_name)
            shutil.copy2(src_file, dst_file)
            print(f" -> Copied Report Figure: {dst_file}")

    print(f"=== All report plots saved to: {os.path.abspath(save_dir)} ===\n")

def train_ball_detector(data_yaml, epochs=50, imgsz=640, batch_size=16, patience=10, model_name="yolo26s.pt", run_name=None, export_tf=True):
    """
    Train YOLO26 Small model exclusively on tennis ball dataset,
    export to TensorFlow SavedModel & TFLite, and generate report plots.
    Includes early stopping mechanism (patience epochs without metric improvement).
    """
    if run_name is None:
        if "best.pt" in model_name:
            run_name = "yolo26s_ball_detector_100e"
        else:
            run_name = "yolo26s_ball_detector"

    print(f"=== Starting YOLO26 Tennis Ball Training ===")
    print(f"Base Model: {model_name} (Ultralytics YOLO26 Small - Optimized for Speed & Accuracy)")
    print(f"Ball Dataset: {data_yaml}")
    print(f"Run Name: {run_name}")
    print(f"Epochs: {epochs} | Image Resolution: {imgsz} | Batch Size: {batch_size} | Early Stopping Patience: {patience}")

    # Load YOLO26 Small pretrained weights
    model = YOLO(model_name)

    import torch
    selected_device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware Accelerator: {device_name} (device={selected_device})")

    # Train model on ball detection (small-object specialized settings with early stopping)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        device=selected_device,
        plots=True,
        save=True,
        project="runs/detect",
        name=run_name,
        exist_ok=True
    )

    candidate_runs = [
        os.path.join("runs", "detect", run_name),
        os.path.join("runs", "detect", "runs", "detect", run_name),
        os.path.join("runs", run_name)
    ]
    run_dir = next((d for d in candidate_runs if os.path.exists(os.path.join(d, "weights", "best.pt")) or os.path.exists(os.path.join(d, "results.csv"))), candidate_runs[0])
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = model_name

    print(f"\n[Ball Training Complete] Best weights: {best_weights}")

    # Save a direct copy to models/ for easy access
    os.makedirs("models", exist_ok=True)
    shutil.copy2(best_weights, "models/ball_detector_yolo26_best.pt")
    print(f"Copied best model weights to: models/ball_detector_yolo26_best.pt")

    # Generate and display charts for the report
    plot_and_display_ball_results(run_dir=run_dir, save_dir="reports/ball_detector_plots")

    if export_tf:
        print("\n=== Exporting Ball Detector to TensorFlow SavedModel & TFLite ===")
        trained_model = YOLO(best_weights)

        # 1. Export to TensorFlow SavedModel
        try:
            tf_path = trained_model.export(format="saved_model", imgsz=imgsz)
            print(f"Successfully exported TensorFlow SavedModel to: {tf_path}")
            
            dst_tf_dir = os.path.join("models", "ball_detector_tf_saved_model")
            if os.path.exists(dst_tf_dir):
                shutil.rmtree(dst_tf_dir)
            shutil.copytree(tf_path, dst_tf_dir)
            print(f"Copied TensorFlow SavedModel to: {dst_tf_dir}")
        except Exception as e:
            print(f"Notice during SavedModel export: {e}")

        # 2. Export to TensorFlow Lite
        try:
            tflite_path = trained_model.export(format="tflite", imgsz=imgsz)
            print(f"Successfully exported TFLite model to: {tflite_path}")
            dst_tflite = os.path.join("models", "ball_detector.tflite")
            shutil.copy2(tflite_path, dst_tflite)
            print(f"Copied TFLite model to: {dst_tflite}")
        except Exception as e:
            print(f"Notice during TFLite export: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26 Small exclusively on tennis ball dataset.")
    parser.add_argument("--dataset", type=str, default="merged",
                        choices=["3", "1", "2", "merged"],
                        help="Choose target ball dataset: 'merged' (Unified 4,454 images), '3' (Dataset by Me), '1' (Viren Dhanwani), or '2' (Tennis Project).")
    parser.add_argument("--data", type=str, default="",
                        help="Custom path to dataset data.yaml file (overrides --dataset).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs with no improvement before stopping).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size (e.g., 640 or 1280 for tiny ball).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="Base model weights or checkpoint path (e.g. runs/detect/yolo26s_ball_detector/weights/best.pt).")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the training run folder.")
    parser.add_argument("--no_export", action="store_true", help="Skip TensorFlow export after training.")

    args = parser.parse_args()

    dataset_map = {
        "3": "training/datasets/ball/dataset_3_me_tennis/data.yaml",
        "1": "training/datasets/ball/dataset_1_tennis_ball/data.yaml",
        "2": "training/datasets/ball/dataset_2_ball_detection/data.yaml",
        "merged": "training/datasets/ball/merged_tennis_dataset/data.yaml"
    }

    if args.data:
        data_yaml = args.data
    else:
        data_yaml = dataset_map.get(args.dataset, "training/datasets/ball/merged_tennis_dataset/data.yaml")

    if not os.path.exists(data_yaml):
        print(f"\n[!] Error: Dataset not found at '{data_yaml}'.")
        print(" -> Available datasets:")
        for k, v in dataset_map.items():
            exists = "Exists" if os.path.exists(v) else "Missing"
            print(f"    --dataset {k}: {v} [{exists}]")
        exit(1)

    print(f"Selected Ball Dataset: {data_yaml}")

    train_ball_detector(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        patience=args.patience,
        model_name=args.model,
        run_name=args.run_name,
        export_tf=not args.no_export
    )
