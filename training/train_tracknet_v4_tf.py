import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import re
import cv2
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.environ["KERAS_BACKEND"] = "torch"

import torch
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print(f"GPU Accelerated: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

import keras
from keras import layers, ops

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

print(f"Working Directory: {project_root}")
print(f"Keras Version: {keras.__version__} (Backend: {keras.backend.backend()})")

os.makedirs("reports/tracknet_v4_reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Dataset Extraction
def get_label_for_img(img_path):
    lbl_path = img_path.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
    lbl_path = os.path.splitext(lbl_path)[0] + '.txt'
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        return float(parts[1]), float(parts[2])
                    except ValueError:
                        pass
    return None

def extract_all_triplets():
    print("Extracting continuous temporal triplets...")
    pattern_d3 = re.compile(r'^(.*?)_mp4-(\d+)_jpg')
    d3_files = glob.glob('training/datasets/ball/dataset_3_me_tennis/**/*.jpg', recursive=True)
    clips = defaultdict(dict)
    for p in d3_files:
        fn = os.path.basename(p)
        m = pattern_d3.match(fn)
        if m:
            clip_id = m.group(1)
            frame_idx = int(m.group(2))
            coord = get_label_for_img(p)
            clips[clip_id][frame_idx] = (p, coord)
    
    triplets = []
    for clip_id, frames in clips.items():
        sorted_indices = sorted(frames.keys())
        for i in range(1, len(sorted_indices) - 1):
            prev_idx, curr_idx, next_idx = sorted_indices[i-1], sorted_indices[i], sorted_indices[i+1]
            if curr_idx == prev_idx + 1 and next_idx == curr_idx + 1:
                triplets.append((frames[prev_idx], frames[curr_idx], frames[next_idx], clip_id))

    pattern_d2 = re.compile(r'^frame_(\d+)_jpg')
    d2_files = glob.glob('training/datasets/ball/dataset_2_ball_detection/**/*.jpg', recursive=True)
    d2_frames = {}
    for p in d2_files:
        fn = os.path.basename(p)
        m = pattern_d2.match(fn)
        if m:
            frame_idx = int(m.group(1))
            coord = get_label_for_img(p)
            d2_frames[frame_idx] = (p, coord)
    
    sorted_idx2 = sorted(d2_frames.keys())
    for i in range(1, len(sorted_idx2) - 1):
        prev_idx, curr_idx, next_idx = sorted_idx2[i-1], sorted_idx2[i], sorted_idx2[i+1]
        if curr_idx == prev_idx + 1 and next_idx == curr_idx + 1:
            triplets.append((d2_frames[prev_idx], d2_frames[curr_idx], d2_frames[next_idx], "dataset_2"))

    print(f"Total valid temporal triplets: {len(triplets)}")
    return triplets

triplets = extract_all_triplets()
random.seed(42)
random.shuffle(triplets)

split_idx = int(len(triplets) * 0.85)
train_triplets = triplets[:split_idx]
val_triplets = triplets[split_idx:]
print(f"Train triplets: {len(train_triplets)}, Val triplets: {len(val_triplets)}")

TARGET_H, TARGET_W = 288, 512

# Cache frames into RAM
print("Caching resized frames into RAM...")
all_img_paths = set()
for t in triplets:
    all_img_paths.add(t[0][0])
    all_img_paths.add(t[1][0])
    all_img_paths.add(t[2][0])

frame_cache = {}
t0 = time.time()
for p in all_img_paths:
    raw = cv2.imread(p)
    if raw is not None:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        frame_cache[p] = resized
print(f"Cached {len(frame_cache)} frames in {time.time() - t0:.2f}s (~{len(frame_cache)*TARGET_H*TARGET_W*3 / 1024**2:.1f} MB)")

def generate_gaussian_heatmap(h, w, cx, cy, sigma=2.5):
    if cx is None or cy is None:
        return np.zeros((h, w, 1), dtype=np.float32)
    gx = int(round(cx * w))
    gy = int(round(cy * h))
    if gx < 0 or gx >= w or gy < 0 or gy >= h:
        return np.zeros((h, w, 1), dtype=np.float32)
    
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist_sq = (xx - gx)**2 + (yy - gy)**2
    heatmap = np.exp(-dist_sq / (2.0 * sigma**2))
    return np.expand_dims(heatmap.astype(np.float32), axis=-1)

# Memory-efficient Triplet Generator with batch_size=2
BATCH_SIZE = 2

class MemoryEfficientTripletGenerator(keras.utils.PyDataset):
    def __init__(self, triplet_list, batch_size=BATCH_SIZE, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.triplets = triplet_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.triplets))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.triplets) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        B = len(batch_idx)
        X = np.zeros((B, TARGET_H, TARGET_W, 9), dtype=np.float32)
        Y_heat = np.zeros((B, TARGET_H, TARGET_W, 1), dtype=np.float32)
        Y_motion = np.zeros((B, TARGET_H, TARGET_W, 1), dtype=np.float32)

        for i, b_i in enumerate(batch_idx):
            f_prev, f_curr, f_next, _ = self.triplets[b_i]
            img_p = frame_cache[f_prev[0]].astype(np.float32) / 255.0
            img_c = frame_cache[f_curr[0]].astype(np.float32) / 255.0
            img_n = frame_cache[f_next[0]].astype(np.float32) / 255.0

            X[i] = np.concatenate([img_p, img_c, img_n], axis=-1)

            coord_c = f_curr[1]
            if coord_c is not None:
                cx, cy = coord_c
                Y_heat[i] = generate_gaussian_heatmap(TARGET_H, TARGET_W, cx, cy, sigma=2.5)

            diff = (np.abs(img_c - img_p) + np.abs(img_n - img_c)).mean(axis=-1, keepdims=True)
            Y_motion[i] = np.clip(diff * 3.0, 0.0, 1.0)

        return X, {'ball_heatmap': Y_heat, 'motion_attention': Y_motion}

# Model Definition
def build_tracknet_v4(input_shape=(TARGET_H, TARGET_W, 9)):
    inputs = layers.Input(shape=input_shape, name='triplet_input')

    # Motion Differencing: D1 = |I_t - I_{t-1}|, D2 = |I_{t+1} - I_t|
    f_prev = inputs[..., 0:3]
    f_curr = inputs[..., 3:6]
    f_next = inputs[..., 6:9]
    d1 = ops.abs(f_curr - f_prev)
    d2 = ops.abs(f_next - f_curr)
    motion_diff = ops.concatenate([d1, d2], axis=-1)

    # Motion Prompt Layer (MPL)
    m = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='mpl_conv1')(motion_diff)
    m = layers.BatchNormalization(name='mpl_bn1')(m)
    m = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='mpl_conv2')(m)
    m = layers.BatchNormalization(name='mpl_bn2')(m)
    motion_att = layers.Conv2D(1, (1, 1), activation='sigmoid', name='motion_attention')(m)

    # Backbone Encoder
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    c1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    c1 = layers.BatchNormalization()(c1)

    c1_fused = layers.Multiply(name='motion_fusion')([c1, 1.0 + motion_att])
    p1 = layers.MaxPooling2D((2, 2))(c1_fused)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(p1)
    x = layers.BatchNormalization()(x)
    c2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(p2)
    x = layers.BatchNormalization()(x)
    c3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(p3)
    x = layers.BatchNormalization()(x)
    b = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    b = layers.BatchNormalization()(b)

    # Decoder
    u1 = layers.UpSampling2D((2, 2), interpolation='bilinear')(b)
    cat1 = layers.Concatenate()([u1, c3])
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(cat1)
    x = layers.BatchNormalization()(x)
    d1_out = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    d1_out = layers.BatchNormalization()(d1_out)

    u2 = layers.UpSampling2D((2, 2), interpolation='bilinear')(d1_out)
    cat2 = layers.Concatenate()([u2, c2])
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(cat2)
    x = layers.BatchNormalization()(x)
    d2_out = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    d2_out = layers.BatchNormalization()(d2_out)

    u3 = layers.UpSampling2D((2, 2), interpolation='bilinear')(d2_out)
    cat3 = layers.Concatenate()([u3, c1])
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(cat3)
    x = layers.BatchNormalization()(x)
    d3_out = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    d3_out = layers.BatchNormalization()(d3_out)

    heatmap = layers.Conv2D(1, (1, 1), activation='sigmoid', name='ball_heatmap')(d3_out)
    model = keras.Model(inputs=inputs, outputs={'ball_heatmap': heatmap, 'motion_attention': motion_att}, name='TrackNetV4_TF')
    return model

class WeightedBCE_MSE_Loss(keras.losses.Loss):
    def __init__(self, pos_weight=40.0, mse_weight=5.0, name='weighted_bce_mse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = pos_weight
        self.mse_weight = mse_weight

    def call(self, y_true, y_pred):
        eps = 1e-7
        y_pred = ops.clip(y_pred, eps, 1.0 - eps)
        bce = - (self.pos_weight * y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred))
        bce_loss = ops.mean(bce)
        mse_loss = ops.mean(ops.square(y_true - y_pred))
        return bce_loss + self.mse_weight * mse_loss

model = build_tracknet_v4()
print(f"TrackNetV4 Total params: {model.count_params():,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'ball_heatmap': WeightedBCE_MSE_Loss(pos_weight=40.0, mse_weight=5.0),
        'motion_attention': keras.losses.MeanSquaredError()
    },
    loss_weights={
        'ball_heatmap': 1.0,
        'motion_attention': 0.1
    }
)

train_gen = MemoryEfficientTripletGenerator(train_triplets, batch_size=BATCH_SIZE, shuffle=True)
val_gen = MemoryEfficientTripletGenerator(val_triplets, batch_size=BATCH_SIZE, shuffle=False)

checkpoint_path = "models/tracknet_v4_ball_detector_best.keras"
if os.path.exists(checkpoint_path):
    print(f"Resuming weights from previous best checkpoint: {checkpoint_path}")
    try:
        model.load_weights(checkpoint_path)
        print("Successfully loaded pre-trained checkpoint weights!")
    except Exception as e:
        print(f"Starting training fresh: {e}")

class MemoryCleanupCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_epoch_end(self, epoch, logs=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_mb = torch.cuda.memory_allocated() / (1024**2)
            print(f" [VRAM in use: {vram_mb:.1f} MB]")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    MemoryCleanupCallback()
]

print(f"Starting TrackNetV4 Training (Max 30 Epochs, Early Stopping patience=4, batch_size={BATCH_SIZE})...")
EPOCHS = 30
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("Training finished! Calculating Validation Metrics...")
val_distances = []
true_positives = 0
false_positives = 0
false_negatives = 0
sample_eval_cases = []

for idx in range(min(40, len(val_gen))):
    X_val, Y_val = val_gen[idx]
    preds = model.predict_on_batch(X_val)
    pred_heats = ops.convert_to_numpy(preds['ball_heatmap'])
    gt_heats = Y_val['ball_heatmap']
    
    for b in range(len(X_val)):
        pred_map = pred_heats[b, :, :, 0]
        gt_map = gt_heats[b, :, :, 0]
        
        has_gt = gt_map.max() > 0.5
        gt_coord = None
        if has_gt:
            gy, gx = np.unravel_index(np.argmax(gt_map), gt_map.shape)
            gt_coord = (int(gx), int(gy))
            
        has_pred = pred_map.max() > 0.30
        pred_coord = None
        if has_pred:
            py, px = np.unravel_index(np.argmax(pred_map), pred_map.shape)
            pred_coord = (int(px), int(py))
            
        dist = None
        if has_gt and has_pred:
            dist = math.sqrt((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)
            val_distances.append(dist)
            if dist <= 8.0:
                true_positives += 1
            else:
                false_positives += 1
        elif has_pred and not has_gt:
            false_positives += 1
        elif has_gt and not has_pred:
            false_negatives += 1

        if len(sample_eval_cases) < 4 and has_gt:
            img_c_raw = (X_val[b, :, :, 3:6] * 255.0).astype(np.uint8)
            sample_eval_cases.append({
                'image': img_c_raw,
                'pred_map': pred_map,
                'gt_coord': gt_coord,
                'pred_coord': pred_coord,
                'dist': dist
            })

precision = true_positives / max(1, true_positives + false_positives)
recall = true_positives / max(1, true_positives + false_negatives)
f1 = 2 * precision * recall / max(1e-6, precision + recall)
mean_dist = np.mean(val_distances) if val_distances else 0.0

print(f"Validation Precision (<=8px): {precision:.4f} ({precision*100:.1f}%)")
print(f"Validation Recall (<=8px):    {recall:.4f} ({recall*100:.1f}%)")
print(f"Validation F1-score:         {f1:.4f} ({f1*100:.1f}%)")
print(f"Mean Distance Error:         {mean_dist:.2f} pixels")

# 1. Plot Training Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

axes[0].plot(history.history['ball_heatmap_loss'], label='Train Heatmap Loss', color='#1f77b4', lw=2)
axes[0].plot(history.history['val_ball_heatmap_loss'], label='Val Heatmap Loss', color='#ff7f0e', lw=2, linestyle='--')
axes[0].set_title('TrackNetV4 Heatmap Loss', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss Value', fontsize=11)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['motion_attention_loss'], label='Train Motion Loss', color='#2ca02c', lw=2)
axes[1].plot(history.history['val_motion_attention_loss'], label='Val Motion Loss', color='#d62728', lw=2, linestyle='--')
axes[1].set_title('Motion Prompt Auxiliary Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('MSE Loss', fontsize=11)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

if val_distances:
    axes[2].hist(val_distances, bins=20, color='#9467bd', edgecolor='black', alpha=0.7)
    axes[2].axvline(mean_dist, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_dist:.2f}px')
    axes[2].axvline(8.0, color='green', linestyle='dotted', linewidth=2, label='Tolerance (8px)')
axes[2].set_title(f'Localization Error Distribution (F1={f1:.3f})', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Pixel Distance Error (px)', fontsize=11)
axes[2].set_ylabel('Sample Count', fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
curve_path = "reports/tracknet_v4_reports/tracknet_v4_training_curves.png"
plt.savefig(curve_path, dpi=300)
plt.close()
print(f"Saved training curves to: {curve_path}")

# 2. Plot Bounce Evaluation
fig, axes = plt.subplots(len(sample_eval_cases), 3, figsize=(16, 4 * len(sample_eval_cases)))
for idx, case in enumerate(sample_eval_cases):
    img = case['image'].copy()
    gt_coord = case['gt_coord']
    pred_coord = case['pred_coord']
    pred_map = case['pred_map']
    dist = case['dist']

    img_gt = img.copy()
    if gt_coord:
        cv2.circle(img_gt, gt_coord, 10, (0, 255, 0), 2)
        cv2.putText(img_gt, "GT Ball", (gt_coord[0] + 12, gt_coord[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    axes[idx, 0].imshow(img_gt)
    axes[idx, 0].set_title(f"Sample {idx+1}: Ground Truth Frame t", fontsize=11, fontweight='bold')
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(pred_map, cmap='magma')
    axes[idx, 1].set_title(f"TrackNetV4 Predicted Heatmap (Max={pred_map.max():.2f})", fontsize=11, fontweight='bold')
    axes[idx, 1].axis('off')

    img_overlay = img.copy()
    if gt_coord:
        cv2.circle(img_overlay, gt_coord, 10, (0, 255, 0), 2)
    if pred_coord:
        cv2.circle(img_overlay, pred_coord, 6, (255, 0, 0), -1)
    dist_str = f"Error: {dist:.1f}px" if dist is not None else "No detection"
    axes[idx, 2].imshow(img_overlay)
    axes[idx, 2].set_title(f"Detected Ball (Red) vs GT (Green) | {dist_str}", fontsize=11, fontweight='bold')
    axes[idx, 2].axis('off')

plt.tight_layout()
bounce_path = "reports/tracknet_v4_reports/tracknet_v4_bounce_evaluation.png"
plt.savefig(bounce_path, dpi=300)
plt.close()
print(f"Saved bounce evaluation to: {bounce_path}")
print("30-Epoch Training pipeline with EarlyStopping complete!")
