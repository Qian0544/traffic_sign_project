"""
Hybrid CNN-SVM for Traffic Sign Classification
Method 3: CNN Feature Extraction + SVM Classification

Author: [Your Name]
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    top_k_accuracy_score, brier_score_loss
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import cv2
from tqdm import tqdm
import warnings
import time
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("HYBRID CNN-SVM - METHOD 3")
print("Traffic Sign Classification")
print("=" * 70)

# ============================================================================
# GPU CHECK
# ============================================================================
print("\n" + "=" * 70)
print("CHECKING HARDWARE")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU Available: {len(gpus)} GPU(s) detected")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    # Set memory growth to avoid OOM errors
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
else:
    print("⚠ No GPU detected - using CPU")
    print("  (Training will be significantly slower)")

print("=" * 70)


# ============================================================================
# CONFIGURATION
# ============================================================================
print("\n" + "=" * 70)
print("CONFIGURATION")
print("=" * 70)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dataset paths (update these to match your structure)
DATASET_NAME = 'Traffic_Sign_Detection'  # or 'Self-Driving_Cars'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

# Output paths
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(RESULTS_DIR, f'run_{TIMESTAMP}')
os.makedirs(RUN_DIR, exist_ok=True)

# Training parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# Split ratios (70% train, 15% val, 15% test)
TEST_SIZE = 0.15      # 15% for test
VAL_SIZE = 0.1765     # 15% of remaining = 0.15/(1-0.15) = 0.1765

# SVM parameters
SVM_KERNEL = 'rbf'
SVM_C = 10
SVM_GAMMA = 'scale'

print(f"Base Directory: {BASE_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"Dataset: {DATASET_NAME}")
print(f"Dataset Path: {DATASET_PATH}")
print(f"Results Directory: {RUN_DIR}")
print(f"\nTraining Parameters:")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"\nSplit Ratios:")
print(f"  Train: 70%")
print(f"  Validation: 15%")
print(f"  Test: 15%")
print(f"\nSVM Parameters:")
print(f"  Kernel: {SVM_KERNEL}")
print(f"  C: {SVM_C}")
print(f"  Gamma: {SVM_GAMMA}")

print("=" * 70)


# ============================================================================
# LOGGING SETUP
# ============================================================================
LOG_FILE = os.path.join(RUN_DIR, 'training_log.txt')

def log(message, also_print=True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')
    
    if also_print:
        print(message)

log("=" * 70)
log("HYBRID CNN-SVM TRAINING LOG")
log("=" * 70)


# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

def load_images_from_folder(root_dir, img_size=(224, 224), max_images=None):
    """
    Load images from folder structure with robust error handling.
    
    Expected structure:
        root_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                
    Args:
        root_dir: Root directory containing class folders
        img_size: Target size for images (width, height)
        max_images: Maximum images to load (None = all)
        
    Returns:
        images: numpy array of images
        labels: numpy array of class labels
    """
    images = []
    labels = []
    class_counts = {}
    
    # Validate directory exists
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"❌ ERROR: Directory not found!\n"
            f"{'='*70}\n"
            f"Looking for: {root_dir}\n\n"
            f"Expected folder structure:\n"
            f"  data/\n"
            f"  └── {DATASET_NAME}/\n"
            f"      ├── class1/\n"
            f"      │   ├── img1.jpg\n"
            f"      │   └── img2.jpg\n"
            f"      └── class2/\n"
            f"          └── img1.jpg\n\n"
            f"Please ensure:\n"
            f"  1. Data folder exists at: {DATA_DIR}\n"
            f"  2. Dataset folder exists: {DATASET_NAME}\n"
            f"  3. Images are organized in class folders\n"
            f"{'='*70}\n"
        )
    
    log(f"Scanning directory: {root_dir}")
    
    # Count total files first
    total_files = 0
    supported_formats = ('.png', '.jpg', '.jpeg', '.ppm', '.bmp')
    
    for root, dirs, files in os.walk(root_dir):
        # Skip label folders
        if 'label' in root.lower():
            continue
        
        img_files = [f for f in files if f.lower().endswith(supported_formats)]
        total_files += len(img_files)
    
    if total_files == 0:
        raise ValueError(
            f"\n{'='*70}\n"
            f"❌ ERROR: No images found!\n"
            f"{'='*70}\n"
            f"Searched in: {root_dir}\n"
            f"Supported formats: {', '.join(supported_formats)}\n\n"
            f"Possible issues:\n"
            f"  1. Images are in a subfolder (e.g., 'train', 'images')\n"
            f"  2. Wrong dataset name (check DATA_DIR variable)\n"
            f"  3. Images not organized in class folders\n"
            f"{'='*70}\n"
        )
    
    log(f"Found {total_files:,} total image files")
    
    # Load images with progress bar
    pbar = tqdm(total=total_files, desc="Loading images", unit="img")
    
    for root, dirs, files in os.walk(root_dir):
        # Skip label folders
        if 'label' in root.lower():
            continue
        
        img_files = [f for f in files if f.lower().endswith(supported_formats)]
        
        if not img_files:
            continue
        
        # Get class name from folder
        class_name = os.path.basename(root)
        
        for img_file in img_files:
            try:
                img_path = os.path.join(root, img_file)
                
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    pbar.write(f"⚠ Could not read: {img_file}")
                    pbar.update(1)
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, img_size)
                
                # Append
                images.append(img)
                labels.append(class_name)
                
                # Update class counts
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                pbar.update(1)
                
                # Check max_images limit
                if max_images and len(images) >= max_images:
                    pbar.close()
                    log(f"Reached max_images limit: {max_images}")
                    return np.array(images), np.array(labels)
                
            except Exception as e:
                pbar.write(f"⚠ Error loading {img_file}: {e}")
                pbar.update(1)
                continue
    
    pbar.close()
    
    if len(images) == 0:
        raise ValueError(
            f"\n{'='*70}\n"
            f"❌ ERROR: No images successfully loaded!\n"
            f"{'='*70}\n"
            f"All image files failed to load.\n"
            f"Check image file integrity.\n"
            f"{'='*70}\n"
        )
    
    log(f"\n✓ Successfully loaded {len(images):,} images from {len(class_counts)} classes")
    log("\nClass distribution:")
    for cls in sorted(class_counts.keys()):
        log(f"  {cls}: {class_counts[cls]:,} images")
    
    return np.array(images), np.array(labels)


# Try loading from multiple possible locations
try:
    possible_paths = [
        DATASET_PATH,
        os.path.join(DATASET_PATH, 'images'),
        os.path.join(DATASET_PATH, 'train'),
        os.path.join(DATASET_PATH, 'images', 'train'),
        os.path.join(DATA_DIR, 'Self-Driving_Cars', 'train', 'images'),
    ]
    
    X_all, y_all = None, None
    
    for path in possible_paths:
        if os.path.exists(path):
            log(f"Attempting to load from: {path}")
            try:
                X_all, y_all = load_images_from_folder(path, img_size=IMG_SIZE)
                if len(X_all) > 0:
                    log(f"✓ Successfully loaded from: {path}")
                    break
            except Exception as e:
                log(f"⚠ Failed: {e}")
                continue
    
    if X_all is None or len(X_all) == 0:
        raise ValueError("No images found in any expected location!")
    
    # Auto-discover classes
    class_names = sorted(list(set(y_all)))
    num_classes = len(class_names)
    
    log(f"\n✓ Dataset loaded successfully!")
    log(f"  Total images: {len(X_all):,}")
    log(f"  Image shape: {X_all[0].shape}")
    log(f"  Number of classes (auto-discovered): {num_classes}")
    log(f"  Class names: {class_names[:10]}{'...' if num_classes > 10 else ''}")
    
    # Check for class imbalance
    unique, counts = np.unique(y_all, return_counts=True)
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    
    log(f"\nClass balance check:")
    log(f"  Most common class: {max_count:,} images")
    log(f"  Least common class: {min_count:,} images")
    log(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 5:
        log(f"  ⚠️ WARNING: Significant class imbalance detected!")
        log(f"     Consider using class_weight parameter")
    else:
        log(f"  ✓ Dataset is reasonably balanced")
    
except Exception as e:
    log(f"\n{'='*70}")
    log(f"❌ FATAL ERROR: Could not load dataset")
    log(f"{'='*70}")
    log(f"Error: {e}")
    log(f"\nPlease check:")
    log(f"  1. Data folder exists: {DATA_DIR}")
    log(f"  2. Dataset folder exists: {DATASET_PATH}")
    log(f"  3. Images are organized in class subfolders")
    log(f"  4. Image files are not corrupted")
    log(f"{'='*70}")
    raise

print("=" * 70)


# ============================================================================
# STEP 2: SPLIT DATA (70% train, 15% val, 15% test)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: SPLITTING DATA")
print("=" * 70)

log("Splitting dataset into train/val/test...")

# First split: 85% temp (train+val), 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    stratify=y_all,
    random_state=42
)

log(f"✓ Test set created: {len(X_test):,} images ({TEST_SIZE*100:.0f}%)")

# Second split: 70% train, 15% val (of total)
# VAL_SIZE = 0.15 / 0.85 = 0.1765
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SIZE,
    stratify=y_temp,
    random_state=42
)

log(f"✓ Training set created: {len(X_train):,} images ({len(X_train)/len(X_all)*100:.1f}%)")
log(f"✓ Validation set created: {len(X_val):,} images ({len(X_val)/len(X_all)*100:.1f}%)")

# Verify split ratios
train_pct = len(X_train) / len(X_all) * 100
val_pct = len(X_val) / len(X_all) * 100
test_pct = len(X_test) / len(X_all) * 100

log(f"\nActual split ratios:")
log(f"  Training:   {len(X_train):,} images ({train_pct:.1f}%)")
log(f"  Validation: {len(X_val):,} images ({val_pct:.1f}%)")
log(f"  Test:       {len(X_test):,} images ({test_pct:.1f}%)")
log(f"  Total:      {len(X_all):,} images")

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

log(f"\n✓ Labels encoded:")
log(f"  Classes: {label_encoder.classes_[:10].tolist()}{'...' if num_classes > 10 else ''}")

# Compute class weights for handling imbalance
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_enc),
    y=y_train_enc
)
class_weight_dict = dict(enumerate(class_weights))

log(f"\n✓ Class weights computed (for handling imbalance)")
log(f"  Sample weights: {dict(list(class_weight_dict.items())[:5])}")

print("=" * 70)


# ============================================================================
# STEP 3: BUILD CNN FEATURE EXTRACTOR
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: BUILDING CNN FEATURE EXTRACTOR")
print("=" * 70)

log("Loading pretrained ResNet50...")

# Load pretrained ResNet50
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

log("✓ ResNet50 loaded (pretrained on ImageNet)")

# Fine-tuning strategy: Unfreeze last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
log(f"✓ Fine-tuning strategy: last {trainable_layers} layers unfrozen")

# Build model
x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = Dense(256, activation='relu', name='dense_256')(x)
x = Dropout(0.5, name='dropout')(x)
predictions = Dense(num_classes, activation='softmax', name='output')(x)

model = Model(inputs=base_model.input, outputs=predictions, name='resnet50_finetuned')

# Compile
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

total_params = model.count_params()
trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])

log(f"\n✓ Model compiled:")
log(f"  Total parameters: {total_params:,}")
log(f"  Trainable parameters: {trainable_params:,}")
log(f"  Frozen parameters: {total_params - trainable_params:,}")
log(f"  Optimizer: Adam (LR={LEARNING_RATE})")

print("=" * 70)


# ============================================================================
# STEP 4: FINE-TUNE CNN
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: FINE-TUNING CNN")
print("=" * 70)

log("Setting up callbacks...")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(RUN_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

log("✓ Callbacks configured:")
log("  - EarlyStopping (patience=3)")
log("  - ReduceLROnPlateau (patience=2)")
log("  - ModelCheckpoint (save best model)")

# Preprocess images
log("\nPreprocessing images...")
X_train_prep = preprocess_input(X_train.astype('float32'))
X_val_prep = preprocess_input(X_val.astype('float32'))

log("✓ Images preprocessed (ImageNet normalization)")

# Train
log(f"\nStarting training for {EPOCHS} epochs...")
log("(This may take 10-30 minutes depending on dataset size and hardware)")

start_time = time.time()

history = model.fit(
    X_train_prep, y_train_enc,
    validation_data=(X_val_prep, y_val_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,  # Handle class imbalance
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

log(f"\n✓ Training completed in {training_time/60:.1f} minutes ({training_time:.0f} seconds)")

# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val_prep, y_val_enc, verbose=0)
log(f"✓ Final validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'], linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Learning Rate', fontsize=11)
    plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

log(f"✓ Training history plot saved")

# Save the trained model
model.save(os.path.join(RUN_DIR, 'cnn_model_final.keras'))
log(f"✓ CNN model saved")

print("=" * 70)


# ============================================================================
# STEP 5: EXTRACT FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: EXTRACTING FEATURES FOR SVM")
print("=" * 70)

log("Creating feature extractor...")

# Create feature extractor (remove final classification layer)
feature_extractor = Model(
    inputs=model.input,
    outputs=model.get_layer('global_avg_pool').output,
    name='feature_extractor'
)

feature_dim = feature_extractor.output_shape[1]
log(f"✓ Feature extractor created")
log(f"  Feature dimension: {feature_dim}")

def extract_features_batched(images, batch_size=32, desc="Extracting"):
    """Extract features in batches to avoid memory issues."""
    features = []
    num_batches = int(np.ceil(len(images) / batch_size))
    
    for i in tqdm(range(num_batches), desc=desc, unit="batch"):
        batch = images[i*batch_size:(i+1)*batch_size]
        batch_prep = preprocess_input(batch.astype('float32'))
        batch_features = feature_extractor.predict(batch_prep, verbose=0)
        features.append(batch_features)
    
    return np.vstack(features)

# Extract features
log("\nExtracting features from all splits...")
log(f"  Using batch size: {BATCH_SIZE}")

X_train_features = extract_features_batched(X_train, BATCH_SIZE, "Train features")
log(f"  ✓ Training features: {X_train_features.shape}")

X_val_features = extract_features_batched(X_val, BATCH_SIZE, "Val features")
log(f"  ✓ Validation features: {X_val_features.shape}")

X_test_features = extract_features_batched(X_test, BATCH_SIZE, "Test features")
log(f"  ✓ Test features: {X_test_features.shape}")

# Save features
np.save(os.path.join(RUN_DIR, 'train_features.npy'), X_train_features)
np.save(os.path.join(RUN_DIR, 'val_features.npy'), X_val_features)
np.save(os.path.join(RUN_DIR, 'test_features.npy'), X_test_features)
log(f"\n✓ Features saved to disk")

# Save feature extractor model
feature_extractor.save(os.path.join(RUN_DIR, 'feature_extractor.keras'))
log(f"✓ Feature extractor model saved")

print("=" * 70)


# ============================================================================
# STEP 6: TRAIN SVM
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: TRAINING SVM CLASSIFIER")
print("=" * 70)

log("Standardizing features...")

# Standardize features (critical for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_val_scaled = scaler.transform(X_val_features)
X_test_scaled = scaler.transform(X_test_features)

log("✓ Features standardized (zero mean, unit variance)")
log(f"  Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")

# Train SVM
log(f"\nTraining SVM classifier...")
log(f"  Kernel: {SVM_KERNEL}")
log(f"  C: {SVM_C}")
log(f"  Gamma: {SVM_GAMMA}")
log("(This may take 5-20 minutes depending on dataset size)")

start_time = time.time()

svm = SVC(
    kernel=SVM_KERNEL,
    C=SVM_C,
    gamma=SVM_GAMMA,
    probability=True,  # Enable probability estimates for metrics
    random_state=42,
    verbose=False
)

svm.fit(X_train_scaled, y_train_enc)

svm_training_time = time.time() - start_time

log(f"\n✓ SVM trained in {svm_training_time/60:.1f} minutes ({svm_training_time:.0f} seconds)")
log(f"  Number of support vectors: {svm.n_support_.sum():,}")
log(f"  Support vectors per class: {svm.n_support_.tolist()[:10]}{'...' if num_classes > 10 else ''}")

# Validate on validation set
y_val_pred = svm.predict(X_val_scaled)
val_accuracy_svm = accuracy_score(y_val_enc, y_val_pred)
log(f"\n✓ Validation accuracy (SVM): {val_accuracy_svm:.4f} ({val_accuracy_svm*100:.2f}%)")

# Save SVM and scaler
joblib.dump(svm, os.path.join(RUN_DIR, 'svm_model.pkl'))
joblib.dump(scaler, os.path.join(RUN_DIR, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(RUN_DIR, 'label_encoder.pkl'))
log(f"✓ SVM model, scaler, and label encoder saved")

print("=" * 70)


# ============================================================================
# STEP 7: EVALUATE ON TEST SET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: FINAL EVALUATION ON TEST SET")
print("=" * 70)

log("Making predictions on test set...")

# Predictions
y_pred_enc = svm.predict(X_test_scaled)
y_pred_proba = svm.predict_proba(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_enc)

log("✓ Predictions completed")

# Calculate comprehensive metrics
log("\nComputing evaluation metrics...")
metrics = {}

# Basic accuracy
metrics['overall_accuracy'] = accuracy_score(y_test_enc, y_pred_enc)
metrics['top5_accuracy'] = top_k_accuracy_score(
    y_test_enc, y_pred_proba, k=min(5, num_classes)
)

log(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
log(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")

# Precision, Recall, F1
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_enc, y_pred_enc, average='macro', zero_division=0
)
p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
    y_test_enc, y_pred_enc, average='micro', zero_division=0
)

metrics['precision_macro'] = p_macro
metrics['recall_macro'] = r_macro
metrics['f1_macro'] = f1_macro
metrics['precision_micro'] = p_micro
metrics['recall_micro'] = r_micro
metrics['f1_micro'] = f1_micro

log(f"  Macro F1-Score: {f1_macro:.4f}")
log(f"  Micro F1-Score: {f1_micro:.4f}")

# AUC scores
try:
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test_enc, classes=range(num_classes))
    
    if num_classes == 2:
        # Binary classification
        metrics['roc_auc'] = roc_auc_score(y_test_bin, y_pred_proba[:, 1])
        metrics['pr_auc'] = average_precision_score(y_test_bin, y_pred_proba[:, 1])
    else:
        # Multi-class classification
        metrics['roc_auc'] = roc_auc_score(
            y_test_bin, y_pred_proba, average='macro', multi_class='ovr'
        )
        
        pr_aucs = [
            average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
            for i in range(num_classes)
        ]
        metrics['pr_auc'] = np.mean(pr_aucs)
    
    log(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    log(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    
except Exception as e:
    log(f"  ⚠ Could not compute AUC scores: {e}")
    metrics['roc_auc'] = 0.0
    metrics['pr_auc'] = 0.0

# Calibration metrics
try:
    if num_classes == 2:
        brier = brier_score_loss(y_test_bin, y_pred_proba[:, 1])
        metrics['brier_score'] = brier
    else:
        brier_scores = [
            brier_score_loss(y_test_bin[:, i], y_pred_proba[:, i])
            for i in range(num_classes)
        ]
        metrics['brier_score'] = np.mean(brier_scores)
    
    # Expected Calibration Error (ECE)
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_test_enc).astype(float)
    
    ece = 0.0
    n_bins = 10
    for i in range(n_bins):
        mask = (confidences > i/n_bins) & (confidences <= (i+1)/n_bins)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += np.abs(bin_acc - bin_conf) * bin_weight
    
    metrics['ece'] = ece
    
    log(f"  Expected Calibration Error: {ece:.4f}")
    log(f"  Brier Score: {metrics['brier_score']:.4f}")
    
except Exception as e:
    log(f"  ⚠ Could not compute calibration metrics: {e}")
    metrics['brier_score'] = 0.0
    metrics['ece'] = 0.0

# Efficiency metrics
log("\nMeasuring inference speed...")
test_sample = X_test[:min(100, len(X_test))]

start = time.time()
sample_features = extract_features_batched(test_sample, batch_size=BATCH_SIZE, desc="Speed test")
sample_scaled = scaler.transform(sample_features)
_ = svm.predict(sample_scaled)
elapsed = time.time() - start

metrics['latency_ms'] = (elapsed / len(test_sample)) * 1000
metrics['throughput_fps'] = len(test_sample) / elapsed

log(f"  Latency: {metrics['latency_ms']:.2f} ms per image")
log(f"  Throughput: {metrics['throughput_fps']:.2f} FPS")

# Training time metrics
metrics['cnn_training_time_min'] = training_time / 60
metrics['svm_training_time_min'] = svm_training_time / 60
metrics['total_training_time_min'] = (training_time + svm_training_time) / 60

log(f"\nTraining time summary:")
log(f"  CNN training: {metrics['cnn_training_time_min']:.1f} minutes")
log(f"  SVM training: {metrics['svm_training_time_min']:.1f} minutes")
log(f"  Total: {metrics['total_training_time_min']:.1f} minutes")

log("\n✓ All metrics computed successfully")

print("=" * 70)


# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: CREATING VISUALIZATIONS")
print("=" * 70)

log("Generating visualizations...")

# 1. Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred_enc)

fig, ax = plt.subplots(figsize=(12, 10))
if num_classes > 20:
    # Too many classes - show heatmap without labels
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, ax=ax)
    ax.set_title('Confusion Matrix (Large Number of Classes)', fontsize=14, fontweight='bold')
else:
    # Show with labels
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar=True, ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

log("  ✓ Confusion matrix saved")

# 2. Per-class Performance
precision, recall, f1, support = precision_recall_fscore_support(
    y_test_enc, y_pred_enc, zero_division=0
)

# Sort by support (most common classes first)
sorted_indices = np.argsort(support)[::-1]
top_n = min(15, num_classes)
top_indices = sorted_indices[:top_n]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(top_n)
width = 0.25

bars1 = ax.bar(x - width, precision[top_indices], width, label='Precision', alpha=0.8, color='steelblue')
bars2 = ax.bar(x, recall[top_indices], width, label='Recall', alpha=0.8, color='coral')
bars3 = ax.bar(x + width, f1[top_indices], width, label='F1-Score', alpha=0.8, color='mediumseagreen')

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Per-Class Performance (Top {top_n} Classes by Frequency)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_[top_indices], rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0.05:  # Only show if bar is visible
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=7)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

log("  ✓ Per-class metrics plot saved")

# 3. Class Distribution across splits
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (data, title, color) in zip(axes, [
    (y_train, 'Training Set', 'steelblue'),
    (y_val, 'Validation Set', 'coral'),
    (y_test, 'Test Set', 'mediumseagreen')
]):
    unique, counts = np.unique(data, return_counts=True)
    
    if num_classes <= 20:
        ax.bar(range(len(unique)), counts, alpha=0.7, color=color)
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(unique, rotation=45, ha='right', fontsize=8)
    else:
        ax.bar(range(len(unique)), counts, alpha=0.7, color=color, width=1.0)
    
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'{title}\n({len(data):,} images)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

log("  ✓ Class distribution plot saved")

# 4. Performance Summary Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

metric_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)', 'Top-5\nAccuracy']
metric_values = [
    metrics['overall_accuracy'],
    metrics['precision_macro'],
    metrics['recall_macro'],
    metrics['f1_macro'],
    metrics['top5_accuracy']
]

bars = ax.bar(metric_names, metric_values, alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, value in zip(bars, metric_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{value:.3f}\n({value*100:.1f}%)',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, 'performance_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

log("  ✓ Performance summary plot saved")

# 5. Top misclassifications
log("\nAnalyzing top misclassifications...")
misclassified_mask = y_test_enc != y_pred_enc
misclassified_indices = np.where(misclassified_mask)[0]

if len(misclassified_indices) > 0:
    # Get confidence scores for misclassifications
    misclassified_confidences = np.max(y_pred_proba[misclassified_indices], axis=1)
    
    # Sort by confidence (most confident mistakes first)
    sorted_misc_indices = misclassified_indices[np.argsort(misclassified_confidences)[::-1]]
    
    # Show top 10 misclassifications
    top_mistakes = min(10, len(sorted_misc_indices))
    
    log(f"  Top {top_mistakes} confident misclassifications:")
    for i, idx in enumerate(sorted_misc_indices[:top_mistakes], 1):
        true_class = label_encoder.classes_[y_test_enc[idx]]
        pred_class = label_encoder.classes_[y_pred_enc[idx]]
        confidence = np.max(y_pred_proba[idx])
        log(f"    {i}. True: {true_class}, Predicted: {pred_class}, Confidence: {confidence:.3f}")

log("\n✓ All visualizations created successfully")

print("=" * 70)


# ============================================================================
# STEP 9: RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY - HYBRID CNN-SVM (METHOD 3)")
print("=" * 70)

summary = f"""
{'='*70}
FINAL RESULTS - HYBRID CNN-SVM
{'='*70}

DATASET INFORMATION:
  Dataset: {DATASET_NAME}
  Total images: {len(X_all):,}
  Number of classes: {num_classes}
  Train/Val/Test split: {len(X_train):,}/{len(X_val):,}/{len(X_test):,} ({train_pct:.1f}%/{val_pct:.1f}%/{test_pct:.1f}%)

PERFORMANCE METRICS:
  Overall Accuracy:              {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)
  Top-5 Accuracy:                {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)
  
  Precision (Macro):             {metrics['precision_macro']:.4f}
  Recall (Macro):                {metrics['recall_macro']:.4f}
  F1-Score (Macro):              {metrics['f1_macro']:.4f}
  
  Precision (Micro):             {metrics['precision_micro']:.4f}
  Recall (Micro):                {metrics['recall_micro']:.4f}
  F1-Score (Micro):              {metrics['f1_micro']:.4f}
  
  ROC-AUC (Macro):               {metrics['roc_auc']:.4f}
  PR-AUC (Macro):                {metrics['pr_auc']:.4f}

CALIBRATION:
  Expected Calibration Error:    {metrics['ece']:.4f}
  Brier Score:                   {metrics['brier_score']:.4f}

EFFICIENCY:
  Latency per image:             {metrics['latency_ms']:.2f} ms
  Throughput:                    {metrics['throughput_fps']:.2f} FPS

TRAINING TIME:
  CNN training:                  {metrics['cnn_training_time_min']:.1f} minutes
  SVM training:                  {metrics['svm_training_time_min']:.1f} minutes
  Total training time:           {metrics['total_training_time_min']:.1f} minutes

MODEL DETAILS:
  CNN Architecture:              ResNet50 (pretrained on ImageNet)
  Feature dimension:             {feature_dim}
  SVM Kernel:                    {SVM_KERNEL}
  SVM C parameter:               {SVM_C}
  Number of support vectors:     {svm.n_support_.sum():,}

{'='*70}
"""

print(summary)
log(summary, also_print=False)

print("=" * 70)


# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: SAVING RESULTS")
print("=" * 70)

log("Saving results to files...")

# 1. Save metrics as CSV
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(RUN_DIR, 'metrics.csv'), index=False)
log("  ✓ metrics.csv")

# 2. Save classification report
report = classification_report(
    y_test_enc, y_pred_enc,
    target_names=[str(c) for c in label_encoder.classes_],
    digits=4
)

report_path = os.path.join(RUN_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORT - HYBRID CNN-SVM\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Total images: {len(X_all):,}\n")
    f.write(f"Number of classes: {num_classes}\n")
    f.write(f"Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}\n\n")
    f.write("=" * 70 + "\n")
    f.write(report)
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}\n")
    f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n")
    f.write("=" * 70 + "\n")

log("  ✓ classification_report.txt")

# 3. Save per-class detailed metrics
per_class_df = pd.DataFrame({
    'class': label_encoder.classes_,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'support': support
})
per_class_df = per_class_df.sort_values('support', ascending=False)
per_class_df.to_csv(os.path.join(RUN_DIR, 'per_class_metrics.csv'), index=False)
log("  ✓ per_class_metrics.csv")

# 4. Save configuration
config = {
    'dataset_name': DATASET_NAME,
    'dataset_path': DATASET_PATH,
    'img_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'test_size': TEST_SIZE,
    'val_size': VAL_SIZE,
    'svm_kernel': SVM_KERNEL,
    'svm_c': SVM_C,
    'svm_gamma': SVM_GAMMA,
    'num_classes': num_classes,
    'total_images': len(X_all),
    'train_images': len(X_train),
    'val_images': len(X_val),
    'test_images': len(X_test)
}

config_df = pd.DataFrame([config])
config_df.to_csv(os.path.join(RUN_DIR, 'config.csv'), index=False)
log("  ✓ config.csv")

# 5. Save predictions for analysis
predictions_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'correct': y_test == y_pred,
    'confidence': np.max(y_pred_proba, axis=1)
})
predictions_df.to_csv(os.path.join(RUN_DIR, 'predictions.csv'), index=False)
log("  ✓ predictions.csv")

# 6. Create a summary README
readme_path = os.path.join(RUN_DIR, 'README.txt')
with open(readme_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("HYBRID CNN-SVM RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Run timestamp: {TIMESTAMP}\n")
    f.write(f"Dataset: {DATASET_NAME}\n\n")
    f.write("FILES IN THIS DIRECTORY:\n")
    f.write("  - best_model.keras          : Best CNN model during training\n")
    f.write("  - cnn_model_final.keras     : Final CNN model after training\n")
    f.write("  - feature_extractor.keras   : Feature extractor (CNN without head)\n")
    f.write("  - svm_model.pkl             : Trained SVM classifier\n")
    f.write("  - scaler.pkl                : Feature scaler (StandardScaler)\n")
    f.write("  - label_encoder.pkl         : Label encoder\n")
    f.write("  - train_features.npy        : Extracted training features\n")
    f.write("  - val_features.npy          : Extracted validation features\n")
    f.write("  - test_features.npy         : Extracted test features\n")
    f.write("  - metrics.csv               : All evaluation metrics\n")
    f.write("  - classification_report.txt : Detailed classification report\n")
    f.write("  - per_class_metrics.csv     : Per-class performance\n")
    f.write("  - predictions.csv           : All test predictions\n")
    f.write("  - config.csv                : Training configuration\n")
    f.write("  - training_history.png      : Training curves\n")
    f.write("  - confusion_matrix.png      : Confusion matrix\n")
    f.write("  - per_class_metrics.png     : Per-class performance plot\n")
    f.write("  - class_distribution.png    : Class distribution across splits\n")
    f.write("  - performance_summary.png   : Overall performance summary\n")
    f.write("  - training_log.txt          : Complete training log\n")
    f.write("\n" + "=" * 70 + "\n")
    f.write("KEY RESULTS:\n")
    f.write(f"  Accuracy: {metrics['overall_accuracy']*100:.2f}%\n")
    f.write(f"  F1-Score: {metrics['f1_macro']:.4f}\n")
    f.write(f"  Training time: {metrics['total_training_time_min']:.1f} minutes\n")
    f.write("=" * 70 + "\n")

log("  ✓ README.txt")

log(f"\n✓ All results saved to: {RUN_DIR}")

print("=" * 70)


# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 70)
print("✓✓✓ TRAINING COMPLETE! ✓✓✓")
print("=" * 70)

completion_message = f"""
🎉 SUCCESS! Training completed successfully!

📊 Final Results:
   Accuracy: {metrics['overall_accuracy']*100:.2f}%
   F1-Score: {metrics['f1_macro']:.4f}
   Training Time: {metrics['total_training_time_min']:.1f} minutes

📁 Results saved to:
   {RUN_DIR}

📄 Key files:
   - cnn_model_final.keras (trained CNN)
   - svm_model.pkl (trained SVM)
   - metrics.csv (all metrics)
   - classification_report.txt (detailed report)
   - *.png (visualizations)

🔍 Next steps:
   1. Review visualizations in the results folder
   2. Check classification_report.txt for per-class performance
   3. Compare with other methods (Method 1 & 2)
   4. Use for predictions on new images

💾 To use the trained model:
   - Load feature_extractor.keras
   - Load svm_model.pkl and scaler.pkl
   - Extract features from new images
   - Predict with SVM
"""

print(completion_message)
log(completion_message, also_print=False)

log("=" * 70)
log("Training log completed")
log("=" * 70)

print("=" * 70)
print("All done! Check the results folder for detailed outputs.")
print("=" * 70)