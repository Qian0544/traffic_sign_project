# ============================================================================
# SETUP
# ============================================================================

print("Installing packages...")
!pip install -q scikit-learn opencv-python seaborn tqdm

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
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import os
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check GPU
print("\n" + "="*70)
if tf.config.list_physical_devices('GPU'):
    print("✓ GPU Available!")
else:
    print("⚠ Using CPU (slower)")
print("="*70)

np.random.seed(42)
tf.random.set_seed(42)

print("\n" + "="*70)
print("HYBRID CNN-SVM WITH TWO DATASETS")
print("="*70)


# ============================================================================
# STEP 1: LOAD DATASETS (FROM GOOGLE DRIVE)
# ============================================================================


print("\n" + "="*70)
print("STEP 1: LOAD DATASETS FROM GOOGLE DRIVE")
print("="*70)



# === 🔧 EDIT THESE PATHS TO MATCH YOUR FILE LOCATIONS ===
finetune_zip = '/content/drive/MyDrive/Self-Driving_Cars.zip'
main_zip     = '/content/drive/MyDrive/Traffic_Sign_Detection.zip'
# =========================================================

# Check that both exist
if not os.path.exists(finetune_zip):
    raise FileNotFoundError(f"❌ Fine-tuning dataset not found: {finetune_zip}")
if not os.path.exists(main_zip):
    raise FileNotFoundError(f"❌ Main dataset not found: {main_zip}")

print(f"\n✓ Fine-tuning dataset: {finetune_zip}")
print(f"✓ Main dataset: {main_zip}")
print("="*70)

# ============================================================================
# EXTRACT DATASETS
# ============================================================================

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    import shutil
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

print("\n" + "="*70)
print("EXTRACTING DATASETS")
print("="*70)

print("Extracting fine-tuning dataset...")
finetune_dir = extract_zip(finetune_zip, 'finetune_data')
print(f"  ✓ Extracted to: {finetune_dir}")

print("Extracting main dataset...")
main_dir = extract_zip(main_zip, 'main_data')
print(f"  ✓ Extracted to: {main_dir}")
print("="*70)

# ============================================================================
# LOAD IMAGES FROM FOLDERS
# ============================================================================

def load_images_from_folders(root_dir, img_size=(160, 160), max_per_folder=None):
    """Load images from any folder structure."""
    images = []
    labels = []

    # Find all image files
    for root, dirs, files in os.walk(root_dir):
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if img_files:
            folder_name = os.path.basename(root)
            img_files = img_files[:max_per_folder]

            for img_file in img_files:
                img_path = os.path.join(root, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, img_size)
                        images.append(img)
                        labels.append(folder_name)
                except:
                    continue

    return np.array(images), np.array(labels)


# ============================================================================
# STEP 2: LOAD FINE-TUNING DATASET (YOLO FORMAT)
# ============================================================================

print("\n" + "="*70)
print("STEP 2: LOADING FINE-TUNING DATASET (YOLO FORMAT)")
print("="*70)

def load_yolo_dataset(base_dir, img_size=(160, 160)):
    """
    Load images and class labels from YOLO dataset.
    Assumes structure:
        base_dir/
            images/
            labels/
    Only uses the first class in the label file per image.
    """
    images = []
    labels = []

    img_dir = os.path.join(base_dir, 'images')
    lbl_dir = os.path.join(base_dir, 'labels')

    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith('.jpg'):
            continue

        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(lbl_dir, img_file.replace('.jpg', '.txt'))

        # read first class from YOLO label file
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                class_id = int(line.split()[0]) if line else 0
        else:
            class_id = 0  # fallback if no label

        # load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)

        images.append(img)
        labels.append(class_id)

    return np.array(images), np.array(labels)

finetune_dir = 'finetune_data/Self-Driving Cars.v6-version-4-prescan-416x416.yolov11'


# Load train set
train_dir = os.path.join(finetune_dir, 'train')
ft_train_images, ft_train_labels = load_yolo_dataset(train_dir)

# Load validation set
valid_dir = os.path.join(finetune_dir, 'valid')
ft_val_images, ft_val_labels = load_yolo_dataset(valid_dir)

print(f"\n✓ Fine-tuning dataset loaded:")
print(f"  Training: {len(ft_train_images):,} images")
print(f"  Validation: {len(ft_val_images):,} images")
print("="*70)


# ============================================================================
# STEP 3: FINE-TUNE CNN
# ============================================================================

print("\n" + "="*70)
print("STEP 3: FINE-TUNING CNN")
print("="*70)

# Prepare labels
label_encoder_ft = LabelEncoder()
y_train_ft = label_encoder_ft.fit_transform(ft_train_labels)
y_val_ft = label_encoder_ft.transform(ft_val_labels)
num_classes_ft = len(label_encoder_ft.classes_)

y_train_categorical = tf.keras.utils.to_categorical(y_train_ft, num_classes_ft)
y_val_categorical = tf.keras.utils.to_categorical(y_val_ft, num_classes_ft)

print(f"Number of classes: {num_classes_ft}")

# Build model
print("Building model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

# Freeze early layers, unfreeze last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes_ft, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built")

# Train
print("\nFine-tuning (5 epochs)...")
history = model.fit(
    preprocess_input(ft_train_images.astype('float32')),
    y_train_categorical,
    validation_data=(preprocess_input(ft_val_images.astype('float32')), y_val_categorical),
    epochs=5,
    batch_size=32,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=1
)

val_loss, val_acc = model.evaluate(
    preprocess_input(ft_val_images.astype('float32')),
    y_val_categorical,
    verbose=0
)

print(f"\n✓ Fine-tuning complete!")
print(f"  Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# Extract feature extractor
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-4].output  # GlobalAveragePooling2D
)
print(f"✓ Feature extractor ready (dimension: {feature_extractor.output_shape[1]})")
print("="*70)


# ============================================================================
# STEP 4: LOAD MAIN DATASET
# ============================================================================

print("\n" + "="*70)
print("STEP 4: LOADING MAIN DATASET")
print("="*70)

# Load training images
print("Loading training images...")
train_images, train_labels = load_images_from_folders(
    os.path.join(main_dir, 'training') if os.path.exists(os.path.join(main_dir, 'training'))
    else os.path.join(main_dir, 'images/training') if os.path.exists(os.path.join(main_dir, 'images/training'))
    else main_dir,
    max_per_folder=None
)

# Load validation images
print("Loading validation images...")
val_images, val_labels = load_images_from_folders(
    os.path.join(main_dir, 'validation') if os.path.exists(os.path.join(main_dir, 'validation'))
    else os.path.join(main_dir, 'images/validation') if os.path.exists(os.path.join(main_dir, 'images/validation'))
    else main_dir,
    max_per_folder=None
)

# Combine and split
all_images = np.concatenate([train_images, val_images])
all_labels = np.concatenate([train_labels, val_labels])

# Split: 70% train, 10% val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
)

class_names = sorted(list(set(all_labels)))

print(f"\n✓ Main dataset loaded:")
print(f"  Training: {len(X_train):,} images (70%)")
print(f"  Validation: {len(X_val):,} images (10%)")
print(f"  Test: {len(X_test):,} images (20%)")
print(f"  Classes: {len(class_names)}")
print("="*70)


# ============================================================================
# STEP 5: EXTRACT FEATURES
# ============================================================================

print("\n" + "="*70)
print("STEP 5: EXTRACTING FEATURES")
print("="*70)

def extract_features(images):
    preprocessed = preprocess_input(images.astype('float32'))
    return feature_extractor.predict(preprocessed, batch_size=32, verbose=0)

print("Extracting training features...")
X_train_features = extract_features(X_train)
print(f"  ✓ Shape: {X_train_features.shape}")

print("Extracting validation features...")
X_val_features = extract_features(X_val)
print(f"  ✓ Shape: {X_val_features.shape}")

print("="*70)


# ============================================================================
# STEP 6: TRAIN SVM
# ============================================================================

print("\n" + "="*70)
print("STEP 6: TRAINING SVM")
print("="*70)

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_val_scaled = scaler.transform(X_val_features)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train_enc)

y_val_pred = svm.predict(X_val_scaled)
val_acc = accuracy_score(y_val_enc, y_val_pred)

print(f"✓ SVM trained!")
print(f"  Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print("="*70)


# ============================================================================
# STEP 7: EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*70)
print("STEP 7: EVALUATING ON TEST SET")
print("="*70)

# Extract test features
print("Extracting test features...")
X_test_features = extract_features(X_test)
X_test_scaled = scaler.transform(X_test_features)

# Predict
y_test_enc = label_encoder.transform(y_test)
y_pred_enc = svm.predict(X_test_scaled)
y_pred_proba = svm.predict_proba(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_enc)

print("Computing metrics...")
metrics = {}

# Accuracy
metrics['overall_accuracy'] = accuracy_score(y_test, y_pred)
metrics['top5_accuracy'] = top_k_accuracy_score(y_test_enc, y_pred_proba, k=min(5, len(class_names)))

# Precision, Recall, F1
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)

metrics['precision_macro'] = p_macro
metrics['recall_macro'] = r_macro
metrics['f1_macro'] = f1_macro
metrics['precision_micro'] = p_micro
metrics['recall_micro'] = r_micro
metrics['f1_micro'] = f1_micro

# AUC
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test_enc, classes=range(len(class_names)))

metrics['roc_auc'] = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
pr_aucs = [average_precision_score(y_test_bin[:, i], y_pred_proba[:, i]) for i in range(len(class_names))]
metrics['pr_auc'] = np.mean(pr_aucs)

# Calibration
brier_scores = [brier_score_loss(y_test_bin[:, i], y_pred_proba[:, i]) for i in range(len(class_names))]
metrics['brier_score'] = np.mean(brier_scores)

confidences = np.max(y_pred_proba, axis=1)
predictions = np.argmax(y_pred_proba, axis=1)
accuracies = (predictions == y_test_enc)

ece = 0.0
for i in range(10):
    in_bin = (confidences > i/10) & (confidences <= (i+1)/10)
    if np.mean(in_bin) > 0:
        ece += np.abs(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin])) * np.mean(in_bin)
metrics['ece'] = ece

# Efficiency
import time
start = time.time()
_ = extract_features(X_test[:100])
_ = svm.predict(scaler.transform(_))
elapsed = time.time() - start

metrics['latency_ms'] = (elapsed / 100) * 1000
metrics['throughput_fps'] = 100 / elapsed

print("✓ All metrics computed")
print("="*70)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("STEP 8: CREATING VISUALIZATIONS")
print("="*70)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
if len(class_names) > 20:
    sns.heatmap(cm, annot=False, cmap='Blues')
else:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 2. Per-class metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
num_show = min(15, len(class_names))

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(num_show)
width = 0.25
ax.bar(x - width, precision[:num_show], width, label='Precision', alpha=0.8)
ax.bar(x, recall[:num_show], width, label='Recall', alpha=0.8)
ax.bar(x + width, f1[:num_show], width, label='F1', alpha=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_title('Per-Class Performance', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names[:num_show], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('per_class_metrics.png', dpi=300)
plt.show()

print("✓ Visualizations created")
print("="*70)


# ============================================================================
# RESULTS TABLE
# ============================================================================

print("\n" + "="*70)
print("RESULTS - HYBRID CNN-SVM")
print("="*70)
print(f"\n{'Metric':<45} {'Value':<20}")
print("-" * 70)
print(f"{'Overall Accuracy':<45} {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
print(f"{'Top-5 Accuracy':<45} {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
print()
print(f"{'Precision (Macro)':<45} {metrics['precision_macro']:.4f}")
print(f"{'Recall (Macro)':<45} {metrics['recall_macro']:.4f}")
print(f"{'PR-AUC (Macro)':<45} {metrics['pr_auc']:.4f}")
print()
print(f"{'F1-Score (Macro)':<45} {metrics['f1_macro']:.4f}")
print(f"{'F1-Score (Micro)':<45} {metrics['f1_micro']:.4f}")
print()
print(f"{'ROC-AUC (Macro)':<45} {metrics['roc_auc']:.4f}")
print()
print(f"{'Expected Calibration Error (ECE)':<45} {metrics['ece']:.4f}")
print(f"{'Brier Score':<45} {metrics['brier_score']:.4f}")
print()
print(f"{'Latency (ms per image)':<45} {metrics['latency_ms']:.2f} ms")
print(f"{'Throughput (FPS)':<45} {metrics['throughput_fps']:.2f} FPS")
print("="*70)


# ============================================================================
# SAVE & DOWNLOAD
# ============================================================================

print("\n" + "="*70)
print("STEP 9: SAVING RESULTS")
print("="*70)

# Save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('metrics.csv', index=False)
print("✓ Metrics saved")

# Classification report
report = classification_report(y_test, y_pred, target_names=class_names)
with open('classification_report.txt', 'w') as f:
    f.write("HYBRID CNN-SVM RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write("CNN fine-tuned on: Self-Driving Cars dataset\n")
    f.write("SVM trained on: Traffic Sign Detection dataset\n\n")
    f.write(report)
print("✓ Report saved")

print("="*70)

print("\n" + "="*70)
print("✓✓✓ COMPLETE! ✓✓✓")
print("="*70)
print(f"\n📊 Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
print(f"📊 Macro F1-Score: {metrics['f1_macro']:.4f}")
print("\n✓ All files downloaded")
print("="*70)