"""EuroSAT Land-Use Classification Training and Evaluation Pipeline.

Supports:
1. Baseline CNN (reproducible implementation of the original repository's architecture)
2. MobileNetV2 Transfer Learning (ImageNet pretrained feature extractor)

Features:
- Deterministic 70/15/15 stratified train/val/test split (no data leakage).
- Rotational data augmentation suitable for nadir satellite imagery (train only).
- Complete evaluation on held-out test set: Accuracy, F1-score, Precision, Recall.
- Auto-generates Confusion Matrix, Learning Curves, and structured JSON metrics.
"""

import argparse
import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

EXPECTED_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def set_reproducible_seeds(seed=42):
    """Ensure consistent execution across runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def discover_image_dataset(data_dir, sample_fraction=1.0, seed=42):
    """Collect image paths and numerical class labels with optional sub-sampling."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found at: '{data_dir}'.\n"
            f"Please run 'python download_data.py' first to download the EuroSAT dataset."
        )

    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        raise ValueError(f"No class folders found in '{data_dir}'.")

    filepaths = []
    labels = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        folder = os.path.join(data_dir, class_name)
        images = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        filepaths.extend(images)
        labels.extend([class_to_idx[class_name]] * len(images))

    filepaths = np.array(filepaths)
    labels = np.array(labels)

    if sample_fraction < 1.0:
        # Stratified subsampling for fast verification / CPU budget
        _, filepaths, _, labels = train_test_split(
            filepaths, labels,
            test_size=sample_fraction,
            stratify=labels,
            random_state=seed
        )
        print(f"[INFO] Subsampled dataset to {sample_fraction * 100:.1f}%: {len(filepaths)} total images.")

    print(f"[INFO] Discovered {len(filepaths)} images across {len(class_names)} classes.")
    return filepaths, labels, class_names


def create_stratified_splits(filepaths, labels, seed=42):
    """Create reproducible 70% Train, 15% Validation, 15% Test splits."""
    # Step 1: 70% Train, 30% Temp (Val + Test)
    train_files, temp_files, train_y, temp_y = train_test_split(
        filepaths, labels,
        test_size=0.30,
        stratify=labels,
        random_state=seed
    )

    # Step 2: Split 30% Temp evenly into 15% Validation and 15% Test
    val_files, test_files, val_y, test_y = train_test_split(
        temp_files, temp_y,
        test_size=0.50,
        stratify=temp_y,
        random_state=seed
    )

    print(f"[INFO] Split summary: Train={len(train_files)} (70%), Val={len(val_files)} (15%), Test={len(test_files)} (15%)")
    return (train_files, train_y), (val_files, val_y), (test_files, test_y)


def parse_image(filepath, label, img_size=(64, 64)):
    """Read and decode JPEG image."""
    img_bytes = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    return img, label


def augment_satellite_image(img, label):
    """Data augmentation tailored for nadir satellite imagery (orientation invariance)."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # 90-degree rotations preserve exact pixel coordinates without interpolation artifacts
    rot_k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=rot_k)
    return img, label


def build_tf_dataset(files, labels, batch_size=32, is_training=False, img_size=(64, 64), seed=42):
    """Build high-performance tf.data input pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(files), 10000), seed=seed)

    ds = ds.map(lambda f, l: parse_image(f, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.map(augment_satellite_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_baseline_cnn(input_shape=(64, 64, 3), num_classes=10):
    """Custom baseline CNN architecture from the original repository."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name="baseline_cnn")
    return model


def build_mobilenetv2(input_shape=(64, 64, 3), num_classes=10):
    """Transfer learning using MobileNetV2 pretrained on ImageNet."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the pretrained weights initially to preserve feature extractor representations
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    # MobileNetV2 expects [-1, 1] inputs:
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer")
    return model


def plot_learning_curves(history, output_path, model_name):
    """Plot and save training and validation loss and accuracy curves."""
    epochs = range(1, len(history.history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history.history['loss'], 'o-', label='Training Loss', color='#2563eb')
    axes[0].plot(epochs, history.history['val_loss'], 's--', label='Validation Loss', color='#dc2626')
    axes[0].set_title(f'{model_name}: Loss vs. Epochs', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Cross-Entropy)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(epochs, history.history['accuracy'], 'o-', label='Training Accuracy', color='#2563eb')
    axes[1].plot(epochs, history.history['val_accuracy'], 's--', label='Validation Accuracy', color='#16a34a')
    axes[1].set_title(f'{model_name}: Accuracy vs. Epochs', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved learning curves to: {output_path}")


def plot_confusion_matrix(cm, class_names, output_path, model_name):
    """Plot both raw count and normalized confusion matrices."""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar=False
    )
    axes[0].set_title(f"{model_name}: Confusion Matrix (Raw Counts)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].set_ylabel("True Class", fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    # Normalized percentages
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Greens',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar=True
    )
    axes[1].set_title(f"{model_name}: Normalized Confusion Matrix (Recall)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].set_ylabel("True Class", fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved confusion matrix visualization to: {output_path}")


def evaluate_model(model, test_ds, test_y, class_names, output_dir, model_name):
    """Compute comprehensive test set metrics and classification report."""
    print(f"\n[INFO] Evaluating {model_name} on held-out test set ({len(test_y)} samples)...")
    start_time = time.time()
    y_probs = model.predict(test_ds, verbose=1)
    eval_time = time.time() - start_time
    y_pred = np.argmax(y_probs, axis=1)

    # Metrics
    acc = float(np.mean(y_pred == test_y))
    macro_f1 = float(f1_score(test_y, y_pred, average='macro'))
    weighted_f1 = float(f1_score(test_y, y_pred, average='weighted'))
    macro_precision = float(precision_score(test_y, y_pred, average='macro', zero_division=0))
    macro_recall = float(recall_score(test_y, y_pred, average='macro', zero_division=0))

    report_str = classification_report(test_y, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 65)
    print(f"TEST CLASSIFICATION REPORT: {model_name}")
    print("=" * 65)
    print(report_str)

    # Save classification report text
    report_file = os.path.join(output_dir, f"classification_report_{model_name}.txt")
    with open(report_file, "w") as f:
        f.write(f"EuroSAT Satellite Classification - {model_name} Test Report\n")
        f.write("=" * 65 + "\n")
        f.write(report_str)
        f.write(f"\nInference Time for {len(test_y)} images: {eval_time:.2f}s ({eval_time/len(test_y)*1000:.2f} ms/sample)\n")

    # Confusion matrix
    cm = confusion_matrix(test_y, y_pred)
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(cm, class_names, cm_path, model_name)

    # Save metrics JSON
    metrics = {
        "model_name": model_name,
        "test_samples": int(len(test_y)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "inference_latency_ms_per_sample": round((eval_time / len(test_y)) * 1000, 2),
        "per_class": classification_report(test_y, y_pred, target_names=class_names, output_dict=True)
    }
    json_path = os.path.join(output_dir, f"metrics_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved test metrics JSON to: {json_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EuroSAT Satellite Image Classification Training")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mobilenetv2"],
                        help="Model architecture to train (default: cnn)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of dataset to use (1.0 = all 27,000 images; e.g. 0.2 for quick smoke tests)")
    parser.add_argument("--data_dir", type=str, default="dataset/EuroSAT_RGB", help="Path to EuroSAT RGB images")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save models and plots")
    args = parser.parse_args()

    set_reproducible_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"EuroSAT Satellite Classification Training: {args.model.upper()}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Seed: {args.seed}")
    print(f"=======================================================\n")

    # 1. Discover dataset
    filepaths, labels, class_names = discover_image_dataset(
        args.data_dir, sample_fraction=args.sample_fraction, seed=args.seed
    )

    # 2. Stratified 70/15/15 split
    (train_files, train_y), (val_files, val_y), (test_files, test_y) = create_stratified_splits(
        filepaths, labels, seed=args.seed
    )

    # Save class names mapping for inference
    classes_path = os.path.join(args.output_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)

    # 3. Build tf.data pipelines
    train_ds = build_tf_dataset(train_files, train_y, batch_size=args.batch_size, is_training=True, seed=args.seed)
    val_ds = build_tf_dataset(val_files, val_y, batch_size=args.batch_size, is_training=False, seed=args.seed)
    test_ds = build_tf_dataset(test_files, test_y, batch_size=args.batch_size, is_training=False, seed=args.seed)

    # 4. Build selected model
    if args.model == "cnn":
        model = build_baseline_cnn(input_shape=(64, 64, 3), num_classes=len(class_names))
    elif args.model == "mobilenetv2":
        model = build_mobilenetv2(input_shape=(64, 64, 3), num_classes=len(class_names))

    model.summary()

    # 5. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Callbacks
    best_model_path = os.path.join(args.output_dir, f"{args.model}_best.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]

    # 7. Train
    print("\n[INFO] Starting training...")
    start_train_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    total_train_time = time.time() - start_train_time
    print(f"\n[INFO] Training finished in {total_train_time:.1f}s ({total_train_time/60:.2f} mins).")

    # 8. Plot and save learning curves
    curves_path = os.path.join(args.output_dir, f"learning_curves_{args.model}.png")
    plot_learning_curves(history, curves_path, args.model.upper())

    # 9. Evaluate on test set
    metrics = evaluate_model(model, test_ds, test_y, class_names, args.output_dir, args.model.upper())
    metrics["training_time_seconds"] = round(total_train_time, 1)

    # Re-save metrics with training time
    json_path = os.path.join(args.output_dir, f"metrics_{args.model.upper()}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save as legacy h5 format for backwards compatibility if needed
    h5_path = os.path.join(args.output_dir, f"landuse_{args.model}.h5")
    try:
        model.save(h5_path)
        print(f"[INFO] Saved legacy h5 weights to: {h5_path}")
    except Exception as e:
        print(f"[WARN] Could not save h5: {e}")

    print(f"\n[SUCCESS] Pipeline completed successfully for model: {args.model.upper()}")
    print(f"           Test Accuracy: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}\n")


if __name__ == "__main__":
    main()
