"""EuroSAT Model Evaluation Script.

Evaluates a saved model (.keras or .h5) on the held-out test split of EuroSAT.
Generates:
- Test Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall
- Full Classification Report (per-class breakdown)
- Raw & Normalized Confusion Matrix heatmaps
- Structured JSON metrics file
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

EXPECTED_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def load_test_split(data_dir, seed=42):
    """Recreate the exact deterministic 15% test split."""
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
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

    # 70% Train, 30% Temp
    _, temp_files, _, temp_y = train_test_split(
        filepaths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    # Split 30% Temp into 15% Val and 15% Test
    _, test_files, _, test_y = train_test_split(
        temp_files, temp_y, test_size=0.50, stratify=temp_y, random_state=seed
    )

    print(f"[INFO] Recreated deterministic test split: {len(test_files)} images across {len(class_names)} classes.")
    return test_files, test_y, class_names


def parse_image(filepath, label, img_size=(64, 64)):
    img_bytes = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    return img, label


def build_test_dataset(files, labels, batch_size=32, img_size=(64, 64)):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(lambda f, l: parse_image(f, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def plot_confusion_matrix(cm, class_names, output_path, title):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
    axes[0].set_title(f"{title}: Confusion Matrix (Raw)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True", fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=True)
    axes[1].set_title(f"{title}: Normalized (Recall)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("True", fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved confusion matrix to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EuroSAT Model Independent Evaluator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .keras or .h5 model")
    parser.add_argument("--data_dir", type=str, default="dataset/EuroSAT_RGB", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for dataset splitting")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model (e.g., CNN or MOBILENETV2)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    raw_name = os.path.splitext(os.path.basename(args.model_path))[0]
    if args.model_name:
        model_name = args.model_name.upper()
    elif "cnn" in raw_name.lower():
        model_name = "CNN"
    elif "mobilenet" in raw_name.lower():
        model_name = "MOBILENETV2"
    else:
        model_name = raw_name.upper()

    print(f"\n[INFO] Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    test_files, test_y, class_names = load_test_split(args.data_dir, seed=args.seed)
    test_ds = build_test_dataset(test_files, test_y, batch_size=args.batch_size)

    print(f"[INFO] Running inference on {len(test_y)} test samples...")
    t0 = time.time()
    y_probs = model.predict(test_ds, verbose=1)
    latency = time.time() - t0
    y_pred = np.argmax(y_probs, axis=1)

    acc = float(np.mean(y_pred == test_y))
    macro_f1 = float(f1_score(test_y, y_pred, average='macro'))
    weighted_f1 = float(f1_score(test_y, y_pred, average='weighted'))
    macro_p = float(precision_score(test_y, y_pred, average='macro', zero_division=0))
    macro_r = float(recall_score(test_y, y_pred, average='macro', zero_division=0))

    report_str = classification_report(test_y, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 65)
    print(report_str)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Avg Latency:   {(latency / len(test_y)) * 1000:.2f} ms/sample")

    # Save classification report text
    report_file = os.path.join(args.output_dir, f"classification_report_{model_name}.txt")
    with open(report_file, "w") as f:
        f.write(f"EuroSAT Satellite Classification - {model_name} Test Report\n")
        f.write("=" * 65 + "\n")
        f.write(report_str)
        f.write(f"\nInference Time for {len(test_y)} images: {latency:.2f}s ({(latency/len(test_y))*1000:.2f} ms/sample)\n")

    # Confusion matrix
    cm = confusion_matrix(test_y, y_pred)
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(cm, class_names, cm_path, model_name)

    metrics = {
        "model_name": model_name,
        "model_file": args.model_path,
        "test_samples": len(test_y),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "inference_latency_ms_per_sample": round((latency / len(test_y)) * 1000, 2),
        "per_class": classification_report(test_y, y_pred, target_names=class_names, output_dict=True)
    }
    json_path = os.path.join(args.output_dir, f"metrics_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SUCCESS] Metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
