"""EuroSAT Satellite Image Inference Pipeline.

Takes a single satellite image (or samples a random test image), runs inference
using the trained deep learning model, and outputs top class predictions with confidence scores.
"""

import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

DEFAULT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def load_classes(classes_path):
    """Load class label mappings from JSON, or use defaults."""
    if os.path.isfile(classes_path):
        with open(classes_path, "r") as f:
            return json.load(f)
    return DEFAULT_CLASSES


def preprocess_image(image_path, target_size=(64, 64)):
    """Load and prepare an image for inference."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    pil_resized = pil_img.resize(target_size)
    img_array = np.array(pil_resized, dtype=np.float32)
    # Expand to batch dimension: (1, 64, 64, 3)
    batch_img = np.expand_dims(img_array, axis=0)
    return batch_img, pil_img


def predict(image_path, model_path, classes_path="outputs/classes.json", top_k=3):
    """Run model prediction on a single image."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}.\n"
            f"Please train a model first using 'python train.py'."
        )

    class_names = load_classes(classes_path)
    model = tf.keras.models.load_model(model_path)
    batch_img, pil_img = preprocess_image(image_path)

    # Note: Model architectures in train.py (Rescaling / preprocess_input)
    # are built into the model pipeline directly, accepting raw [0, 255] float arrays.
    probabilities = model.predict(batch_img, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "rank": rank,
            "class": class_names[idx],
            "confidence": float(probabilities[idx]),
            "percentage": f"{probabilities[idx] * 100:.2f}%"
        })

    return results, pil_img, class_names


def display_prediction(image_path, results, pil_img, save_plot=None):
    """Print readable terminal output and optionally save visualization."""
    top_result = results[0]
    print("\n" + "=" * 55)
    print(f"EuroSAT Satellite Classification Prediction")
    print("=" * 55)
    print(f"Image Source: {os.path.basename(image_path)}")
    print(f"Top Prediction: {top_result['class']} ({top_result['percentage']})")
    print("-" * 55)
    print("Top Candidates:")
    for res in results:
        bar = "#" * int(res['confidence'] * 25)
        print(f"  {res['rank']}. {res['class']:<22} | {res['percentage']:>7}  {bar}")
    print("=" * 55 + "\n")

    if save_plot:
        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))
        ax_img.imshow(pil_img)
        ax_img.set_title(f"Predicted: {top_result['class']}\n({top_result['percentage']})", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        classes = [r['class'] for r in reversed(results)]
        confs = [r['confidence'] * 100 for r in reversed(results)]
        bars = ax_bar.barh(classes, confs, color='#2563eb')
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel('Confidence (%)')
        ax_bar.set_title('Top Class Probabilities', fontsize=12, fontweight='bold')
        for bar in bars:
            width = bar.get_width()
            ax_bar.text(width + 2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                        ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_plot, dpi=200)
        plt.close()
        print(f"[INFO] Saved prediction visualization to: {save_plot}")


def get_random_sample(data_dir="dataset/EuroSAT_RGB"):
    """Pick a random image from the dataset for quick demonstration."""
    if not os.path.isdir(data_dir):
        return None
    classes = [c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))]
    if not classes:
        return None
    chosen_class = random.choice(classes)
    class_folder = os.path.join(data_dir, chosen_class)
    images = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return random.choice(images) if images else None


def main():
    parser = argparse.ArgumentParser(description="Predict Land-Use Class for a Satellite Image")
    parser.add_argument("--image", type=str, help="Path to input satellite image")
    parser.add_argument("--random_sample", action="store_true", help="Select a random image from the dataset")
    parser.add_argument("--model_path", type=str, default="outputs/cnn_best.keras",
                        help="Path to trained model (.keras or .h5)")
    parser.add_argument("--classes_path", type=str, default="outputs/classes.json",
                        help="Path to classes JSON mapping")
    parser.add_argument("--save_plot", type=str, default="outputs/sample_prediction.png",
                        help="Path to save output visualization")
    args = parser.parse_args()

    # Determine image path
    image_path = args.image
    if args.random_sample or not image_path:
        image_path = get_random_sample()
        if not image_path:
            print("[ERROR] Please provide --image <path> or ensure 'dataset/EuroSAT_RGB' exists.")
            sys.exit(1)
        print(f"[INFO] Using random dataset sample: {image_path}")

    # Fallback to .h5 if .keras does not exist yet
    model_path = args.model_path
    if not os.path.isfile(model_path):
        alt_h5 = model_path.replace(".keras", ".h5")
        if os.path.isfile(alt_h5):
            model_path = alt_h5

    results, pil_img, _ = predict(image_path, model_path, args.classes_path)
    os.makedirs(os.path.dirname(args.save_plot) or ".", exist_ok=True)
    display_prediction(image_path, results, pil_img, save_plot=args.save_plot)


if __name__ == "__main__":
    main()
