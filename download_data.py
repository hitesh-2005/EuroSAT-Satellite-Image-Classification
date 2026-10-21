"""EuroSAT RGB Dataset Downloader and Validator.

Downloads and extracts the official EuroSAT RGB dataset (Sentinel-2, 64x64, 10 classes).
Dataset source: Open Access EuroSAT (CC BY-NC 4.0).
Mirror: HuggingFace torchgeo/eurosat (verified direct link).
"""

import os
import sys
import zipfile
import urllib.request

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TARGET_DIR = os.path.join(DATASET_DIR, "EuroSAT_RGB")
ZIP_PATH = os.path.join(DATASET_DIR, "EuroSAT.zip")
DOWNLOAD_URL = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip"

EXPECTED_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def check_dataset_exists(target_dir):
    """Check if the dataset directory is already populated with expected classes."""
    if not os.path.isdir(target_dir):
        return False
    present = [c for c in EXPECTED_CLASSES if os.path.isdir(os.path.join(target_dir, c))]
    if len(present) == len(EXPECTED_CLASSES):
        counts = {c: len(os.listdir(os.path.join(target_dir, c))) for c in EXPECTED_CLASSES}
        total = sum(counts.values())
        if total >= 25000:
            print(f"[INFO] EuroSAT dataset already verified at: {target_dir}")
            print(f"[INFO] Total verified images: {total} across {len(EXPECTED_CLASSES)} classes.")
            return True
    return False


def download_with_progress(url, dest_path):
    """Download a file with user-friendly console progress updates."""
    print(f"[INFO] Downloading EuroSAT RGB archive (~90 MB) from:\n       {url}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            msg = f"\r[INFO] Progress: {percent:5.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)"
        else:
            msg = f"\r[INFO] Downloaded: {downloaded / (1024 * 1024):.1f} MB"
        sys.stdout.write(msg)
        sys.stdout.flush()

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 1024 * 64
        downloaded = 0
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            reporthook(downloaded // block_size, block_size, total_size)
    print("\n[INFO] Download completed successfully.")


def extract_and_organize(zip_path, target_dir):
    """Extract zip archive and ensure uniform class folder hierarchy."""
    print(f"[INFO] Extracting archive to {DATASET_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATASET_DIR)

    # Some EuroSAT zip files unpack directly to 2750/ or EuroSAT/ or dataset/
    extracted_candidates = [
        os.path.join(DATASET_DIR, "2750"),
        os.path.join(DATASET_DIR, "EuroSAT"),
        target_dir
    ]

    source_dir = None
    for cand in extracted_candidates:
        if os.path.isdir(cand) and any(os.path.isdir(os.path.join(cand, c)) for c in EXPECTED_CLASSES):
            source_dir = cand
            break

    if source_dir and source_dir != target_dir:
        if os.path.exists(target_dir):
            import shutil
            shutil.rmtree(target_dir)
        os.rename(source_dir, target_dir)
        print(f"[INFO] Dataset organized at: {target_dir}")

    # Remove temporary zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("[INFO] Temporary zip archive removed to save disk space.")


def verify_classes(target_dir):
    """Verify class structure and counts."""
    print("\n" + "=" * 50)
    print(f"{'Class Name':<25} | {'Image Count':<10}")
    print("-" * 50)
    total = 0
    missing = []
    for c in EXPECTED_CLASSES:
        c_path = os.path.join(target_dir, c)
        if os.path.isdir(c_path):
            count = len([f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{c:<25} | {count:<10}")
            total += count
        else:
            print(f"{c:<25} | MISSING")
            missing.append(c)
    print("=" * 50)
    print(f"Total Dataset Images: {total}")
    if missing:
        raise RuntimeError(f"Dataset verification failed. Missing classes: {missing}")
    print("[SUCCESS] All 10 EuroSAT classes successfully verified.\n")


def setup_dataset():
    """Main workflow entrypoint."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    if check_dataset_exists(TARGET_DIR):
        verify_classes(TARGET_DIR)
        return TARGET_DIR

    download_with_progress(DOWNLOAD_URL, ZIP_PATH)
    extract_and_organize(ZIP_PATH, TARGET_DIR)
    verify_classes(TARGET_DIR)
    return TARGET_DIR


if __name__ == "__main__":
    setup_dataset()
