from pathlib import Path
from PIL import Image
import argparse

def check_images(root_dir, delete=False):
    root = Path(root_dir)
    supported_exts = [".jpg", ".jpeg", ".png"]
    broken_files = []

    print(f"Checking images in: {root.resolve()}\n")

    for img_path in root.rglob("*"):
        if img_path.suffix.lower() in supported_exts:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check for corruption
            except Exception as e:
                print(f"[BROKEN] {img_path} ({e})")
                broken_files.append(img_path)

    print(f"\nTotal broken files: {len(broken_files)}")

    if delete and broken_files:
        for f in broken_files:
            try:
                f.unlink()
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and optionally remove broken image files.")
    parser.add_argument("folder", type=str, help="Root folder of the image dataset")
    parser.add_argument("--delete", action="store_true", help="Delete broken files")
    args = parser.parse_args()

    check_images(args.folder, delete=args.delete)
