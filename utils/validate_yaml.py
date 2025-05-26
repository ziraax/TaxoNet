from pathlib import Path

base = Path("/home/huwalter/WorkingFolderHugoWALTER/YOLOTaxa/DATA/yolo_dataset")
print("Train exists:", (base / "train").exists())
print("Val exists:", (base / "val").exists())
print("Classes:", [d.name for d in (base / "train").iterdir() if d.is_dir()])
