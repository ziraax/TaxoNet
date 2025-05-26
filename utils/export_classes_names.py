import os

def export_class_names(test_folder, output_file="class_names.txt"):
    # List all directories (classes) inside the test folder
    classes = [d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))]
    classes.sort()  # Optional: sort alphabetically

    # Write class names to the output file
    with open(output_file, "w") as f:
        for cls in classes:
            f.write(cls + "\n")

    print(f"Exported {len(classes)} classes to {output_file}")

if __name__ == "__main__":
    test_folder_path = "DATA/yolo_dataset/test"  # change this to your actual test folder path
    export_class_names(test_folder_path)
