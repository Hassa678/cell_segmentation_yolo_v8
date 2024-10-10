import shutil
import os

# Define source and destination paths using raw strings
source_path = r"C:\cell_segmentation_yolo_v8\runs\segment\train\weights\best.pt"
destination_dir = r"C:\cell_segmentation_yolo_v8\artifacts\model_trainer"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Construct the full destination path
destination_path = os.path.join(destination_dir, "best.pt")

# Print paths for debugging
print(f"Source path: {source_path}")
print(f"Destination path: {destination_path}")

# Check if source file exists
if not os.path.exists(source_path):
    print(f"Error: Source file not found: {source_path}")
else:
    try:
        # Copy the file
        shutil.copy(source_path, destination_path)
        print(f"File successfully copied to {destination_path}")
    except Exception as e:
        print(f"An error occurred while copying the file: {str(e)}")