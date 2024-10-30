import os

# Set the path to the labels directory
label_dir = r"yolo_data_labels"  # Replace with the path to your labels folder

# Iterate through the files in the directory
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        # Check if the file is empty or contains only whitespace
        with open(file_path, 'r') as file:
            content = file.read().strip()  # Read and strip whitespace
        
        if not content:  # If content is empty, delete the file
            os.remove(file_path)  # Delete the empty file
            print(f"Deleted empty file: {filename}")

print("Empty text files deleted successfully.")
