import os
import shutil

# Set paths
label_dir = r"train/labels"  # Replace with the path to your label files
image_dir = r"train/images"   # Replace with the path to your image files
output_dir = "output1"   # Replace with the path to your output folder

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize total count
total_count = 0

# Process each label file
for label_filename in os.listdir(label_dir):
    if label_filename.endswith(".txt"):
        label_path = os.path.join(label_dir, label_filename)
        image_base_name = os.path.splitext(label_filename)[0]  # Get the base name without extension
        image_path = os.path.join(image_dir, f"{image_base_name}.jpg")  # Adjust if your images have a different extension

        # Count the number of labels
        with open(label_path, 'r') as file:
            lines = file.readlines()
            label_count = len(lines)

        # Add to total count
        total_count += label_count

        # Create the output line with image base name and label count
        output_line = f"{image_base_name}: {label_count}\n"
        # get  current working directory
        cwd = os.getcwd()
        # Change to the output directory


        # Write to the output file
        output_file_path = os.path.join(cwd, 'label_counts.txt')
        with open(output_file_path, 'a') as output_file:
            output_file.write(output_line)

        # Move the label file to the output folder
        shutil.move(label_path, os.path.join(output_dir, label_filename))
        
        # Move the corresponding image file to the output folder
        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(output_dir, os.path.basename(image_path)))

# Print total count of labels processed
print(f"Total labels counted: {total_count}")
print("Label counts and files processed successfully.")
