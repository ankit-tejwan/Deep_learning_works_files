import os

# Set paths
label_dir = r"train/labels"  # Replace with the path to your label files

# Define the mapping
class_mapping = {0: 1, 1: 2}  # maps old indices to new indices

# Process each label file
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        # Read the file contents
        with open(os.path.join(label_dir, filename), 'r') as file:
            lines = file.readlines()

        # Update the class indices in each line
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # Original class ID
            # Update class ID if it exists in the mapping
            if class_id in class_mapping:
                parts[0] = str(class_mapping[class_id])  # Update class ID
            updated_lines.append(" ".join(parts))

        # Write the updated content back to the file
        with open(os.path.join(label_dir, filename), 'w') as file:
            file.write("\n".join(updated_lines) + "\n")

print("Label indices updated successfully.")
