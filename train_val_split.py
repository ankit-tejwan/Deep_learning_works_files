import os
import random
import shutil

# Set the source and destination folder paths
source_folder = r'output'
train_folder = r'train'
val_folder = r'valid'

# Create the destination folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get a list of all image files in the source folder
image_files = [
    file for file in os.listdir(source_folder)
    if file.endswith(('.png', '.jpg', '.bmp', '.jpeg'))
]

# Shuffle the image files to randomize selection
random.shuffle(image_files)

# Calculate the split for training and validation
num_files = len(image_files)
num_train_files = int(0.7 * num_files)  # 70% for training
num_val_files = num_files - num_train_files  # Remaining for validation

# Split the files into training and validation sets
train_files = image_files[:num_train_files]
val_files = image_files[num_train_files:]

# Move the selected image files and their corresponding text files for training
for image_file in train_files:
    image_path = os.path.join(source_folder, image_file)
    # Find the corresponding text file
    txt_file = os.path.splitext(image_file)[0] + '.txt'
    txt_path = os.path.join(source_folder, txt_file)

    if os.path.exists(image_path) and os.path.exists(txt_path):
        # Move the image file
        shutil.move(image_path, os.path.join(train_folder, image_file))
        # Move the text file
        shutil.move(txt_path, os.path.join(train_folder, txt_file))
        
        # Print which files are being moved
        print(f"Moved to train: {image_file} and {txt_file}")

# Move the selected image files and their corresponding text files for validation
for image_file in val_files:
    image_path = os.path.join(source_folder, image_file)
    # Find the corresponding text file
    txt_file = os.path.splitext(image_file)[0] + '.txt'
    txt_path = os.path.join(source_folder, txt_file)

    if os.path.exists(image_path) and os.path.exists(txt_path):
        # Move the image file
        shutil.move(image_path, os.path.join(val_folder, image_file))
        # Move the text file
        shutil.move(txt_path, os.path.join(val_folder, txt_file))
        
        # Print which files are being moved
        print(f"Moved to valid: {image_file} and {txt_file}")

print(f"Moved {num_train_files} files to the training folder and {num_val_files} files to the validation folder.")
