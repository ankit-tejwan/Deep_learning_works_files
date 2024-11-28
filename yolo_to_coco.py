import os
import json
from PIL import Image

def convert_yolo_to_coco(yolo_images_dir, output_json, categories):
    """
    Convert YOLO annotations to COCO format.

    :param yolo_images_dir: Path to the directory with images and YOLO annotations (.txt files).
    :param output_json: Path to the output COCO JSON file.
    :param categories: A list of dictionaries for the COCO categories. 
                       Example: [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, ...]
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    image_id = 1

    # Supported image file formats
    supported_formats = ['.jpg', '.jpeg', '.png']

    # Loop through all files in the directory
    for filename in os.listdir(yolo_images_dir):
        if not filename.endswith('.txt'):
            continue
        
        # Extract corresponding image file name (handle different extensions)
        base_filename = filename[:-4]  # Remove .txt
        img_path = None

        # Find the matching image file
        for ext in supported_formats:
            temp_img_path = os.path.join(yolo_images_dir, base_filename + ext)
            if os.path.exists(temp_img_path):
                img_path = temp_img_path
                break

        if not img_path:
            print(f"No image found for annotation file: {filename}, skipping.")
            continue

        # Load the image to get its dimensions
        try:
            img = Image.open(img_path)
            width, height = img.size
        except FileNotFoundError:
            print(f"Image not found: {img_path}, skipping.")
            continue

        # Add image details to COCO data
        coco_data['images'].append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": width,
            "height": height
        })

        # Read YOLO annotation
        txt_path = os.path.join(yolo_images_dir, filename)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        # Parse each line of the YOLO annotation
        if not lines:
            print(f"Empty annotation file: {txt_path}, skipping.")
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid annotation in file {txt_path}, skipping line: {line}")
                continue

            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

            # Convert normalized YOLO coordinates to absolute COCO coordinates
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            abs_bbox_width = bbox_width * width
            abs_bbox_height = bbox_height * height

            # Add annotation to COCO format
            coco_data['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,  # COCO categories start from 1, YOLO starts from 0
                "bbox": [x_min, y_min, abs_bbox_width, abs_bbox_height],
                "area": abs_bbox_width * abs_bbox_height,
                "iscrowd": 0  # YOLO doesn't support crowd annotations
            })
            annotation_id += 1

        image_id += 1

    # Save COCO format data to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"Conversion complete! COCO data saved to {output_json}")

# Usage
yolo_images_dir = r'train'  # Directory containing images and YOLO .txt files
output_coco_json = r'train.json'  # Specify the output JSON file with .json extension
categories = [
    {"id": 1, "name": "Rust"},
    {"id": 2, "name": "dent"}
]

# Convert YOLO to COCO
convert_yolo_to_coco(yolo_images_dir, output_coco_json, categories)

