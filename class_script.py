import os
import matplotlib.pyplot as plt
from collections import Counter

# Set paths
# label_dir =  r"train/labels"  # Replace with the path to your label files
label_dir =  r"output1"

# Class names
names = ['class 0', 'class 1', 'class 2']
num_classes = len(names)

# Counter for each class
class_counts = Counter()

# Read each label file and count class occurrences
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(label_dir, filename), 'r') as file:
            for line in file:
                class_id = int(line.split()[0])  # Assuming class ID is the first element
                if 0 <= class_id < num_classes:
                    class_counts[class_id] += 1

# Prepare data for plotting
class_labels = [names[i] for i in range(num_classes)]
counts = [class_counts[i] for i in range(num_classes)]

# Plotting the bar graph
plt.figure(figsize=(8, 6))
plt.bar(class_labels, counts, color=['blue', 'orange', 'green'])
plt.xlabel('Class Names')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()
