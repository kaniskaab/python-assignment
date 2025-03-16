import cv2
import numpy as np
import os
import random
from google.colab.patches import cv2_imshow  # Colab-specific image display

def load_data(image_path, label_path, class_names):
    """Loads image and YOLO annotation data."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}. Exiting.")
        exit()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_name = class_names[int(class_id)]
            annotations.append({
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
            })
    return image, annotations

def create_synthetic_data(image, annotations, image_size=(64, 64), num_augmentations=5):
    """Creates synthetic training data from YOLO annotations."""
    synthetic_images = []
    synthetic_labels = []
    img_height, img_width, _ = image.shape
    for _ in range(num_augmentations):
        for annotation in annotations:
            class_name = annotation['class_name']
            x_center, y_center, width, height = annotation['x_center'], annotation['y_center'], annotation['width'], annotation['height']

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            # Ensure crop boundaries are within image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            cropped_object = image[y1:y2, x1:x2]
            if cropped_object.size == 0:
                print(f"Warning: Cropped object has zero size. Skipping.")
                continue

            resized_object = cv2.resize(cropped_object, image_size)
            synthetic_images.append(resized_object)
            synthetic_labels.append(class_name)

            # Simple Augmentation - Flipping
            if random.random() < 0.5:
                augmented_object = cv2.flip(resized_object, 1)
                synthetic_images.append(augmented_object)
                synthetic_labels.append(class_name)

    synthetic_images = np.array(synthetic_images, dtype=np.float32) / 255.0
    return synthetic_images, synthetic_labels

def template_match(image, synthetic_images, synthetic_labels, threshold=0.7):
    """Performs template matching on single-channel images."""
    detections = []
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) # Convert the main image to grayscale and specify dtype
    #img_gray = img_gray.astype(np.float32) # Ensure correct type

    for i, template in enumerate(synthetic_images):
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY).astype(np.float32)  # Convert template to grayscale and specify dtype
        #template = template.astype(np.float32) # Ensure correct type

        # Ensure template is not larger than image
        if template.shape[0] > img_gray.shape[0] or template.shape[1] > img_gray.shape[1]:
            print(f"Skipping template {i} because it's larger than the input image")
            continue

        #Debugging: Print the datatypes at runtime
        #print(f"Image dtype: {img_gray.dtype}, Template dtype: {template.dtype}") #For debugging
        if template.dtype != img_gray.dtype:
            print(f"Warning: Template {i} has a different data type than the image. Skipping.")
            continue

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            detections.append({
                'class_name': synthetic_labels[i],
                'x': pt[0],
                'y': pt[1],
                'confidence': res[pt[1], pt[0]]
            })
    return detections


def visualize_detections(image, detections):
    """Visualizes the detections on the image (Colab-compatible)."""
    image_copy = image.copy()
    for detection in detections:
        x = detection['x']
        y = detection['y']
        class_name = detection['class_name']
        confidence = detection['confidence']

        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(image_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image_copy, (x, y), (x + 20, y + 20), (0, 255, 0), 2)

    cv2_imshow(image_copy)  # Use Colab's cv2_imshow

def load_class_names(yaml_file):
    """Loads class names from data.yaml file."""
    with open(yaml_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('names:'):
                class_names_str = line.split('[')[1].split(']')[0].replace("'", "").strip()
                return [name.strip() for name in class_names_str.split(',')]
    return None  # Or raise an exception if 'names' is not found

if __name__ == "__main__":
    # 1. Load Class Names
    yaml_path = 'data.yaml'
    class_names = load_class_names(yaml_path)
    class_names = [name.replace('"','').strip() for name in class_names] # Sanitize names

    if not class_names:
        print("Error: Could not load class names from data.yaml. Exiting.")
        exit()

    # 2. Load Data
    image_path = 'train/images/demo.jpg'
    label_path = 'train/labels/demo.txt'
    image, annotations = load_data(image_path, label_path, class_names)
    print(f"Successfully loaded {len(annotations)} annotations")

    # 3. Create Synthetic Data
    synthetic_images, synthetic_labels = create_synthetic_data(image, annotations, num_augmentations=10)
    print(f"Created {len(synthetic_images)} synthetic images for training")

    # 4. Template Matching
    detections = template_match(image, synthetic_images, synthetic_labels, threshold=0.4)  # Increased sensitivity

    # 5. Visualize Detections
    visualize_detections(image, detections)