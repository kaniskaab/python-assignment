
import cv2
import numpy as np
import os
import yaml
import easyocr
from google.colab.patches import cv2_imshow

def load_class_names(yaml_file):
    """Loads class names from data.yaml file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
        return data['names']

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
            class_name = class_names[int(float(class_id))]  # Class ID to index
            annotations.append({
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
            })
    return image, annotations

def create_synthetic_data(image, annotations, image_size=(64, 64), num_augmentations=5):
    """Creates synthetic data by cropping annotated objects and augmenting."""
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
                continue
            resized_object = cv2.resize(cropped_object, image_size)
            synthetic_images.append(resized_object)
            synthetic_labels.append(class_name)

    synthetic_images = np.array(synthetic_images, dtype=np.float32) / 255.0
    return synthetic_images, synthetic_labels

def template_match(image, synthetic_images, synthetic_labels, threshold=0.6):
    """Performs template matching to detect objects in the image."""
    detections = []
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    for i, template in enumerate(synthetic_images):
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if template.shape[0] > img_gray.shape[0] or template.shape[1] > img_gray.shape[1]:
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

def extract_text_from_image(image, reader):
    """Extracts text and bounding box information from the image using EasyOCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)  # Pass grayscale image to EasyOCR

    text_data = []
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = tuple([int(x) for x in tl])
        tr = tuple([int(x) for x in tr])
        br = tuple([int(x) for x in br])
        bl = tuple([int(x) for x in bl])
        x_center = int((tl[0] + br[0]) / 2)
        y_center = int((tl[1] + br[1]) / 2)
        width = int(br[0] - tl[0])
        height = int(br[1] - tl[1])
        text_data.append({
            'text': text,
            'x': x_center,
            'y': y_center,
            'width': width,
            'height': height
        })
    return text_data

def associate_tags(detections, text_data, max_distance=100):
    """Associates detections with nearby text tags."""
    tagged_detections = []
    for detection in detections:
        best_match = None
        min_distance = float('inf')
        detection_x = detection['x']
        detection_y = detection['y']

        for text_item in text_data:
            text_x = text_item['x']
            text_y = text_item['y']
            distance = np.sqrt((detection_x - text_x)**2 + (detection_y - text_y)**2)
            if distance < min_distance:
                min_distance = distance
                best_match = text_item['text']

        if min_distance <= max_distance:
            detection['tag'] = best_match
        else:
            detection['tag'] = None
        tagged_detections.append(detection)
    return tagged_detections

def visualize_and_count(image, tagged_detections, class_names):
    """Visualizes detections and returns the counts in the specified format."""
    image_copy = image.copy()
    counts = {}

    # Initialize counts dictionary with all class names from `data.yaml`.
    # This will ensure we output 0 for classes that are not detected.
    for class_name in class_names:
        counts[class_name] = {} # Create empty dict for each class.

    for detection in tagged_detections:
        x = detection['x']
        y = detection['y']
        class_name = detection['class_name']
        confidence = detection['confidence']
        tag = detection.get('tag', None)

        label = f"{class_name} ({confidence:.2f})"

        if tag:
            label += f" - Tag: {tag}"
            if tag not in counts[class_name]:
                counts[class_name][tag] = 0 #Init for each Tag for each Class
            counts[class_name][tag] += 1  # Class with tag
        else:
            label += " - No nearby tag"
            if "No Tag" not in counts[class_name]:
                counts[class_name]["No Tag"] = 0
            counts[class_name]["No Tag"] += 1 #Class without tag

        cv2.putText(image_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image_copy, (x, y), (x + 20, y + 20), (0, 255, 0), 2)

    cv2_imshow(image_copy)
    return counts

def main():
    # 1. Load Class Names
    yaml_path = 'data.yaml'
    class_names = load_class_names(yaml_path)

    # 2. Load Data
    image_path = 'train/images/demo.jpg'
    label_path = 'train/labels/demo.txt'
    image, annotations = load_data(image_path, label_path, class_names)

    # 3. Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])  # Specify the language(s)

    # 4. Extract Text with EasyOCR
    text_data = extract_text_from_image(image, reader)

    # 5. Create Synthetic Data (if annotations are not available)
    synthetic_images, synthetic_labels = create_synthetic_data(image, annotations, class_names, num_augmentations=10)

    # 6. Template Matching
    detections = template_match(image, synthetic_images, synthetic_labels, threshold=0.3)

    # 7. Associate Tags with Detections
    tagged_detections = associate_tags(detections, text_data)

    # 8. Visualize Detections and Count Tagged Items
    counts = visualize_and_count(image, tagged_detections, class_names)

    # 9. Print Summary Statistics in Desired Format:
    total_count = 0
    for class_name in class_names:
      print(f"{class_name.capitalize()}:")
      class_count = 0
      for tag, count in counts[class_name].items():
          print(f"    {tag}: {count}") #Proper indents
          class_count +=count
      total_count += class_count #Sum all classes

      print("\n")

    print(f"Total Counts:{total_count} \n")

if __name__ == "__main__":
    main()


