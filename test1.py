!pip install easyocr opencv-python pyyaml

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
            class_name = class_names[int(float(class_id))] # Fixed conversion and indexing
            annotations.append({
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
            })
    return image, annotations

def extract_text_from_image(image, reader):
    """Extracts text and bounding box information from the image using EasyOCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    text_data = []
    for (bbox, text, confidence) in results:
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
            'height': height,
            'confidence': confidence,  #Add confidence from OCR
        })
    return text_data

def associate_tags(detections, text_data, image_width, image_height, max_distance=100):
    """Associates detections with nearby text tags."""
    tagged_detections = []
    for detection in detections:
        best_match = None
        min_distance = float('inf')

        # Convert YOLO center coordinates to pixel coordinates
        detection_x = int(detection['x_center'] * image_width)  #Direct Value
        detection_y = int(detection['y_center'] * image_height)  #Direct Value

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

def visualize_detections(image, tagged_detections, class_names):
    """Visualizes the detections on the image with tag associations."""
    image_copy = image.copy()
    counts = {}

    # Initialize counts dictionary
    for class_name in class_names:
        counts[class_name] = {}

    for detection in tagged_detections:
        x_center = detection['x_center']
        y_center = detection['y_center']
        width = detection['width']
        height = detection['height']
        class_name = detection['class_name']
        confidence = detection['confidence']
        tag = detection.get('tag', None)  #Safe Get

        img_height, img_width, _ = image.shape
        x1 = int((x_center - width / 2) * img_width) #Int, to ensure
        y1 = int((y_center - height / 2) * img_height) #Int, to ensure
        x2 = int((x_center + width / 2) * img_width) #Int, to ensure
        y2 = int((y_center + height / 2) * img_height) #Int, to ensure

        label = f"{class_name} ({confidence:.2f})"

        if tag:
            label += f" - Tag: {tag}"
        else:
            label += " - No nearby tag"

        cv2.putText(image_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #Count and use a key method
        if class_name not in counts: #Make checks to generate if they are all fine or not
                counts[class_name] = {}
        if tag not in counts[class_name]: #This ensures that nothing goes array.
               counts[class_name][tag] = 0 #Generate a new base here for the check

        counts[class_name][tag] += 1  #Increment Counts to ensure things go brr

    cv2_imshow(image_copy)
    return counts

def main(image_path, label_path, yaml_path):
    """Main function to perform object detection and tag association."""
    # 1. Load Class Names
    class_names = load_class_names(yaml_path)

    # 2. Load Data
    image, annotations = load_data(image_path, label_path, class_names)

    # 3. Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])

    # 4. Extract Text with EasyOCR
    text_data = extract_text_from_image(image, reader)

    # 5. Associate Tags with Detections
    for anno in annotations:  #Ensure for 
        if 'confidence' not in anno:
            anno['confidence'] = 0.99; #Use 1.0 will show for a small time but 0.99 is good.
    tagged_detections = associate_tags(annotations, text_data, image.shape[1], image.shape[0])

    # 6. Visualize Detections
    counts = visualize_detections(image, tagged_detections, class_names) # Fixed: Pass class_names

    # 7. Print Summary Statistics in Desired Format:
    for class_name in class_names:
         print(f"{class_name.capitalize()}:")
         if class_name in counts:
                for tag, count in counts[class_name].items():
                    print(f"   {tag}: {count}")
         else:
               print(" No objects are detected to load with OCR or Annotate") #If this pops up, then ocr may have failed.

test_image_path = 'test/test.png'
test_label_path = 'train/labels/demo.txt'  #Reusing data.txt to show code can parse and work
yaml_path = 'data.yaml'

if os.path.exists(test_image_path) and os.path.exists(test_label_path):
    main(test_image_path, test_label_path, yaml_path)
else:
    print("Error: Ensure 'test.png' exists in the 'test/' directory and 'demo.txt' exists in the 'train/labels/' directory.")
