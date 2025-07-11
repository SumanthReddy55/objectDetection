#!/usr/bin/env python3
import cv2
#import numpy as np

# Load the model
model_path = 'frozen_inference_graph.pb'
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Load the COCO labels
with open('labels.txt', 'r') as f:
    class_names = f.read().strip().split('\n')

# Load the neural network
net = cv2.dnn_DetectionModel(model_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Dummy image directory
import os
image_folder = 'data'  # Create a folder named 'data' and place your images there
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Process each image
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        continue

    # Object detection
    class_ids, confidences, boxes = net.detect(img, confThreshold=0.5)

    # Draw detections
    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            label = f"{class_names[class_id - 1].upper()} {round(confidence * 100, 2)}%"
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection Output", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
