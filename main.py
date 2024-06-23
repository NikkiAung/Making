import cv2
import numpy as np
import csv
import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPEN_AI_KEY")

# Function to generate recipe using OpenAI
def generate_recipe(ingredients):
    prompt = f"Create a recipe using the following ingredients: {', '.join(ingredients)}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    recipe = response.choices[0].message["content"].strip()
    return recipe

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Debugging print statement to check the output of getUnconnectedOutLayers
print("Unconnected out layers: ", net.getUnconnectedOutLayers())

# Corrected way to get the output layers
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except Exception as e:
    print("Error while getting output layers: ", e)
    exit(1)

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to initialize the video capture
def initialize_video_capture():
    for i in range(5):  # Try the first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera initialized with index {i}")
            return cap
        cap.release()
    return None

# Initialize video capture
cap = initialize_video_capture()
if not cap:
    print("No camera found")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Clear the detection data for the current frame
    detection_data = []

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))  # Convert to float
                class_ids.append(class_id)

                # Save detection data to the list
                detected_class = classes[class_id]
                if detected_class != "person":  # Ignore "person"
                    detection_data.append(detected_class)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                continue  # Skip drawing bounding box for "person"
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    cv2.imshow("Image", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

    if detection_data:
        # Save the detection data to a CSV file
        with open("detection_results.csv", "w", newline='') as csv_file:
            fieldnames = ["class"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()
            for detected_class in detection_data:
                writer.writerow({"class": detected_class})

        print("Detection data saved to detection_results.csv")

        # Generate a recipe based on the detected ingredients
        ingredients = set(detection_data)
        recipe = generate_recipe(ingredients)
        print("Generated Recipe:")
        print(recipe)

# Release the video capture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
