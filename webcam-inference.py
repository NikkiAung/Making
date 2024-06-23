import cv2
import json
import supervision as sv
from ultralytics import YOLOv10

# Load the model
model = YOLOv10('/Users/aungnandaoo/Desktop/making/best.pt')

# Initialize annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")

# Counter for image frames
img_counter = 0

# List to hold detection results
detection_results = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate the frame
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # Display the annotated frame
    cv2.imshow('Webcam', annotated_image)

    # Save detections to JSON without duplicates
    detected_labels = set()
    for detection in detections:
        print("Detection:", detection)
        label = tuple(detection[5])  # Convert label dict to a tuple

        if label not in detected_labels:
            detection_data = {
                'label': detection[5]  # Use the original dict for saving
            }
            detection_results.append(detection_data)
            detected_labels.add(label)

    # Handle key events
    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("Escape hit, closing...")
        break

    img_counter += 1

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Save the detection results to a JSON file
with open('detection_results.json', 'w') as json_file:
    json.dump(detection_results, json_file, indent=4)

print("Detections saved to detection_results.json")
