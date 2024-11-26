import cv2
import torch
from torchvision import models, transforms
from playsound import playsound
import threading


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO class names 
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform = transforms.Compose([
    transforms.ToTensor()
])

# Function to play a warning sound
def play_warning_sound():
    def sound_thread():
        playsound("D:\inlustro_object_detection\\alarm\\burglar_alarm.mp3")  # Replace with the path to your alarm sound file
    threading.Thread(target=sound_thread, daemon=True).start()

# Function to draw bounding boxes and trigger alarm
def draw_boxes_and_detect(frame, outputs, roi, confidence_threshold=0.5):
    """
    Draw bounding boxes and trigger alarm if a person is detected in the ROI.
    """
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score >= confidence_threshold and COCO_INSTANCE_CATEGORY_NAMES[label.item()] == "person":
            x1, y1, x2, y2 = map(int, box)
            # Check if the bounding box intersects with the ROI
            if roi is not None:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi
                if (x1 < roi_x2 and x2 > roi_x1 and y1 < roi_y2 and y2 > roi_y1):  # Check overlap
                    play_warning_sound()
                    cv2.putText(frame, "WARNING: Person Detected!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Real-time object detection
def main():
    cap = cv2.VideoCapture(0)  # Use webcam (or change to a video file path)

    # Define the Region of Interest (ROI)
    roi_x1, roi_y1, roi_x2, roi_y2 = 200, 100, 400, 300  # Example ROI coordinates
    roi = (roi_x1, roi_y1, roi_x2, roi_y2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Draw ROI on the frame
        cv2.rectangle(frame_resized, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(frame_resized, "Restricted Area", (roi_x1, roi_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert frame to tensor and add batch dimension
        input_tensor = transform(frame_resized).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = model(input_tensor)[0]

        # Draw bounding boxes and check for person in ROI
        frame_with_boxes = draw_boxes_and_detect(frame_resized, predictions, roi, confidence_threshold=0.5)

        # Display the frame
        cv2.imshow("Surveillance System", frame_with_boxes)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
