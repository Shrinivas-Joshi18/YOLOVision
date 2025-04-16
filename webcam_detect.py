import cv2
import torch

print("Loading YOLOv5 model...")

# Load the YOLOv5 model from torch.hub
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

model.eval()  # Set model to evaluation mode

# Open the default webcam (0 = default device)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam. Please check your webcam connection and ensure no other app is using it.")
    exit()
else:
    print("✅ Webcam opened successfully.")

print("✅ Starting webcam feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to grab frame from webcam.")
        break

    # Print frame info for debugging (optional)
    print("Frame captured: shape =", frame.shape)

    # Convert captured frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection using YOLOv5 model
    results = model(frame_rgb)

    # Render detection results on the frame (bounding boxes, labels, etc.)
    annotated_frame = results.render()[0]

    # Convert annotated frame back to BGR (for OpenCV display)
    frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the processed frame in a window
    cv2.imshow('YOLOv5 Webcam Detection', frame_bgr)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit signal received. Exiting...")
        break

# Release webcam and close display window(s)
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam feed stopped and resources have been released.")
