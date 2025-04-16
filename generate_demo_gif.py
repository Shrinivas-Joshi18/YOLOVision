import cv2
import imageio
from ultralytics import YOLO

# Load the model
model = YOLO("yolov5s.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
frames = []
frame_count = 0
max_frames = 60  # ~3 seconds if 20 fps

print("🎥 Recording demo... Press 'q' to stop early.")

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 detection
    results = model.predict(source=frame, conf=0.4, save=False)
    annotated_frame = results[0].plot()

    # Convert BGR to RGB for GIF output
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)

    cv2.imshow("Recording Demo", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Save to GIF
imageio.mimsave("demo.gif", frames, fps=10) 
print("✅ Demo GIF saved as 'demo.gif'")
print("🛑 Demo recording stopped.")
