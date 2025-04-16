import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class YOLOv5GUI(QMainWindow):
    def __init__(self):
        super(YOLOv5GUI, self).__init__()
        self.setWindowTitle("YOLOv5 Real-Time Detection (PyQt5)")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Label to display video frames
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Start Button
        self.start_btn = QPushButton("Start", self)
        self.start_btn.clicked.connect(self.start_video)
        self.layout.addWidget(self.start_btn)

        # Stop Button
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop_video)
        self.layout.addWidget(self.stop_btn)

        # Load the YOLOv5 model using torch.hub
        try:
            print("Loading YOLOv5 model for GUI...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = 0.5  # Set confidence threshold
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading YOLOv5 model:\n{e}")
            sys.exit(1)
            
        # Initialize video capture object and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        print("YOLOv5 GUI initialized successfully.")

    def start_video(self):
        print("Attempting to start the webcam for GUI...")
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open webcam. Please ensure it is connected.")
                self.cap = None
                return
            print("Webcam started successfully for GUI.")
        self.timer.start(30)  # Update frame every 30 ms

    def stop_video(self):
        print("Stopping the webcam for GUI...")
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        print("Webcam stopped.")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame using YOLOv5 model
            results = self.model(frame_rgb)
            # Render predictions on the frame
            annotated_frame = results.render()[0]
            # Convert the annotated frame to QImage for display
            h, w, ch = annotated_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        else:
            print("Error: Failed to retrieve frame from webcam.")
            self.stop_video()

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

if __name__ == '__main__':
    print("Starting YOLOv5 PyQt5 GUI Application...")
    app = QApplication(sys.argv)
    window = YOLOv5GUI()
    window.show()
    sys.exit(app.exec_())
    print("YOLOv5 PyQt5 GUI Application exited.")
# This code provides a GUI application using PyQt5 to display real-time object detection using YOLOv5.