import cv2
import threading
from ultralytics import YOLO

# 1. Path to your downloaded trained weights
MODEL_PATH = "best.pt"

# 2. ESP32-CAM Stream URL (Update IP tomorrow)
ESP32_STREAM_URL = "http://192.168.1.50:81/stream" 

# Area < 5% = Small Fire, >= 5% = Large Fire
SMALL_FIRE_THRESHOLD = 0.05 

class StreamReceiver:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# Load trained YOLOv8 fire model
model = YOLO(MODEL_PATH)

# NOTE: Change ESP32_STREAM_URL to 0 if you want to test with your webcam today
stream = StreamReceiver(ESP32_STREAM_URL).start()

print("Pipeline Ready. Press 'q' to quit.")

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    frame_h, frame_w = frame.shape[:2]
    total_area = frame_w * frame_h

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area_ratio = ((x2 - x1) * (y2 - y1)) / total_area

        if area_ratio < SMALL_FIRE_THRESHOLD:
            label = f"Small {cls_name} ({area_ratio*100:.1f}%)"
            color = (0, 255, 255)  # Yellow for Small
        else:
            label = f"Large {cls_name} ({area_ratio*100:.1f}%)"
            color = (0, 0, 255)    # Red for Large

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("ESP32-CAM Fire Detection Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()