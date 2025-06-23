import cv2
from ultralytics import YOLO

# ✅ Load your trained YOLOv8m model (update this path if needed)
model = YOLO(r'C:\Users\Anya gupta\PycharmProjects\pregraddata\pythonProject\yolomv8_trained\content\dataset\runs\Hand_Sign_Detection\Hand_Sign_Detection\weights\best.pt')

# ✅ Define class names (A-Z only)
class_names = [chr(i) for i in range(65, 91)] + ['asl_letters']  # ['A', 'B', ..., 'Z']

# ✅ Start webcam (0 = default cam)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # 🔍 Run prediction
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)

    # ✏️ Annotate frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id < len(class_names):
                label = f"{class_names[cls_id]}: {conf:.2f}"
            else:
                label = f"ID {cls_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 🖼️ Show frame
    cv2.imshow("YOLOv8 Hand Sign Detection", frame)

    # 🔴 Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
