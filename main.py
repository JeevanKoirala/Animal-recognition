import cv2
import time
from ultralytics import YOLO

INPUT_VIDEO = "animals.mp4"
OUTPUT_VIDEO = "output/annotated_output.mp4"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, frame_fps, (frame_width, frame_height))

frame_count = 0
start_time = time.time()

resize_width = 640
resize_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            if confidences[i] > 0.5:
                x1, y1, x2, y2 = boxes[i].astype(int)
                label = f"{model.names[int(class_ids[i])]} ({confidences[i]:.2f})"

                font_scale = 1.5
                thickness = 4

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    resized_frame = cv2.resize(frame, (resize_width, resize_height))

    out_video.write(frame)

    elapsed_time = time.time() - start_time
    cv2.putText(resized_frame, f"Elapsed Time: {int(elapsed_time)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Animal Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

print(f"Detection completed. Annotated video saved as '{OUTPUT_VIDEO}'.")
