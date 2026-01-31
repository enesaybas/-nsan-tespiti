import cv2
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Model
model = YOLO("yolov8n.pt")

# Kamera
cap = cv2.VideoCapture(0)

# Zaman ayarları
total_duration = 30      # toplam 30 saniye
measure_interval = 5     # her 5 saniyede ölç
start_time = time.time()
last_measure_time = start_time

# Grafik verileri
time_labels = []
person_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed_time = int(time.time() - start_time)
    remaining_time = total_duration - elapsed_time

    if remaining_time <= 0:
        break

    # İnsan say
    results = model(frame, conf=0.4)
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "insan",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # 5 saniyede bir kayıt al
    if time.time() - last_measure_time >= measure_interval:
        time_labels.append(f"{elapsed_time}s")
        person_counts.append(person_count)
        last_measure_time = time.time()

    # Sayaç yazısı
    cv2.putText(
        frame,
        f"Kalan Sure: {remaining_time} sn",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Insan Takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Grafik
plt.figure()
plt.bar(time_labels, person_counts)
plt.xlabel("Zaman")
plt.ylabel("Insan Sayisi")
plt.title("30 Saniye Boyunca 5 Saniyede Bir Insan Sayisi")
plt.show()
