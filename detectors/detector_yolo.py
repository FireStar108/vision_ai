import torch
from ultralytics import YOLOWorld
import cv2
import config

class YOLODetector:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"

        print(f"[YOLO] Инициализация, CUDA: {torch.cuda.is_available()}")

        # Загружаем модель
        self.model = YOLOWorld(getattr(config, "YOLO_MODEL_PATH", "yolo/yolov8x-world.pt"))
        self.model.set_classes(getattr(config, "YOLO_CLASSES", ["person"]))

        # Универсальный способ указать устройство
        self.model.overrides["device"] = self.device

        print(f"[YOLO] Модель настроена на устройство: {self.model.overrides['device']}")

    def detect(self, frame):
        result = self.model.predict(
            frame,
            device=self.device,
            conf=getattr(config, "YOLO_CONFIDENCE", 0.4),
            verbose=False
        )
        return result[0]

    def render(self, frame, result):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = f"{result.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame