import json
import os
import time
from datetime import datetime
import requests

WEBHOOK_URL = "https://n8n.ubric.ru/webhook/1cd7fa02-1b74-4210-b685-c08e4e271058"

def send_to_n8n(event):
    try:
        requests.post(WEBHOOK_URL, json=event, timeout=1)
    except Exception:
        pass  # не мешаем основной работе камеры
DEBUG_LOG_PATH = "debug_logs"

class DebugLogger:
    def __init__(self):
        os.makedirs(DEBUG_LOG_PATH, exist_ok=True)

    def log(self, data, save_image=False, frame=None):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename_json = os.path.join(DEBUG_LOG_PATH, f"{ts}.json")

        # --- сохраняем кадр, если нужно ---
        if save_image and frame is not None:
            img_path = os.path.join(DEBUG_LOG_PATH, f"{ts}.jpg")
            from cv2 import imwrite
            imwrite(img_path, frame)
            data["image_path"] = img_path

        # --- сохраняем JSON ---
        with open(filename_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        send_to_n8n(data)