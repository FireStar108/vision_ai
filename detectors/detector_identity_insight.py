from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
import json
import onnxruntime as ort
import config

class InsightFaceRecognizer:
    def __init__(self, faces_dir="faces", db_path="faces_db.json", threshold=0.35):
        self.faces_dir = faces_dir
        self.db_path = db_path
        self.threshold = threshold

        # --- выбираем провайдера ---
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() \
            else ["CPUExecutionProvider"]

        # --- загружаем модель ---
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)

        # --- база лиц ---
        self.db = self.load_db() if os.path.exists(self.db_path) else {}
        if not self.db:
            self.build_db()

    # ---------- работа с JSON базой ----------
    def load_db(self):
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.db, f, ensure_ascii=False, indent=2)
        except:
            print("[⚠️] Ошибка сохранения базы лиц")

    # ---------- построение базы ----------
    def build_db(self):
        print("[⚙️] Поиск лиц и создание базы...")
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            return

        temp = {}  # {name: [emb1, emb2, ...]}

        for img_name in os.listdir(self.faces_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(self.faces_dir, img_name)
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img)
            if not faces:
                continue

            emb = faces[0].embedding
            name = os.path.splitext(img_name)[0].split("_")[0]
            temp.setdefault(name, []).append(emb)

        # усредняем эмбеддинги
        self.db = {
            name: (np.mean(np.stack(embs), axis=0) /
                   np.linalg.norm(np.mean(np.stack(embs), axis=0))).tolist()
            for name, embs in temp.items()
        }

        print(f"[📦] База лиц построена: {len(self.db)} человек")
        self.save_db()

    # ---------- распознавание ----------
    def recognize(self, frame):
        faces = self.app.get(frame)
        if not faces:
            return frame

        for face in faces:
            emb = face.embedding
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # сравнение
            best_match = "Неизвестный"
            best_sim = -1.0
            for name, known_emb in self.db.items():
                sim = float(np.dot(emb, np.array(known_emb)))
                if sim > self.threshold and sim > best_sim:
                    best_sim = sim
                    best_match = name

            color = (0, 255, 0) if best_match != "Неизвестный" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, best_match, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame