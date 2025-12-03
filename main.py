import cv2
import config
import time
import numpy as np
import threading

from detectors.detector_yolo import YOLODetector
from detectors.detector_identity_insight import InsightFaceRecognizer
from logger_debug import DebugLogger
from brain.simple_interpreter import describe_scene


# -----------------------------
#      ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# -----------------------------
llm_answer = "нет данных"
llm_busy = False    # ⬅️ защита от запуска нескольких воркеров одновременно


# -----------------------------
#         LLM worker
# -----------------------------
def llm_worker(objects, faces):
    global llm_answer, llm_busy

    try:
        print("\n[LLM] Запуск воркера...")
        print("[LLM] scene_text будет сгенерирован describe_scene()")

        answer = describe_scene(objects, faces)

        print("[LLM] Ответ LLM:", answer)
        llm_answer = answer

    except Exception as e:
        print("[LLM] Ошибка:", e)
        llm_answer = f"Ошибка LLM: {e}"

    llm_busy = False  # ⬅️ освобождаем воркер


# -----------------------------
#              MAIN
# -----------------------------
def main():
    global llm_answer, llm_busy

    logger = DebugLogger()
    yolo = YOLODetector()
    identity = InsightFaceRecognizer()

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

    frame_id = 0
    last_llm_time = time.time()

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Камера не отвечает")
            break

        debug_data = {"objects": [], "faces": []}

        # -------- YOLO --------
        t_yolo = time.time()
        result = yolo.detect(frame)
        yolo_time = time.time() - t_yolo

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            conf = float(box.conf)

            debug_data["objects"].append({
                "label": result.names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        # -------- FACE --------
        t_face = time.time()
        faces = identity.app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_time = time.time() - t_face

        for face in faces:
            emb = face.embedding
            bbox = face.bbox.astype(int).tolist()

            best_name = "Неизвестный"
            best_sim = -1

            for name, known_emb in identity.db.items():
                sim = float(np.dot(emb, np.array(known_emb)))
                if sim > identity.threshold and sim > best_sim:
                    best_sim = sim
                    best_name = name

            debug_data["faces"].append({
                "name": best_name,
                "similarity": best_sim,
                "bbox": bbox
            })

        # -------- LLM every 5 sec --------
        if not llm_busy and (time.time() - last_llm_time > 5):
            llm_busy = True
            last_llm_time = time.time()

            # Запускаем worker ПРАВИЛЬНО
            threading.Thread(
                target=llm_worker,
                args=(debug_data["objects"], debug_data["faces"]),
                daemon=True
            ).start()

        # -------- FPS --------
        dt = time.time() - loop_start
        fps = 1.0 / dt if dt > 0 else 0

        # каждые 10 кадров — лог в консоль
        if frame_id % 10 == 0:
            print(f"\nFPS: {fps:.1f} | YOLO: {yolo_time:.3f} | FACE: {face_time:.3f}")
            print(">>> AI:", llm_answer)

        frame_id += 1

        # -------- Draw --------
        frame = yolo.render(frame, result)
        cv2.imshow("Vision AI", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()