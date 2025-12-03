from flask import Flask, request, jsonify
import os, datetime

app = Flask(__name__)

VIDEO_DIR = "recordings"
CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

@app.post("/cut")
def cut_video():
    data = request.json
    start = data["start"]
    end = data["end"]

    target_name = datetime.datetime.now().strftime("clip_%Y-%m-%d_%H-%M-%S.mp4")
    target_path = os.path.join(CLIPS_DIR, target_name)

    # ищем актуальный записи файл
    files = sorted(os.listdir(VIDEO_DIR))
    if not files:
        return {"error":"no files"}, 404
    return {"saved": target_path}

app.run(port=5000)