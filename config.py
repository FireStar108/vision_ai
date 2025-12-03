# ===============================
# ⚙️ CONFIGURATION FILE
# ===============================

# --- Система ---
DEBUG_MODE = False
CAMERA_INDEX = 0            # 0 - встроенная камера, 1 - внешняя

# --- Параметры камеры ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_LIMIT = 30              # Ограничение FPS (0 = без лимита)

# --- Режимы логирования ---
# --- Настройки YOLO ---
YOLO_TYPE = "world"         # варианты: v8, v9, world

YOLO_CLASSES = [
    "person",        
    "car",          
    "punching bag", 
    "door",           
    "bad",
    "cup",
    "phone",
    "headphones",
    "book",
    "joystick"

]
SHOW_FPS = True             # Показывать FPS на экране

# --- Настройки модулей ---
USE_YOLO = True             # Использовать детектор объектов
USE_FACE = True             # Использовать распознавание лиц

# --- InsightFace ---
FACE_MODEL = "buffalo_l"    # Модель InsightFace
FACE_DIR = "faces"          # Папка с лицами
FACE_DB = "faces_db.json"   # Путь к JSON базе лиц
FACE_RECOGNITION_THRESHOLD = 0.35       # Чувствительность распознавания лиц
