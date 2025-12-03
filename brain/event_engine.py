# brain/event_engine.py

def iou(box1, box2):
    # пересечение прямоугольников
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter)


def analyze_events(objects, faces):
    events = []

    persons = [o for o in objects if o["label"] == "person"]
    phones = [o for o in objects if o["label"] in ("phone", "cell phone")]
    cigarettes = [o for o in objects if o["label"] in ("cigarette", "smoke")]
    doors = [o for o in objects if o["label"] == "door"]

    # связь человек → лицо
    for p in persons:
        for f in faces:
            if iou(p["bbox"], f["bbox"]) > 0.2:
                events.append({
                    "type": "face_match",
                    "person_bbox": p["bbox"],
                    "name": f["name"],
                    "similarity": f["similarity"]
                })

    # человек + телефон
    for p in persons:
        for ph in phones:
            if iou(p["bbox"], ph["bbox"]) > 0.05:
                events.append({
                    "type": "holding_phone",
                    "person_bbox": p["bbox"],
                    "object_bbox": ph["bbox"]
                })

    # человек + дым
    for p in persons:
        for c in cigarettes:
            if iou(p["bbox"], c["bbox"]) > 0.05:
                events.append({
                    "type": "smoking",
                    "person_bbox": p["bbox"],
                    "object_bbox": c["bbox"]
                })

    # человек у двери
    for p in persons:
        for d in doors:
            if iou(p["bbox"], d["bbox"]) > 0.05:
                events.append({
                    "type": "near_door",
                    "person_bbox": p["bbox"]
                })

    return events
