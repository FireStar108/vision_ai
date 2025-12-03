from brain.simple_interpreter import describe_scene

dummy_objects = [
    {"label": "person", "confidence": 0.9, "bbox": [10, 10, 100, 200]},
    {"label": "headphones", "confidence": 0.6, "bbox": [120, 50, 200, 150]},
]

dummy_faces = [
    {"name": "matvej", "similarity": 0.9, "bbox": [20, 20, 80, 180]}
]

print(describe_scene(
    [{"label": "person", "confidence": 0.9, "bbox": [1,2,3,4]}],
    [{"name": "matvej", "similarity": 0.9, "bbox": [1,2,3,4]}]
))