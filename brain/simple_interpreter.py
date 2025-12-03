import subprocess
from brain.scene_to_text import make_human_readable

MODEL = "mistral:7b"

SYSTEM = """
Ты — модуль видеонаблюдения.
Говори КОРОТКО: 1–2 предложения.
Не используй технические термины (bbox, confidence).
Говори как человек: «В кадре человек», «Похоже это Матвей».
"""

def ask_llm(prompt: str):
    process = subprocess.Popen(
        ["ollama", "run", MODEL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate(prompt)
    return out.strip()


def describe_scene(objects, faces):
    scene_text = make_human_readable(objects, faces)

    prompt = f"""
{SYSTEM}

Вот что видит камера:
{scene_text}

Опиши ситуацию коротко.
"""
    return ask_llm(prompt)