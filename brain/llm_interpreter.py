import subprocess
import json
import tempfile

MODEL = "mistral:7b"

def ask_llm(prompt: str):
    process = subprocess.Popen(
        ["ollama", "run", MODEL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate(prompt)
    if err:
        print("[LLM][stderr]:", err)
    return out.strip()



def interpret_events(events, question):
    system_prompt = """
Ты — эксперт по анализу видеособытий.
Формат ответа: одна короткая фраза, максимум 1 предложение.
Никаких пояснений. Никакой философии. Прямо по фактам.
"""

    prompt = f"""
{system_prompt}

Данные событий:
{json.dumps(events, ensure_ascii=False, indent=2)}

Вопрос:
{question}

Ответ:
"""

    return ask_llm(prompt)