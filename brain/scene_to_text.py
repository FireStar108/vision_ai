def make_human_readable(objects, faces):
    desc = []

    for o in objects:
        name = o["label"]
        conf = o["confidence"]

        if name == "person":
            if conf > 0.7:
                desc.append("В кадре человек.")
            else:
                desc.append("Вероятно человек в кадре.")
        else:
            desc.append(f"Обнаружен объект: {name}.")

    for f in faces:
        person = f["name"]
        sim = f["similarity"]

        if person != "Неизвестный" and sim > 0.3:
            desc.append(f"Лицо похоже на {person}.")
        else:
            desc.append("Обнаружено лицо, но личность неизвестна.")

    return " ".join(desc)