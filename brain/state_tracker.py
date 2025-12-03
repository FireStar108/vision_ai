# brain/state_tracker.py

class StateTracker:
    def __init__(self):
        self.events = []     # список всех событий
        self.last_seen = {}  # имя → bbox/time
        self.room_state = {} # имя → в комнате/вышел

    def update(self, events, timestamp):
        for ev in events:
            ev["time"] = timestamp
            self.events.append(ev)

            # отслеживание курения
            if ev["type"] == "smoking":
                self.last_seen["smoke"] = timestamp

            # телефон
            if ev["type"] == "holding_phone":
                self.last_seen["phone"] = timestamp

            # лица
            if ev["type"] == "face_match":
                self.last_seen[ev["name"]] = timestamp

            # двери
            if ev["type"] == "near_door":
                self.last_seen["door"] = timestamp

    def query(self, question):
        # пока заглушка — LLM заполнит
        return {
            "events": self.events,
            "last_seen": self.last_seen
        }