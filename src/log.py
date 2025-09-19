import csv, os, time

class EpisodeLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","episode","return","length"])

    def log(self, episode:int, ep_return:float, ep_length:int):
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()), episode, ep_return, ep_length])
