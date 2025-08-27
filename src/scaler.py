import yaml
import time
from threading import Thread

class ElasticScaler:
    def __init__(self, config_path="/app/models.yml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.models = {m["name"]: m for m in cfg["models"]}
        self.replicas = {m["name"]: m.get("min_replicas", 1) for m in cfg["models"]}
        self.queues = {m["name"]: 0 for m in cfg["models"]}  # placeholder for queue lengths
        self.running = False

    def simulate_request(self, model_name: str):
        """Simulate a request hitting the queue."""
        self.queues[model_name] += 1

    def loop(self):
        while self.running:
            for model_name, cfg in self.models.items():
                qlen = self.queues[model_name]
                current = self.replicas[model_name]

                # Scale up
                if qlen > cfg["scale_up_threshold"] and current < cfg["max_replicas"]:
                    self.replicas[model_name] += 1
                    print(f"[Scaler] Scaled UP {model_name} to {self.replicas[model_name]} replicas")

                # Scale down
                elif qlen == 0 and current > cfg["min_replicas"]:
                    time.sleep(cfg["scale_down_delay"])
                    if self.queues[model_name] == 0:
                        self.replicas[model_name] -= 1
                        print(f"[Scaler] Scaled DOWN {model_name} to {self.replicas[model_name]} replicas")

                # reset queue after handling
                self.queues[model_name] = 0

            time.sleep(5)

    def start(self):
        if not self.running:
            self.running = True
            Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False

    def get_status(self):
        return {m: {"replicas": self.replicas[m], "queue": self.queues[m]} for m in self.models}