import time
import json
import threading

from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaConsumer, KafkaAdminClient


def check_kafka_connection(bootstrap_servers: str) -> bool:
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        admin_client.list_topics()
        admin_client.close()
        return True
    except Exception as e:
        print(f"Kafka connection error: {e}")
        return False

def wait_for_kafka(bootstrap_servers: str, max_retries: int, retry_interval: int) -> bool:
    for _ in range(max_retries):
        if check_kafka_connection(bootstrap_servers):
            return True
        print(f"Kafka not reachable, retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)
    return False

class PredictionMap:
    class Entry:
        def __init__(self, timestamp: float) -> None:
            self.timestamp = timestamp
            self.predictions: list[float] = []

        def add(self, prediction: float) -> None:
            self.predictions.append(prediction)

    def __init__(self) -> None:
        self._data: dict[int, self.Entry] = {}
        self._lock = threading.Lock()

    def put(self, record_id: int, prediction: float) -> None:
        with self._lock:
            if record_id not in self._data:
                self._data[record_id] = self.Entry(time.time())
            self._data[record_id].add(prediction)

    def get(self, record_id: int) -> tuple[float, list[float]]:
        with self._lock:
            entry = self._data.get(record_id)
            return (entry.timestamp, entry.predictions) if entry else (None, [])

    def remove(self, record_id: int) -> None:
        with self._lock:
            self._data.pop(record_id, None)

    def __str__(self) -> str:
        with self._lock:
            return str({k: (v.timestamp, v.predictions) for k, v in self._data.items()})


def make_prediction(record_id: int, predictions: list[float], record_registry: dict) -> None:
    print(record_id)
    print(predictions, record_registry.get(record_id))


def handle_predictions(prediction_map: PredictionMap, record_registry: dict, threshold: float, callback: callable) -> None:
    while True:
        current_time = time.time()
        old_entries = []

        with prediction_map._lock:
            for record_id, entry in list(prediction_map._data.items()):
                if current_time - entry.timestamp > threshold:
                    old_entries.append((record_id, entry.predictions))

        for record_id, predictions in old_entries:
            prediction_map.remove(record_id)
            callback(record_id, predictions, record_registry)

        time.sleep(1)


def read_predictions() -> None:
    if not wait_for_kafka("kafka:9092", 1000, 1):
        print("Kafka is not reachable after several retries. Exiting...")
        return

    consumer = KafkaConsumer(
        'predictions',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='temporal_consumer',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    prediction_map = PredictionMap()
    record_registry = {}
    STAMP_THRESHOLD = 5

    with ThreadPoolExecutor(max_workers=2) as executor:
        monitor_future = executor.submit(handle_predictions, prediction_map, record_registry, STAMP_THRESHOLD, make_prediction)

        try:
            for message in consumer:
                print("Message received!")
                data = message.value
                record_id = data["record_id"]
                connection_tuple = data["connection_tuple"]
                prediction = data["prediction"]
                ground_truth = data["ground_truth"]

                print(f"prediction: {prediction} gt: {ground_truth}")

                prediction_map.put(record_id, prediction)

                if record_id not in record_registry:
                    record_registry[record_id] = (connection_tuple, ground_truth)
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            consumer.close()
            monitor_future.cancel()
            monitor_future.result()  # Ensure handle_predictions completes

if __name__ == "__main__":
    read_predictions()
