import time
import json
import threading

from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaConsumer, KafkaAdminClient
import torch

from nids_framework.training import metrics

STAMP_THRESHOLD = 20

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
        self._data: dict[str, self.Entry] = {}
        self._lock = threading.Lock()
        self._blacklist = set()  # Use a set for O(1) lookups

    def put(self, record_id: str, prediction: str) -> None:
        with self._lock:
            if record_id in self._blacklist:
                return

            entry = self._data.setdefault(record_id, self.Entry(time.time()))
            entry.add(float(prediction))

    def get(self, record_id: str) -> tuple[float, list[float]]:
        with self._lock:
            entry = self._data.get(record_id)
            return (entry.timestamp, entry.predictions) if entry else (None, [])

    def remove(self, record_id: str) -> None:
        with self._lock:
            if record_id in self._data:
                self._data.pop(record_id, None)
                self._blacklist.add(record_id)

    def __str__(self) -> str:
        with self._lock:
            return str({k: (v.timestamp, v.predictions) for k, v in self._data.items()})


def make_prediction(
    record_id: int, predictions: list[float], record_registry: dict, metric: metrics.Metric
) -> None:
    print(f"Record ID: {record_id} -> Predictions: {predictions}")
    
    record_info = record_registry.get(record_id)
    print(f"Record Info: {record_info}")

    threshold = 0.5

    confidence_scores = [2 * abs(pred - threshold) for pred in predictions] 
    weighted_sum = sum(conf * pred for conf, pred in zip(confidence_scores, predictions))
    total_confidence = sum(confidence_scores)

    aggregated_prediction = weighted_sum / total_confidence if total_confidence > 0 else threshold
    print(f"Aggregated Prediction: {aggregated_prediction}")

    metric.step(torch.tensor(aggregated_prediction).unsqueeze(0), torch.tensor(record_info[1]).unsqueeze(0))
    metric.compute_metrics()
    print(metric)



def handle_predictions(prediction_map: PredictionMap, record_registry: dict, threshold: float, metric: metrics.Metric, callback: callable) -> None:
    while True:
        current_time = time.time()
        old_entries = []

        with prediction_map._lock:
            old_entries = [(record_id, entry.predictions)
                           for record_id, entry in prediction_map._data.items()
                           if current_time - entry.timestamp > threshold]

        for record_id, predictions in old_entries:
            prediction_map.remove(record_id)
            callback(record_id, predictions, record_registry, metric)


def read_predictions() -> None:
    if not wait_for_kafka("kafka:9092", 1000, 1):
        print("Kafka is not reachable after several retries. Exiting...")
        return

    consumer = KafkaConsumer(
        'predictions',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='predictor',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    metric = metrics.BinaryClassificationMetric()
    prediction_map = PredictionMap()
    record_registry = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        monitor_future = executor.submit(handle_predictions, prediction_map, record_registry, STAMP_THRESHOLD, metric, make_prediction)

        try:
            for message in consumer:
                print("Message received!")
                data = message.value
                record_id = data["record_id"]
                connection_tuple = data["connection_tuple"]
                prediction = data["prediction"]
                ground_truth = data["ground_truth"]

                prediction_map.put(record_id, prediction)

                if record_id not in record_registry:
                    record_registry[record_id] = (connection_tuple, float(ground_truth))
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            consumer.close()
            monitor_future.cancel()
            monitor_future.result()  # Ensure handle_predictions completes

            metric.compute_metrics()
            print(metric)

if __name__ == "__main__":
    read_predictions()
