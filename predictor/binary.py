import time
import json
import threading

from concurrent.futures import ThreadPoolExecutor
import torch

from modules.kafka_utilities import Consumer
from nids_framework.training import metrics

STAMP_THRESHOLD = 90
NUM_CLASSIFIERS = 3

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
        self._blacklist = set()

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
    
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._data) == 0

    def __str__(self) -> str:
        with self._lock:
            return str({k: (v.timestamp, v.predictions) for k, v in self._data.items()})

def majority_prediction(
    record_id: int, predictions: list[float], record_registry: dict, metric: metrics.Metric
) -> None:
    
    record_info = record_registry.get(record_id)

    positive_count = sum(1 for pred in predictions if pred >= 0.5)

    if positive_count >= 2:
        aggregated_prediction = 1.0
    else:
        aggregated_prediction = 0.0 

    metric.step(torch.tensor(aggregated_prediction).unsqueeze(0), torch.tensor(record_info[1]).unsqueeze(0))
    
    if aggregated_prediction != record_info[1]:
        print(f"Record ID: {record_id} -> predicted: {predictions} [{aggregated_prediction}] over {record_info[1]}")
        metric.compute_metrics()
        print(metric, flush=True)

def max_confidence_prediction(
    record_id: int, predictions: list[float], record_registry: dict, metric: metrics.Metric
) -> None:
    
    record_info = record_registry.get(record_id)

    threshold = 0.5
    confidence_scores = [abs(pred - threshold) for pred in predictions]
    
    max_confidence_index = confidence_scores.index(max(confidence_scores))
    aggregated_prediction = predictions[max_confidence_index]

    metric.step(torch.tensor(aggregated_prediction).unsqueeze(0), torch.tensor(record_info[1]).unsqueeze(0))
    
    real_prediction = 1.0 if aggregated_prediction >= 0.5 else 0.0
    if real_prediction != record_info[1]:
        print(f"Record ID: {record_id} -> predicted: {predictions} [{aggregated_prediction}] over {record_info[1]}")
        metric.compute_metrics()
        print(metric, flush=True)

def weighted_confidence_prediction(
    record_id: int, predictions: list[float], record_registry: dict, metric: metrics.Metric
) -> None:
    
    record_info = record_registry.get(record_id)

    threshold = 0.5

    confidence_scores = [2 * abs(pred - threshold) for pred in predictions] 
    weighted_sum = sum(conf * pred for conf, pred in zip(confidence_scores, predictions))
    total_confidence = sum(confidence_scores)

    aggregated_prediction = weighted_sum / total_confidence if total_confidence > 0 else threshold

    metric.step(torch.tensor(aggregated_prediction).unsqueeze(0), torch.tensor(record_info[1]).unsqueeze(0))
    
    real_prediction = 1.0 if aggregated_prediction >= 0.6 else 0.0
    if real_prediction != record_info[1]:
        print(f"Record ID: {record_id} -> predicted: {predictions} [{aggregated_prediction}] over {record_info[1]}")
        metric.compute_metrics()
        print(metric, flush=True)

def weighted_prediction(
    record_id: int, predictions: list[float], record_registry: dict, metric: metrics.Metric
) -> None:
    
    record_info = record_registry.get(record_id)
    weights = [0.3, 0.3, 0.5]
    aggregated_prediction = sum(w * pred for w, pred in zip(weights, predictions)) / sum(weights)

    metric.step(torch.tensor(aggregated_prediction).unsqueeze(0), torch.tensor(record_info[1]).unsqueeze(0))
    
    real_prediction = 1.0 if aggregated_prediction >= 0.6 else 0.0
    if real_prediction != record_info[1]:
        print(f"Record ID: {record_id} -> predicted: {predictions} [{aggregated_prediction}] over {record_info[1]}")
        metric.compute_metrics()
        print(metric, flush=True)

def max_prediction(
    record_id: int, predictions: list[float], record_registry: dict, metric: metrics.Metric
) -> None:
    record_info = record_registry.get(record_id)
    aggregated_prediction = max(predictions)
    
    metric.step(torch.tensor(aggregated_prediction).unsqueeze(0), torch.tensor(record_info[1]).unsqueeze(0))

    real_prediction = 1.0 if aggregated_prediction >= 0.5 else 0.0
    if real_prediction != record_info[1]:
        print(f"Record ID: {record_id} -> predicted: {predictions} [{aggregated_prediction}] over {record_info[1]}")
        metric.compute_metrics()
        print(metric, flush=True)

def min_prediction(
    record_id: int, predictions: list[float], record_registry: dict, metric: metrics.Metric
) -> None:
    record_info = record_registry.get(record_id)
    aggregated_prediction = min(predictions)

    metric.step(torch.tensor(aggregated_prediction).unsqueeze(0), torch.tensor(record_info[1]).unsqueeze(0))
    
    real_prediction = 1.0 if aggregated_prediction >= 0.5 else 0.0
    if real_prediction != record_info[1]:
        print(f"Record ID: {record_id} -> predicted: {predictions} [{aggregated_prediction}] over {record_info[1]}")
        metric.compute_metrics()
        print(metric, flush=True)

def handle_predictions(prediction_map: PredictionMap, record_registry: dict, threshold: float, metric: metrics.Metric, callback: callable, stop_event: threading.Event) -> None:
    while True:

        if stop_event.is_set() and prediction_map.is_empty():
            break

        current_time = time.time()
        old_entries = []

        with prediction_map._lock:
            old_entries = [(record_id, entry.predictions)
                           for record_id, entry in prediction_map._data.items()
                           if current_time - entry.timestamp > threshold or len(entry.predictions) == NUM_CLASSIFIERS]

        for record_id, predictions in old_entries:
            prediction_map.remove(record_id)
            if len(predictions) >= 2:
                callback(record_id, predictions, record_registry, metric)

        time.sleep(1)


def aggregate_predictions() -> None:
    consumer = Consumer("kafka:9092", "predictor", lambda x: json.loads(x.decode('utf-8')))
    consumer.subscribe(['predictions'])

    metric = metrics.BinaryClassificationMetric(threshold=0.6)
    prediction_map = PredictionMap()
    record_registry = {}

    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        monitor_future = executor.submit(handle_predictions, prediction_map, record_registry, STAMP_THRESHOLD, metric, weighted_confidence_prediction, stop_event)

        try:
            eos_cuont = 0
            for message in consumer:
                #print("Message received!")
                data = message.value

                if data == "EOS":
                    eos_cuont += 1
                    print(f"Arrived 'EOS' N.{eos_cuont}", flush=True)
                    if eos_cuont == NUM_CLASSIFIERS:
                        print("Stream ended", flush=True)
                        stop_event.set()
                        break
                else:
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
            monitor_future.result()
            metric.compute_metrics()
            print(metric)
            consumer.close()

if __name__ == "__main__":
    aggregate_predictions()
