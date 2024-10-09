import json

from modules.kafka_utilities import Producer, Consumer
from modules.live_classifier import LiveClassifier

CONFIG_PATH = "shared/dataset/dataset_properties.ini"
DATASET_NAME = "nf_unsw_nb15_v2_binary_anonymous"
MODEL_PATH = "shared/models/unsw/Reconnaissance_ip_dst.pt"
#MODEL_PATH = "shared/models/unsw/DoS_ip_dst.pt"

TARGET = "IPV4_DST_ADDR"
ID = 1

CATEGORICAL_LEV = 32
INPUT_SHAPE = 382

WINDOW_SIZE = 10
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.2
FF_DIM = 256
        
def read_and_predict() -> None:
    producer = Producer("kafka:9092", lambda v: json.dumps(v).encode("utf-8"))
    producer.create_topic("predictions", 1, 1)

    consumer = Consumer("kafka:9092", "ip_dst_consumer", lambda x: json.loads(x.decode('utf-8')))
    consumer.subscribe(['queue'])

    classifier = LiveClassifier(MODEL_PATH, CONFIG_PATH, DATASET_NAME,
                 WINDOW_SIZE, CATEGORICAL_LEV, INPUT_SHAPE,
                 EMBED_DIM, NUM_HEADS, NUM_LAYERS,
                 DROPOUT, FF_DIM)
    
    try:
        for message in consumer:
            print("Message received!")
            data = message.value

            if data == "EOS":
                print("Stream ended")
                producer.send("predictions", value="EOS")
                producer.flush()
                break

            record_id = data["record_id"]
            row = data["row"]
            connection_tuple = data["connection_tuple"]
            ground_truth = data["ground_truth"]

            prediction = classifier.process(row, connection_tuple[TARGET])

            if prediction:
                new_message = {
                    "record_id": record_id,
                    "connection_tuple": connection_tuple,
                    "prediction": str(prediction.item()),
                    "ground_truth": ground_truth,
                    "id": ID
                }
                producer.send("predictions", new_message)
                print(f"Message '{record_id}' sent successfully to topic 'predictions'")
                producer.flush()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    read_and_predict()
