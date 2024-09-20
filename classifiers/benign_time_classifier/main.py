import time
import json
from collections import deque

import pandas as pd
import torch
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from torchvision.transforms import Compose

from nids_framework.data import properties, utilities, transformation_builder
from nids_framework.model import transformer

CONFIG_PATH = "shared/dataset/dataset_properties.ini"
DATASET_NAME = "nf_unsw_nb15_v2_binary_anonymous"
MODEL_PATH = "shared/models/unsw/benign_time.pt"

CATEGORICAL_LEV = 32
INPUT_SHAPE = 382
EMBED_DIM = 256
NUM_HEADS = 2
NUM_LAYERS = 4
DROPOUT = 0.1
FF_DIM = 128
WINDOW_SIZE = 8

def create_topic(topic_name: str, num_partitions: int, replication_factor: int) -> None:
    admin_client = KafkaAdminClient(bootstrap_servers="kafka:9092")
    topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=replication_factor
    )
    
    try:
        admin_client.create_topics(new_topics=[topic], validate_only=False)
        print(f"Topic '{topic_name}' created successfully.")
    except TopicAlreadyExistsError:
        print(f"Topic '{topic_name}' already exists.")
    finally:
        admin_client.close()

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
    
class Buffer:
    def __init__(self, size: int) -> None:
        self.rows = deque(maxlen=size)
        self.size = size
    
    def update(self, row: dict) -> None: 
        self.rows.append(row)

        if len(self.rows) == self.size:
            return pd.DataFrame(self.rows)
        
        return None

def read_and_predict() -> None:
    if not wait_for_kafka("kafka:9092", 1000, 1):
        print("Kafka is not reachable after several retries. Exiting...")
        return

    consumer = KafkaConsumer(
        'queue',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='time_consumer',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    producer = KafkaProducer(
        bootstrap_servers="kafka:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    create_topic("predictions", 1, 1)

    buffer = Buffer(WINDOW_SIZE)
    model = transformer.TransformerClassifier(
        num_classes=1,
        input_dim=INPUT_SHAPE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    prop = properties.NamedDatasetProperties(CONFIG_PATH).get_properties(DATASET_NAME)
    
    trans_builder = transformation_builder.TransformationBuilder()
    
    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEV):
        return utilities.one_hot_encoding(sample, categorical_levels)
    
    transformations = trans_builder.build()
    lazy_transformation = Compose(transformations)
    
    try:
        for message in consumer:
            print("Message received!")
            data = message.value

            record_id = data["record_id"]
            row = data["row"]
            connection_tuple = data["connection_tuple"]
            ground_truth = data["ground_truth"]

            window = buffer.update(row)

            if window is not None:
                numeric = torch.tensor(window[prop.numeric_features].values, dtype=torch.float32)
                categorical = torch.tensor(window[prop.categorical_features].values, dtype=torch.long)
                categorical = lazy_transformation(categorical)
                categorical = categorical.float()
                input_data = torch.cat((numeric, categorical), dim=-1).unsqueeze(0)

                with torch.no_grad():
                    prediction = model(input_data).squeeze(0)
            
                new_message = {
                    "record_id": record_id,
                    "connection_tuple": connection_tuple,
                    "prediction": str(prediction.item()),
                    "ground_truth": ground_truth
                }
                producer.send("predictions", value=new_message)
                print(f"Message '{record_id}' sent successfully to topic 'predictions'")
                producer.flush()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    read_and_predict()
