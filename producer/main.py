import time
import pickle
import json

import pandas as pd
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from nids_framework.data import (
    properties,
    processor,
    utilities,
    transformation_builder,
)

CONFIG_PATH = "shared/dataset/dataset_properties.ini"
DATASET_NAME = "nf_ton_iot_v2_binary_anonymous"
DATASET_PATH = "shared/dataset/NF-ToN-IoT-V2-Test.csv"
TRAIN_META = "shared/dataset/train_meta.pkl"
CATEGORICAL_LEVEL = 32
BOUND = 100000000

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

def prepare_data():
    prop = properties.NamedDatasetProperties(CONFIG_PATH).get_properties(DATASET_NAME)
    df = pd.read_csv(DATASET_PATH, nrows=100000)
    
    with open(TRAIN_META, "rb") as f:
        min_values, max_values, unique_values = pickle.load(f)

    trans_builder = transformation_builder.TransformationBuilder()

    @trans_builder.add_step(order=1)
    def base_pre_processing(dataset):
        return utilities.base_pre_processing_row(dataset, prop, BOUND)

    @trans_builder.add_step(order=2)
    def log_pre_processing(dataset):
        return utilities.log_pre_processing_row(dataset, prop, min_values, max_values)

    @trans_builder.add_step(order=3)
    def categorical_conversion(dataset):
        return utilities.categorical_pre_processing_row(dataset, prop, unique_values, CATEGORICAL_LEVEL)

    @trans_builder.add_step(order=4)
    def binary_label_conversion(dataset):
        return utilities.binary_label_conversion_row(dataset, prop)
    
    @trans_builder.add_step(order=5)
    def split_data_for_torch(dataset):
        return utilities.split_data_for_torch_row(dataset, prop)

    transformations = trans_builder.build()

    proc = processor.Processor(transformations)

    return df, proc

def get_connection_tuple(row: pd.Series) -> pd.Series:
    connection_tuple = row[[
        "IPV4_SRC_ADDR",
        "L4_SRC_PORT",
        "IPV4_DST_ADDR",
        "L4_DST_PORT",
        "PROTOCOL"
    ]]
    return connection_tuple

def send_to_queue() -> None:
    if not wait_for_kafka("kafka:9092", 1000, 1):
        print("Kafka is not reachable after several retries. Exiting...")
        return

    producer = KafkaProducer(
        bootstrap_servers="kafka:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    
    create_topic("queue", 1, 1)

    df, proc = prepare_data()

    try:
        for index, row in df.iterrows():
            connection_tuple = get_connection_tuple(row)
            X, y = proc.apply(row)
            y = int(y)

            message = {
                "record_id": str(index),
                "row": X.to_dict(),
                "connection_tuple": connection_tuple.to_dict(),
                "ground_truth": str(y)
            }
            producer.send("queue", value=message)
            print(f"Message sent successfully to topic 'queue'")
            producer.flush()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        producer.close()

if __name__ == "__main__":
    send_to_queue()
