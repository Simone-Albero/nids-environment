import time
import pickle
import json

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import pandas as pd

from nids_framework.data import (
    properties,
    processor,
    utilities,
    transformation_builder,
)

def get_preprocessed_df() -> pd.DataFrame:
    CONFIG_PATH = "shared/dataset_properties.ini"
    DATASET_NAME = "nf_ton_iot_v2_binary_anonymous"
    DATASET_PATH = "shared/NF-ToN-IoT-V2-Test.csv"
    TRAIN_META = "shared/train_meta.pkl"
    
    CATEGORICAL_LEV = 32
    BOUND = 100000000

    prop = properties.NamedDatasetProperties(CONFIG_PATH).get_properties(DATASET_NAME)
    df = pd.read_csv(DATASET_PATH)
    with open(TRAIN_META, "rb") as f:
        min_values, max_values, unique_values = pickle.load(f)

    trans_builder = transformation_builder.TransformationBuilder()

    @trans_builder.add_step(order=1)
    def base_pre_processing(dataset, properties):
        utilities.base_pre_processing(dataset, properties, BOUND)

    @trans_builder.add_step(order=2)
    def log_pre_processing(dataset, properties):
        utilities.log_pre_processing(dataset, properties, min_values, max_values)

    @trans_builder.add_step(order=3)
    def categorical_conversion(dataset, properties):
        utilities.categorical_pre_processing(
            dataset, properties, unique_values, CATEGORICAL_LEV
        )

    @trans_builder.add_step(order=4)
    def bynary_label_conversion(dataset, properties):
        utilities.bynary_label_conversion(dataset, properties)

    transformations = trans_builder.build()
    proc = processor.Processor(df, prop)
    proc.transformations = transformations
    proc.apply()
    X, y = proc.build()
    return X, y

def create_topic(topic_name: str, num_partitions: int, replication_factor: int):
    admin_client = KafkaAdminClient(bootstrap_servers="kafka:9092")
    
    topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=replication_factor
    )
    
    admin_client.create_topics(new_topics=[topic], validate_only=False)
    admin_client.close()

def send_to_queue() -> None:
    time.sleep(5)
    X, y = get_preprocessed_df()

    create_topic('queue', 1, 1)
    create_topic('ground_truth', 1, 1)

    producer = KafkaProducer(
        bootstrap_servers="kafka:9092",
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    try:
        for index, row in X.iterrows():
            producer.send("queue", value=row.to_dict())
            print(f"Message sent successfully to topic 'queue'")

            message = {'true_label': str(y.iloc[index])}
            producer.send("ground_truth", value=message)
            print(f"Message sent successfully to topic 'ground_truth'")

            producer.flush()
            time.sleep(1)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        producer.close()

if __name__ == "__main__":
    send_to_queue()
