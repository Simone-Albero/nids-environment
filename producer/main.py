import pickle
import json

import pandas as pd

from modules.kafka_utilities import Producer
from nids_framework.data import (
    properties,
    processor,
    utilities,
    transformation_builder,
)

CONFIG_PATH = "shared/dataset/dataset_properties.ini"
DATASET_NAME = "nf_unsw_nb15_v2_binary_anonymous"

DATASET_PATH = "shared/dataset/unsw/Test.csv"
TRAIN_META = "shared/dataset/unsw/train_meta.pkl"
TEST_LIMIT = 5000

CATEGORICAL_LEVEL = 32
BOUND = 100000000

def prepare_data():
    prop = properties.NamedDatasetProperties(CONFIG_PATH).get_properties(DATASET_NAME)
    df = pd.read_csv(DATASET_PATH, nrows=TEST_LIMIT)
    
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
    producer = Producer("kafka:9092", lambda v: json.dumps(v).encode("utf-8"))
    producer.create_topic("queue", 1, 1)

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
            producer.flush()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        producer.close()

if __name__ == "__main__":
    send_to_queue()
