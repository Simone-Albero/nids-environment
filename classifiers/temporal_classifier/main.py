import time
import json

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import pandas as pd
import torch
from torchvision.transforms import Compose

from nids_framework.data import properties, utilities, transformation_builder
from nids_framework.model import transformer

class Buffer:

    def __init__(self, size: int) -> None:
        self.df: pd.DataFrame = pd.DataFrame()
        self.offset: int = 0
        self.size = size
    
    def update(self, row: dict) -> None: 
        new_row = pd.DataFrame([row])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        if len(self.df) > self.size:
            self.df = self.df.iloc[1:]
        self.offset += 1

    def is_ready(self) -> bool:
        return len(self.df) == self.size
    
def create_topic(topic_name: str, num_partitions: int, replication_factor: int):
    admin_client = KafkaAdminClient(bootstrap_servers="kafka:9092")
    
    topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=replication_factor
    )
    
    admin_client.create_topics(new_topics=[topic], validate_only=False)
    admin_client.close()


def read_and_predict():
    CONFIG_PATH = "shared/dataset/dataset_properties.ini"
    DATASET_NAME = "nf_ton_iot_v2_binary_anonymous"
    MODEL_PATH = "shared/models/temporal_bin.pt"

    CATEGORICAL_LEV = 32
    INPUT_SHAPE = 381
    EMBED_DIM = 256
    NUM_HEADS = 2
    NUM_LAYERS = 4
    DROPUT = 0.1
    FF_DIM = 128

    time.sleep(10)
    consumer = KafkaConsumer(
        'queue',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='temporal_consumer',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    producer = KafkaProducer(
        bootstrap_servers="kafka:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    create_topic("predictions", 1, 1)

    buffer = Buffer(8)
    model = transformer.TransformerClassifier(
        num_classes=1,
        input_dim=INPUT_SHAPE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPUT
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
            print("Message recived!")
            data = message.value

            record_id = data["record_id"]
            row = data["row"]
            connection_tuple = data["connection_tuple"]
            ground_truth = data["ground_truth"]

            buffer.update(row)

            if buffer.is_ready():
                numeric = torch.tensor(buffer.df[prop.numeric_features].values, dtype=torch.float32)
                categorical = torch.tensor(buffer.df[prop.categorical_features].values, dtype=torch.long)
                categorical_sample = lazy_transformation({"data": categorical})
                categorical = categorical_sample["data"].float()

                input_data = torch.cat((numeric, categorical), dim=-1).unsqueeze(0)

                with torch.no_grad():
                    prediction = model(input_data).squeeze(0)
            
                new_message = {"record_id": record_id, "connection_tuple": connection_tuple, "prediction": str(prediction.item()), "ground_truth": ground_truth}
                producer.send("predictions", value=new_message)
                print(f"Message sent successfully to topic 'predictions'")
                producer.flush()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    read_and_predict()



