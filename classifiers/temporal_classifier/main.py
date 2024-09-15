import time
import json

from kafka import KafkaConsumer
import pandas as pd


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


def read_from_queue():
    time.sleep(5)
    consumer = KafkaConsumer(
        'queue',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='temporal_consumer',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    buffer = Buffer(8)
    
    try:
        for message in consumer:
            row = message.value
            buffer.update(row)
            print(buffer.df)
            print(f"Current offset: {buffer.offset}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    read_from_queue()
