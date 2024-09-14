import time
import json

from kafka import KafkaConsumer

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
    
    try:
        for message in consumer:
            row = message.value
            print(f"Reading row: {row}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    read_from_queue()
