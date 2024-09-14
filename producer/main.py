import time
import json

from kafka import KafkaProducer
import pandas as pd

def send_to_queue(file_path):
    time.sleep(5)
    producer = KafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    try:
        df = pd.read_csv(file_path)
        
        for index, row in df.iterrows():
            producer.send('queue', value=row.to_dict())
            print(f"Sent row: {row.to_dict()}")
            producer.flush()
            time.sleep(1)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        producer.close()

if __name__ == "__main__":
    send_to_queue('shared/NF-ToN-IoT-V2-Test.csv')
