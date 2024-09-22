import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

class Producer:
    def __init__(self, bootstrap_servers: str, value_serializer: callable):
        self.bootstrap_servers = bootstrap_servers

        if not self.wait_for_kafka(1000, 1):
            raise ConnectionError("Kafka is not reachable after several retries. Exiting...")

        self.producer = KafkaProducer(
            bootstrap_servers="kafka:9092",
            value_serializer=value_serializer
        )

    def check_kafka_connection(self) -> bool:
        try:
            admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            admin_client.list_topics()
            admin_client.close()
            return True
        except Exception as e:
            print(f"Kafka connection error: {e}")
            return False

    def wait_for_kafka(self, max_retries: int, retry_interval: int) -> bool:
        for attempt in range(max_retries):
            if self.check_kafka_connection():
                print("Kafka is reachable.")
                return True
            print(f"Attempt {attempt + 1}/{max_retries}: Kafka not reachable, retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        print("Max retries reached. Kafka is not reachable.")
        return False

    def create_topic(self, topic_name: str, num_partitions: int, replication_factor: int) -> None:
        admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
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
        except Exception as e:
            print(f"Error creating topic '{topic_name}': {e}")
        finally:
            admin_client.close()

    def send(self, topic: str, value: any):
        if not self.producer:
            print("KafkaProducer is not initialized. Call start_producer() first.")
            return

        try:
            self.producer.send(topic, value=value)
            print(f"Message sent to topic {topic}.")
        except Exception as e:
            print(f"Error sending message to Kafka: {e}")

    def flush(self):
        self.producer.flush()

    def close(self):
        if self.producer:
            self.producer.close()
            print("KafkaProducer closed.")


class Consumer:
    def __init__(self, bootstrap_servers: str, group_id: str, value_deserializer: callable):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id

        if not self.wait_for_kafka(1000, 1):
            raise ConnectionError("Kafka is not reachable after several retries. Exiting...")

        self.consumer = KafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=value_deserializer
        )

    def check_kafka_connection(self) -> bool:
        try:
            admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            admin_client.list_topics()
            admin_client.close()
            return True
        except Exception as e:
            print(f"Kafka connection error: {e}")
            return False

    def wait_for_kafka(self, max_retries: int, retry_interval: int) -> bool:
        for attempt in range(max_retries):
            if self.check_kafka_connection():
                print("Kafka is reachable.")
                return True
            print(f"Attempt {attempt + 1}/{max_retries}: Kafka not reachable, retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        print("Max retries reached. Kafka is not reachable.")
        return False

    def subscribe(self, topics: list):
        self.consumer.subscribe(topics)
        print(f"Subscribed to topics: {topics}")

    def close(self):
        if self.consumer:
            self.consumer.close()
            print("KafkaConsumer closed.")

    def __iter__(self):
        return self.consumer
