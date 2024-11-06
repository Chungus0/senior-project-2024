import os
from dotenv import load_dotenv

def load_environment_variables():
    """
    Load environment variables from a .env file.
    """
    load_dotenv()

    kafka_ip = os.getenv("KAFKA_IP")
    kafka_port = os.getenv("KAFKA_PORT")
    kafka_username = os.getenv("KAFKA_USERNAME")
    kafka_password = os.getenv("KAFKA_PASSWORD")

    print("Kafka Configuration:")
    print(f"KAFKA_IP: {kafka_ip}")
    print(f"KAFKA_PORT: {kafka_port}")
    if kafka_username:
        print(f"KAFKA_USERNAME: {kafka_username}")
    if kafka_password:
        print("KAFKA_PASSWORD: [HIDDEN]")

if __name__ == "__main__":
    load_environment_variables()