import os
from dotenv import load_dotenv


def load_environment_variables():
    """
    Load environment variables from a .env file.
    """
    load_dotenv()

    # Example of accessing environment variables after loading
    KAFKA_IP = os.getenv("KAFKA_IP")
    KAFKA_PORT = os.getenv("KAFKA_PORT")
    if not KAFKA_IP or not KAFKA_PORT:
        raise ValueError("Required environment variables KAFKA_IP or KAFKA_PORT are not set.")
