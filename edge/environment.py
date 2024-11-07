from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from a .env file

def load_environment_variables():
    return {
        "KAFKA_IP": os.getenv("KAFKA_IP", "localhost"),
        "KAFKA_PORT": os.getenv("KAFKA_PORT", "9092"),
        "KAFKA_USERNAME": os.getenv("KAFKA_USERNAME"),
        "KAFKA_PASSWORD": os.getenv("KAFKA_PASSWORD")
    }
