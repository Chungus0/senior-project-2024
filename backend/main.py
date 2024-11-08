from flask import Flask, Response, stream_with_context
from confluent_kafka import Consumer
import os
import cv2
import numpy as np
from environment import load_environment_variables
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def initialize_kafka_consumer(topic):
    """
    Initialize a Kafka consumer for the given topic.
    """
    load_environment_variables()
    KAFKA_SERVER = f"{os.environ['KAFKA_IP']}:{os.environ['KAFKA_PORT']}"
    KAFKA_USERNAME = os.environ.get("KAFKA_USERNAME")
    KAFKA_PASSWORD = os.environ.get("KAFKA_PASSWORD")

    consumer_config = {
        "bootstrap.servers": KAFKA_SERVER,
        "group.id": f"flask-consumer-group-{topic}",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "session.timeout.ms": 10000,
    }

    if KAFKA_USERNAME and KAFKA_PASSWORD:
        consumer_config.update(
            {
                "security.protocol": "SASL_PLAINTEXT",
                "sasl.mechanisms": "PLAIN",
                "sasl.username": KAFKA_USERNAME,
                "sasl.password": KAFKA_PASSWORD,
            }
        )

    consumer = Consumer(consumer_config)
    consumer.subscribe([topic])
    return consumer


@app.route("/video_stream/<topic>")
@stream_with_context
def video_stream(topic):
    """
    Endpoint to stream video frames from Kafka.
    """

    def generate():
        consumer = initialize_kafka_consumer(topic)
        print(f"Started Kafka consumer for topic {topic}")

        try:
            while True:
                msg = consumer.poll(0.5)
                if msg is None:
                    yield (b"--frame\r\n\r\n")  # Heartbeat to keep connection alive
                    continue

                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    yield (b"--frame\r\n\r\n")
                    continue

                if msg.topic() == topic:
                    frame = np.frombuffer(msg.value(), dtype=np.uint8)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    _, jpeg = cv2.imencode(".jpg", frame)
                    if jpeg is not None:
                        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

        except Exception as e:
            print(f"Error in Kafka consumer for topic {topic}: {e}")
        finally:
            consumer.close()
            print(f"Closed Kafka consumer for topic {topic}")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
