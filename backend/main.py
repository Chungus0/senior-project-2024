from flask import Flask, Response, jsonify, request
from confluent_kafka import Consumer
import os
import cv2
import json
import numpy as np
from environment import load_environment_variables
from flask_cors import CORS


def initialize_kafka_consumer():
    """
    Initialize the Kafka consumer with configuration.

    Returns:
        consumer: The Kafka Consumer object.
    """
    load_environment_variables()

    KAFKA_SERVER = f"{os.environ['KAFKA_IP']}:{os.environ['KAFKA_PORT']}"
    KAFKA_USERNAME = os.environ.get("KAFKA_USERNAME")
    KAFKA_PASSWORD = os.environ.get("KAFKA_PASSWORD")

    consumer_config = {
        "bootstrap.servers": KAFKA_SERVER,
        "group.id": "flask-consumer-group",
        "auto.offset.reset": "latest",  # Read only the latest messages
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

    return Consumer(consumer_config)


app = Flask(__name__)
CORS(app)


@app.route("/video_stream/<topic>")
def video_stream(topic):
    """
    Flask endpoint to stream video frames from Kafka.

    Args:
        topic: Kafka topic to consume video stream from.

    Returns:
        Response: Flask Response object containing video stream frames.
    """
    consumer = initialize_kafka_consumer()
    consumer.subscribe([topic])

    def generate():
        max_polls = 10  # Set a limit for retries if no messages are received
        poll_count = 0

        while poll_count < max_polls:
            msg = consumer.poll(1.0)
            if msg is None:
                poll_count += 1
                print(f"No message received. Poll count: {poll_count}")
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                break  # Stop if there’s a consumer error

            # Reset poll count upon receiving a message
            poll_count = 0
            if msg.topic() == topic:
                frame = np.frombuffer(msg.value(), dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                _, jpeg = cv2.imencode(".jpg", frame)
                if jpeg is not None:
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            else:
                print(f"Unexpected topic: {msg.topic()}")

        # Close consumer after polling limit or error
        consumer.close()
        print("Closed Kafka consumer after max polls or error.")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/pose_data/<topic>", methods=["GET"])
def pose_data(topic):
    """
    Flask endpoint to get the latest pose data from Kafka.

    Args:
        topic: Kafka topic to consume pose data from.

    Returns:
        jsonify: JSON response containing pose landmarks.
    """
    consumer = initialize_kafka_consumer()
    consumer.subscribe([topic])

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        if msg.topic() == topic:
            pose_json = msg.value().decode("utf-8")
            pose_data = json.loads(pose_json)
            return jsonify(pose_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
