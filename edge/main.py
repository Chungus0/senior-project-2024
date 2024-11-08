import cv2
import mediapipe as mp
from confluent_kafka import Producer
import json
import signal
import sys
from environment import load_environment_variables
import time


def initialize_kafka_producer():
    """
    Initialize the Kafka producer with configuration.

    Returns:
        producer: The Kafka Producer object.
    """
    environment = load_environment_variables()

    KAFKA_SERVER = f"{environment['KAFKA_IP']}:{environment['KAFKA_PORT']}"
    KAFKA_USERNAME = environment.get("KAFKA_USERNAME")
    KAFKA_PASSWORD = environment.get("KAFKA_PASSWORD")

    producer_config = {
        "bootstrap.servers": KAFKA_SERVER,
        "linger.ms": 1,  # Reduce delay for immediate transmission
        "batch.size": 8192,  # Smaller batch size for faster sending
        "compression.type": "lz4",  # Compress messages to reduce network load
        "acks": 1,
    }

    if KAFKA_USERNAME and KAFKA_PASSWORD:
        producer_config.update(
            {
                "security.protocol": "SASL_PLAINTEXT",
                "sasl.mechanisms": "PLAIN",
                "sasl.username": KAFKA_USERNAME,
                "sasl.password": KAFKA_PASSWORD,
            }
        )

    return Producer(producer_config)


def delivery_report(err, msg):
    """
    Callback function for Kafka message delivery report.

    Args:
        err: Error information if message delivery failed.
        msg: Kafka message that was sent.
    """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")


def signal_handler(sig, frame):
    """
    Handle graceful shutdown of the application.

    Args:
        sig: Signal number.
        frame: Current stack frame.
    """
    print("\nGracefully shutting down...")
    cap.release()
    pose.close()
    producer.flush()  # Ensure all messages are sent before exiting
    sys.exit(0)


def initialize_video_capture():
    """
    Initialize the video capture from the webcam.

    Returns:
        cap: The VideoCapture object for reading frames from the webcam.
    """
    return cv2.VideoCapture(0)


def initialize_mediapipe_pose():
    """
    Initialize MediaPipe Pose for pose detection.

    Returns:
        pose: The MediaPipe Pose object.
    """
    return mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process_frame(frame, send_frame_num):
    """
    Process a video frame to detect pose landmarks and send the data to Kafka.

    Args:
        frame: The video frame to process.
        send_frame_num: Boolean flag to indicate if the current frame should be sent.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks and send_frame_num:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        pose_data = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in results.pose_landmarks.landmark
        ]

        pose_json = json.dumps(pose_data)

        try:
            producer.produce(KAFKA_TOPIC_POSE, pose_json.encode("utf-8"), callback=delivery_report)
        except BufferError:
            print("Local producer queue is full (pose data). Dropping message.")


def encode_and_send_frame(frame, send_frame_num):
    """
    Encode a video frame and send it to Kafka.

    Args:
        frame: The video frame to encode and send.
        send_frame_num: Boolean flag to indicate if the current frame should be sent.
    """
    if send_frame_num:
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])  # Lower quality to reduce size
        if ret:
            try:
                producer.produce(KAFKA_TOPIC_VIDEO, buffer.tobytes(), callback=delivery_report)
            except BufferError:
                print("Local producer queue is full (video frame). Dropping message.")


def main():
    """
    Main function to run the video capture, process frames, and send data to Kafka.
    """
    global cap, pose, mp_pose, mp_drawing, producer, KAFKA_TOPIC_VIDEO, KAFKA_TOPIC_POSE

    KAFKA_TOPIC_VIDEO = "video-stream-laptop-doni"
    KAFKA_TOPIC_POSE = "pose-data-laptop-doni"

    cap = initialize_video_capture()
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = initialize_mediapipe_pose()
    producer = initialize_kafka_producer()

    signal.signal(signal.SIGINT, signal_handler)

    frame_count = 0  # Track frame number for conditional processing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting...")
            break

        # Process every nth frame to reduce load (e.g., every 3rd frame)
        send_frame_num = frame_count % 3 == 0

        # Process and send data
        process_frame(frame, send_frame_num)
        encode_and_send_frame(frame, send_frame_num)

        producer.poll(0)  # Check for delivery reports

        # Increment frame count
        frame_count += 1

        # Small delay to avoid overwhelming the producer
        time.sleep(0.03)

    cap.release()
    pose.close()
    producer.flush()


if __name__ == "__main__":
    main()
