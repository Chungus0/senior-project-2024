import cv2
import mediapipe as mp
from confluent_kafka import Producer
import json
import signal
import sys
from environment import load_environment_variables

def initialize_kafka_producer():
    """
    Initialize the Kafka producer with configuration.

    Returns:
        producer: The Kafka Producer object.
    """
    environment=load_environment_variables()

    KAFKA_SERVER = f"{environment['KAFKA_IP']}:{environment['KAFKA_PORT']}"
    KAFKA_USERNAME = environment.get("KAFKA_USERNAME")
    KAFKA_PASSWORD = environment.get("KAFKA_PASSWORD")

    producer_config = {
        "bootstrap.servers": KAFKA_SERVER,
        "linger.ms": 5,
        "acks": 1,
    }

    if KAFKA_USERNAME and KAFKA_PASSWORD:
        producer_config.update({
            "security.protocol": "SASL_PLAINTEXT",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": KAFKA_USERNAME,
            "sasl.password": KAFKA_PASSWORD,
        })

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

def process_frame(frame):
    """
    Process a video frame to detect pose landmarks and send the data to Kafka.

    Args:
        frame: The video frame to process.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        pose_data = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            }
            for lm in results.pose_landmarks.landmark
        ]

        pose_json = json.dumps(pose_data)

        try:
            producer.produce(KAFKA_TOPIC_POSE, pose_json.encode("utf-8"), callback=delivery_report)
        except BufferError:
            print("Local producer queue is full (pose data). Dropping message.")

def encode_and_send_frame(frame):
    """
    Encode a video frame and send it to Kafka.

    Args:
        frame: The video frame to encode and send.
    """
    ret, buffer = cv2.imencode(".jpg", frame)
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

    KAFKA_TOPIC_VIDEO = "video-stream-laptop-2"
    KAFKA_TOPIC_POSE = "pose-data-laptop"

    cap = initialize_video_capture()
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = initialize_mediapipe_pose()
    producer = initialize_kafka_producer()

    signal.signal(signal.SIGINT, signal_handler)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting...")
            break

        process_frame(frame)
        encode_and_send_frame(frame)

        producer.poll(0)

    cap.release()
    pose.close()
    producer.flush()

if __name__ == "__main__":
    main()
