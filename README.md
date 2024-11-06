# README

## Overview

This project captures video from a webcam, uses MediaPipe to detect human poses, and streams both the video frames and pose data to Kafka topics. The purpose of this code is to provide real-time analysis of human poses while distributing the data for further processing or storage through a Kafka server.

## How the Code Works

1. **Environment Setup**: The `environment_loader.py` script loads the necessary environment variables from a `.env` file. These variables include Kafka server configurations such as IP, port, username, and password.

2. **Kafka Producer Initialization**: The Kafka producer is configured using the details loaded from the environment. If a username and password are provided, the producer is set up to authenticate with the Kafka server.

3. **Webcam Capture**: The code uses OpenCV to capture video frames from the default webcam (index 0).

4. **Pose Detection**: MediaPipe is used to process the video frames and detect pose landmarks. If pose landmarks are detected, they are drawn onto the frame for visualization.

5. **Data Streaming to Kafka**:

   - **Pose Data**: The detected pose landmarks are serialized into JSON and sent to the Kafka topic `pose-data-laptop`.
   - **Video Frames**: The video frame with the drawn landmarks is encoded as a JPEG image and sent to the Kafka topic `video-stream-laptop`.

6. **Graceful Shutdown**: The code handles signals (e.g., Ctrl+C) to ensure that resources like the webcam and Kafka producer are properly released before exiting.

## Requirements

To run the project, you will need the following Python packages, which are listed in `requirements.txt`:

- `opencv-python`
- `mediapipe`
- `confluent-kafka`
- `dotenv`

## How to Launch the Code

1. **Clone the Repository**: First, clone this repository to your local machine.

2. **Install Dependencies**:
   Run the following command to install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root directory with the following content:

   ```env
   KAFKA_IP=<Your Kafka Server IP>
   KAFKA_PORT=<Your Kafka Server Port>
   KAFKA_USERNAME=<Your Kafka Username>  # Optional
   KAFKA_PASSWORD=<Your Kafka Password>  # Optional
   ```

   Replace `<Your Kafka Server IP>` and `<Your Kafka Server Port>` with the actual values for your Kafka server. If authentication is required, add the username and password.

4. **Run the Code**:
   Run the Python script using the following command:
   ```bash
   python main.py
   ```
   This will start the video capture, process the frames for pose detection, and stream the data to Kafka.

## Important Notes

- Ensure that the Kafka server is accessible and configured correctly.
- If you encounter issues related to Kafka connectivity, double-check the environment variables and ensure that the Kafka server is up and running.
- To gracefully stop the program, press `Ctrl+C`. This will ensure that all resources are properly cleaned up.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
