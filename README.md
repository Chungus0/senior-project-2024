# Backend

---

## Project Structure

- **`requirements.txt`**: Contains the list of Python dependencies required for the application.
- **`.env`**: A file for storing environment variables such as Kafka configurations.
- **`main.py`**: The entry point for the Flask application. It sets up routes and Kafka consumers to handle video streaming.
- **`environment.py`**: A utility module for loading and managing environment variables securely.

---

## Requirements

- Python 3.8+
- Kafka Broker
- Virtual environment (recommended)

---

## Setup Instructions

1. **Clone the Repository**  
   Clone this repository to your local machine:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies**  
   Create a virtual environment (optional) and install the required Python libraries:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Unix
   venv\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**  
   Create a `.env` file in the project root directory and add the following variables:

   ```env
   KAFKA_IP=<Kafka broker IP>
   KAFKA_PORT=<Kafka broker port>
   KAFKA_USERNAME=<Kafka username (optional)>
   KAFKA_PASSWORD=<Kafka password (optional)>
   ```

   Example:

   ```env
   KAFKA_IP=127.0.0.1
   KAFKA_PORT=9092
   KAFKA_USERNAME=admin
   KAFKA_PASSWORD=admin123
   ```

4. **Run the Application**  
   Start the Flask application:
   ```bash
   python main.py
   ```
   The server will be accessible at `http://localhost:5000`.

---

## Key Features

1. **Environment Variable Handling**:

   - Variables are securely loaded from the `.env` file using the `dotenv` package.
   - Ensures the application doesn’t start without the necessary Kafka configurations.

2. **Kafka Integration**:

   - Uses `confluent_kafka` for robust Kafka consumer management.
   - Supports secure communication with optional username/password authentication.

3. **Real-Time Video Streaming**:
   - Streams video frames from Kafka topics to the frontend using Flask routes.
   - Provides frames in `multipart/x-mixed-replace` format for real-time updates.

---

## Example Usage

- To stream video data from a Kafka topic:
  - Visit the endpoint:
    ```plaintext
    http://localhost:5000/video_stream/<topic_name>
    ```
    Replace `<topic_name>` with the name of your Kafka topic.

---

## Troubleshooting

- **Environment Variable Errors**:  
  If you encounter errors related to `KAFKA_IP` or `KAFKA_PORT`, ensure these variables are correctly set in the `.env` file.
- **Consumer Errors**:  
  If Kafka consumers fail to connect or produce errors, check the Kafka broker status and ensure the topic exists.

---

# Cloud Server

## Overview

This setup provides a Kafka environment using Docker Compose with **Zookeeper** and **Kafka** services. It is optimized for low-latency, high-performance data streaming.

---

## Project Structure

- **`docker-compose.yml`**: Defines the Zookeeper and Kafka services.

---

## Setup Instructions

1. **Launch the Services**

   ```bash
   docker-compose up -d
   ```

2. **Verify the Setup**
   ```bash
   docker ps
   ```
   Confirm Zookeeper is on port `2181` and Kafka on `9092` (internal) / `29092` (external).

---

## Configuration Highlights

- **Zookeeper**: Coordination service on port `2181`.
- **Kafka**:
  - Internal Listener: `localhost:9092`
  - External Listener: `192.168.1.88:29092`
  - Retention Policy: Data stored for 10 seconds (`KAFKA_LOG_RETENTION_MS`).

---

## Troubleshooting

- Check logs for issues:
  ```bash
  docker-compose logs
  ```
- Verify Kafka connectivity:
  ```bash
  kafka-topics.sh --bootstrap-server localhost:9092 --list
  ```

---

# Edge Device

---

## Project Structure

- **`.env`**: Contains environment variables for Kafka configurations.
- **`environment.py`**: Loads and manages environment variables.
- **`main.py`**: The main script for video capture, pose detection, and data streaming.
- **`requirements.txt`**: Lists the required Python packages.

---

## Requirements

- Python 3.8+
- A webcam
- Kafka Broker

---

## Setup Instructions

1. **Install Dependencies**  
   Create a virtual environment (optional) and install the required libraries:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Unix
   venv\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**  
   Create a `.env` file in the project root with the following variables:

   ```env
   KAFKA_IP=<Kafka broker IP>
   KAFKA_PORT=<Kafka broker port>
   KAFKA_USERNAME=<Kafka username (optional)>
   KAFKA_PASSWORD=<Kafka password (optional)>
   ```

   Example:

   ```env
   KAFKA_IP=127.0.0.1
   KAFKA_PORT=9092
   ```

3. **Run the Application**  
   Start the edge application:
   ```bash
   python main.py
   ```

---

## Key Features

1. **Pose Detection**:  
   Uses MediaPipe to detect and extract pose landmarks from webcam video frames.

2. **Kafka Integration**:  
   Streams video frames and pose data to two Kafka topics:

   - Video: `video-stream-laptop`
   - Pose Data: `pose-data-laptop`

3. **Efficient Processing**:  
   Processes every 3rd frame to reduce load and encodes frames with reduced quality for smaller message sizes.

---

## Troubleshooting

- **Environment Variables Missing**:  
  Ensure the `.env` file is correctly set up.
- **Kafka Connection Issues**:  
  Verify the Kafka broker is running and accessible at the specified IP and port.

- **Webcam Errors**:  
  Ensure the webcam is properly connected and accessible by OpenCV.

---
