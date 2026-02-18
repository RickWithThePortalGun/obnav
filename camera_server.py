#!/usr/bin/env python3
"""
Camera server that streams video from the server's camera to remote clients.
This allows the Raspberry Pi to use the server's camera instead of its own.
"""

import os
import cv2
import zmq
import msgpack
import numpy as np
import time

def main():
    """Main camera server function."""
    # Configuration
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))  # Default to first camera
    port = int(os.getenv("CAMERA_PORT", "5556"))
    fps = int(os.getenv("CAMERA_FPS", "30"))
    width = int(os.getenv("CAMERA_WIDTH", "640"))
    height = int(os.getenv("CAMERA_HEIGHT", "480"))
    
    print(f"Starting camera server on port {port}")
    print(f"Camera index: {camera_index}, Resolution: {width}x{height}, FPS: {fps}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Initialize ZMQ publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    print(f"Camera server ready. Clients can connect to: tcp://<server_ip>:{port}")
    print("Press 'q' to quit")
    
    frame_time = 1.0 / fps
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, jpg_data = cv2.imencode('.jpg', frame, encode_param)
            
            # Send frame data
            frame_data = {
                'frame': jpg_data.tobytes(),
                'timestamp': time.time(),
                'width': width,
                'height': height
            }
            
            socket.send(msgpack.packb(frame_data, use_bin_type=True))
            
            # Show preview (optional)
            cv2.imshow("Camera Server Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(frame_time)
            
    except KeyboardInterrupt:
        print("\nShutting down camera server...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        socket.close()
        context.term()
        print("Camera server stopped")

if __name__ == "__main__":
    main()
