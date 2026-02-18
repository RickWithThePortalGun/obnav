#!/usr/bin/env python3
"""
Setup script for remote camera functionality.
This script helps configure and test the remote camera setup.
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['cv2', 'zmq', 'msgpack', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'zmq':
                import zmq
            elif package == 'msgpack':
                import msgpack
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install them using: pip install opencv-python pyzmq msgpack-python numpy")
        return False
    return True

def test_camera():
    """Test if camera is accessible."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera is accessible")
                return True
            else:
                print("✗ Camera is accessible but cannot read frames")
                return False
        else:
            print("✗ Cannot open camera")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def get_server_ip():
    """Get the server IP address."""
    import socket
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    print("=== Remote Camera Setup ===")
    print("This script helps you set up the remote camera functionality.")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies are available")
    
    # Test camera
    print("\nTesting camera...")
    if not test_camera():
        print("Camera test failed. Please check your camera connection.")
        sys.exit(1)
    
    # Get server IP
    server_ip = get_server_ip()
    print(f"\nServer IP: {server_ip}")
    
    # Configuration
    camera_port = input("Enter camera streaming port (default 5556): ").strip() or "5556"
    detection_port = input("Enter detection server port (default 5555): ").strip() or "5555"
    
    print(f"\n=== Configuration Summary ===")
    print(f"Server IP: {server_ip}")
    print(f"Camera Port: {camera_port}")
    print(f"Detection Port: {detection_port}")
    print()
    
    print("=== Usage Instructions ===")
    print("1. On the server (this machine), run:")
    print(f"   python camera_server.py")
    print()
    print("2. On the server, also run the detection server:")
    print(f"   python server.py")
    print()
    print("3. On the Raspberry Pi, set environment variables:")
    print(f"   export USE_REMOTE_CAMERA=1")
    print(f"   export HOST_IP={server_ip}")
    print(f"   export SERVER_PORT={detection_port}")
    print("   python run_prod_visual_enhanced.py")
    print()
    print("4. Or run without environment variables to get prompted for settings")
    print()
    print("=== Alternative: Direct Mode ===")
    print("You can also run the enhanced script directly and choose remote camera mode:")
    print("   python run_prod_visual_enhanced.py")
    print("   (Then select option 2 for remote camera)")

if __name__ == "__main__":
    main()
