import os
import time
import cv2
import pigpio
import zmq
import msgpack
import numpy as np
from core import step_once
from adapters_prod_remote import RemoteDetector, LocalDetector
from enhancements import suggest_path
from eval_logger import EvalLogger
from picamera2 import Picamera2
from improved_io import UltrasonicSensor, HapticMotor

class PiCamera:
    def __init__(self, size=(640, 480)):
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration({"format": "RGB888", "size": size})
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.5)

    def read(self):
        frame = self.picam2.capture_array()   # RGB
        if frame is None or frame.size == 0:
            raise RuntimeError("camera read failed")
        return frame[:, :, ::-1]

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass

class RemoteCamera:
    """Remote camera that captures video from a server's camera via ZMQ."""
    
    def __init__(self, server_ip, port=5556, size=(640, 480)):
        self.server_ip = server_ip
        self.port = port
        self.size = size
        self.context = zmq.Context()
        self.socket = None
        self._connect()
    
    def _connect(self):
        """Connect to the remote camera server."""
        try:
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
            print(f"Connected to remote camera at {self.server_ip}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to remote camera: {e}")
            raise
    
    def read(self):
        """Read a frame from the remote camera."""
        try:
            if self.socket is None:
                self._connect()
            
            if self.socket.poll(1000, zmq.POLLIN):  # 1 second timeout
                data = self.socket.recv(zmq.NOBLOCK)
                frame_data = msgpack.unpackb(data, raw=False)
                
                frame_bytes = frame_data['frame']
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame = cv2.resize(frame, self.size)
                    return frame
                else:
                    raise RuntimeError("Failed to decode remote frame")
            else:
                raise RuntimeError("No frame received from remote camera")
                
        except Exception as e:
            print(f"Remote camera read error: {e}")
            # Return a black frame as fallback
            return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
    
    def close(self):
        """Close the remote camera connection."""
        try:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
        except Exception:
            pass

class GPIOHaptics:
    """
    Exposes buzz(sector: str, ms: int) to match existing code.
    Internally uses HapticMotor instances per pin.
    If you have only one motor, set SECTOR_MAP to point all sectors to same pin.
    """
    SECTOR_MAP = {
        "left": 22,
        "front": 19,
        "right": 27  # Changed from 3 to 18 (supports PWM)
    }

    def __init__(self, pi):
        self.pi = pi
        self.motors = {}
        for sec, pin in self.SECTOR_MAP.items():
            try:
                self.motors[sec] = HapticMotor(self.pi, pin=pin, frequency=200)
            except Exception:
                self.motors[sec] = HapticMotor(self.pi, pin=pin, frequency=200)

    def buzz(self, channel: str, ms: int = 100):
        """
        channel: sector name (e.g. "left", "front", "right")
        ms: pulse duration in milliseconds
        """
        if channel == "front":
            # For front sector, vibrate both motors with high interval, low pulse
            left_motor = self.motors.get("left")
            right_motor = self.motors.get("right")
            if left_motor and right_motor:
                # High interval (longer off time), low pulse (shorter on time)
                pulse_ms = max(20, ms // 3)  # Shorter pulse
                interval_ms = max(100, ms * 2)  # Longer interval
                intensity = 40  # Lower intensity for subtle effect
                
                # Pulse both motors simultaneously
                left_motor.pulse(on_ms=pulse_ms, off_ms=interval_ms, repeats=1, intensity=intensity)
                right_motor.pulse(on_ms=pulse_ms, off_ms=interval_ms, repeats=1, intensity=intensity)
        else:
            # For left/right sectors, use single motor
            if channel not in self.motors:
                motor = self.motors.get("front")
            else:
                motor = self.motors[channel]
            if motor is None:
                return
            intensity = 80 if ms >= 150 else 60
            motor.pulse(on_ms=ms, off_ms=50, repeats=1, intensity=intensity)

    def stop_all(self):
        for m in self.motors.values():
            try:
                m.stop()
            except Exception:
                pass

class HCSR04:
    def __init__(self, pi, trig=17, echo=27, timeout_s=0.05, calibration_cm=0.0):
        self.sensor = UltrasonicSensor(pi, trig=trig, echo=echo, timeout_s=timeout_s, calibration_cm=calibration_cm)

    def distance_cm(self):
        d = self.sensor.measure_distance()
        return 120.0 if d is None else d

def get_remote_server_ip():
    """Prompt user for remote server IP address."""
    while True:
        ip = input("Enter the IP address of the remote server: ").strip()
        if ip:
            # If user enters IP:port format, extract just the IP
            if ':' in ip:
                ip = ip.split(':')[0]
            return ip
        print("Please enter a valid IP address.")

def prompt_camera_mode():
    """Prompt user to choose between local and remote camera."""
    print("\nCamera Mode Selection:")
    print("1. Local camera (Raspberry Pi camera)")
    print("2. Remote camera (Server camera)")
    
    while True:
        choice = input("Choose camera mode (1 or 2): ").strip()
        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            print("Please enter 1 or 2")

def main():
    pi = pigpio.pi()
    if not pi.connected:
        exit("Start pigpiod first: sudo systemctl start pigpiod")
    
    use_remote_camera = os.getenv("USE_REMOTE_CAMERA", "0") == "1"
    if not use_remote_camera and os.getenv("USE_REMOTE_CAMERA") is None:
        use_remote_camera = prompt_camera_mode()
    
    remote_server_ip = None
    cam = None
    
    try:
        if use_remote_camera:
            remote_server_ip = get_remote_server_ip()
            print(f"Initializing remote camera from {remote_server_ip}...")
            cam = RemoteCamera(remote_server_ip)
            print("Remote camera initialized successfully")
        else:
            print("Initializing local Pi camera...")
            cam = PiCamera()
            print("Local camera initialized successfully")
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        if use_remote_camera:
            print("Falling back to local camera...")
            try:
                cam = PiCamera()
                use_remote_camera = False
                print("Local camera fallback successful")
            except Exception as e2:
                print(f"Local camera fallback also failed: {e2}")
                exit("No working camera available")
        else:
            exit("Local camera failed and no fallback available")
    
    use_local = os.getenv("USE_LOCAL", "0") == "1"
    
    try:
        if use_remote_camera and remote_server_ip:
            print(f"Initializing remote detector for {remote_server_ip}...")
            det = RemoteDetector(server=remote_server_ip)
        else:
            print("Initializing remote detector...")
            det = RemoteDetector()
        
        print("Testing connection...")
        if not det.test_connection():
            raise ValueError("Remote test failed")
        print("Using remote detector")
    except Exception as e:
        print(f"Remote detector failed: {e}. Falling back to local.")
        use_local = True

    if use_local:
        print("Using local detector")
        det = LocalDetector()
    rng = HCSR04(pi, trig=int(os.getenv("TRIG_PIN", "23")), echo=int(os.getenv("ECHO_PIN", "24")))
    hap = GPIOHaptics(pi)
    
    # Determine detector type for logging
    detector_type = "local" if use_local else "remote"
    logger = EvalLogger(path=os.getenv("EVAL_LOG", "eval_prod.csv"), detector_type=detector_type)

    camera_mode = "Remote" if use_remote_camera else "Local"
    detector_mode = "Local" if use_local else "Remote"
    print(f"\n=== Configuration ===")
    print(f"Camera: {camera_mode} ({remote_server_ip if use_remote_camera else 'Pi Camera'})")
    print(f"Detector: {detector_mode}")
    print(f"Press 'q' or ESC to quit")
    print("=" * 20)

    try:
        while True:
            logger.tick_start()
            frame = cam.read()
            detections = det.infer(frame)
            us_cm = rng.distance_cm()
            info = step_once(frame, detections, ultrasonic_cm=us_cm, ts=time.time())
            plan = suggest_path(info)
            hap.buzz(plan["target_sector"], plan["buzz_ms"])
            
            header = (
                f"path:{plan['path']}  "
                f"L~{int(plan['sector_dist']['left'])}  "
                f"F~{int(plan['sector_dist']['front'])}  "
                f"R~{int(plan['sector_dist']['right'])}  "
                f"US:{int(us_cm)}cm"
            )
            mode_info = f"Cam:{camera_mode} Det:{detector_mode}"
            
            cv2.rectangle(frame, (6, 6), (6 + 420, 50), (0, 0, 0), -1)
            cv2.putText(frame, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, mode_info, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.imshow("Obstacle Detection + Path", frame)
            logger.tick_end(info, action=f"{plan['target_sector']}:{plan['buzz_ms']}ms", path=plan["path"])
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break
            # Remove artificial FPS limit - let it run as fast as possible

    except KeyboardInterrupt:
        pass

    finally:
        logger.close()
        if hasattr(cam, "close"):
            cam.close()
        hap.stop_all()
        pi.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()