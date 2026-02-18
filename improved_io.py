import time
import pigpio

class UltrasonicSensor:
    def __init__(self, pi, trig, echo, timeout_s=0.05, calibration_cm=0.0):
        self.pi = pi
        self.trig = trig
        self.echo = echo
        self.timeout_s = timeout_s
        self.calibration_cm = calibration_cm
        self.pi.set_mode(self.trig, pigpio.OUTPUT)
        self.pi.set_mode(self.echo, pigpio.INPUT)
        self.pi.write(self.trig, 0)
        time.sleep(0.05)

    def measure_distance(self):
        """Return distance in cm, or None if timeout."""
        self.pi.gpio_trigger(self.trig, 10, 1)  # 10µs pulse
        start_time = time.time()
        timeout = start_time + self.timeout_s

        while self.pi.read(self.echo) == 0:
            if time.time() > timeout:
                return None
        echo_on = time.time()

        while self.pi.read(self.echo) == 1:
            if time.time() > timeout:
                return None
        echo_off = time.time()
        pulse_duration = echo_off - echo_on
        distance_cm = (pulse_duration * 17150.0) + self.calibration_cm
        return round(distance_cm, 2)


class HapticMotor:
    def __init__(self, pi, pin, frequency=200):
        self.pi = pi
        self.pin = pin
        self.frequency = frequency
        self.pi.set_mode(self.pin, pigpio.OUTPUT)
        self.stop()

    def set_intensity(self, percent):
        """Control vibration strength with software PWM."""
        if percent <= 0:
            self.stop()
            return
        if percent > 100:
            percent = 100
        duty = int(percent * 10000)  # 0–1e6 range
        try:
            self.pi.hardware_PWM(self.pin, self.frequency, duty)
        except:
            # Fallback to software PWM if hardware PWM not available
            self.pi.set_PWM_dutycycle(self.pin, int(percent * 255 / 100))

    def pulse(self, on_ms=200, off_ms=200, repeats=3, intensity=80):
        """Pulse the motor on/off a few times."""
        for _ in range(repeats):
            self.set_intensity(intensity)
            time.sleep(on_ms / 1000.0)
            self.stop()
            time.sleep(off_ms / 1000.0)

    def stop(self):
        """Turn off PWM output."""
        try:
            self.pi.hardware_PWM(self.pin, 0, 0)
        except Exception:
            try:
                self.pi.write(self.pin, 0)
            except Exception:
                pass