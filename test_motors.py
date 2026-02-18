#!/usr/bin/env python3
"""
Test script for haptic motors.
This script helps verify that the motors are working correctly
and allows testing different vibration patterns.
"""

import os
import time
import pigpio
from improved_io import HapticMotor

class MotorTester:
    def __init__(self, left_pin=2, right_pin=3, frequency=200):
        """Initialize the motor tester."""
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio. Make sure pigpiod is running: sudo systemctl start pigpiod")
        
        self.left_pin = left_pin
        self.right_pin = right_pin
        self.frequency = frequency
        
        # Initialize motors
        try:
            self.left_motor = HapticMotor(self.pi, pin=left_pin, frequency=frequency)
            self.right_motor = HapticMotor(self.pi, pin=right_pin, frequency=frequency)
            print(f"Motors initialized successfully:")
            print(f"  Left motor:  GPIO {left_pin}")
            print(f"  Right motor: GPIO {right_pin}")
            print(f"  Frequency:   {frequency} Hz")
            print()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize motors: {e}")

    def test_single_motor(self, motor_name, motor, duration=2):
        """Test a single motor."""
        print(f"Testing {motor_name} motor for {duration} seconds...")
        try:
            # Test different intensities
            for intensity in [20, 40, 60, 80, 100]:
                print(f"  Intensity {intensity}%...", end="", flush=True)
                motor.set_intensity(intensity)
                time.sleep(0.5)
                motor.stop()
                time.sleep(0.2)
            print(" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")

    def test_pulse_patterns(self, motor_name, motor):
        """Test different pulse patterns."""
        print(f"Testing {motor_name} motor pulse patterns...")
        
        patterns = [
            {"name": "Short pulse", "on_ms": 50, "off_ms": 100, "repeats": 3, "intensity": 60},
            {"name": "Medium pulse", "on_ms": 100, "off_ms": 100, "repeats": 3, "intensity": 60},
            {"name": "Long pulse", "on_ms": 200, "off_ms": 100, "repeats": 3, "intensity": 60},
            {"name": "Rapid fire", "on_ms": 30, "off_ms": 30, "repeats": 5, "intensity": 80},
            {"name": "Slow pulse", "on_ms": 100, "off_ms": 200, "repeats": 3, "intensity": 40},
        ]
        
        for pattern in patterns:
            try:
                print(f"  {pattern['name']}...", end="", flush=True)
                motor.pulse(
                    on_ms=pattern["on_ms"],
                    off_ms=pattern["off_ms"],
                    repeats=pattern["repeats"],
                    intensity=pattern["intensity"]
                )
                time.sleep(0.5)  # Wait for pattern to complete
                print(" ✓")
            except Exception as e:
                print(f" ✗ Error: {e}")

    def test_both_motors(self):
        """Test both motors simultaneously."""
        print("Testing both motors simultaneously...")
        
        patterns = [
            {"name": "Synchronized", "delay": 0},
            {"name": "Alternating", "delay": 0.1},
            {"name": "Left first", "delay": 0.2},
            {"name": "Right first", "delay": -0.2},
        ]
        
        for pattern in patterns:
            try:
                print(f"  {pattern['name']}...", end="", flush=True)
                
                # Start left motor
                if pattern["delay"] >= 0:
                    self.left_motor.pulse(on_ms=100, off_ms=100, repeats=2, intensity=60)
                    time.sleep(pattern["delay"])
                    self.right_motor.pulse(on_ms=100, off_ms=100, repeats=2, intensity=60)
                else:
                    self.right_motor.pulse(on_ms=100, off_ms=100, repeats=2, intensity=60)
                    time.sleep(abs(pattern["delay"]))
                    self.left_motor.pulse(on_ms=100, off_ms=100, repeats=2, intensity=60)
                
                time.sleep(1)  # Wait for patterns to complete
                print(" ✓")
            except Exception as e:
                print(f" ✗ Error: {e}")

    def test_sector_patterns(self):
        """Test sector-specific patterns (left, right, front)."""
        print("Testing sector-specific patterns...")
        
        # Left sector pattern
        print("  Left sector pattern...", end="", flush=True)
        try:
            self.left_motor.pulse(on_ms=80, off_ms=50, repeats=1, intensity=70)
            time.sleep(0.5)
            print(" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")
        
        # Right sector pattern
        print("  Right sector pattern...", end="", flush=True)
        try:
            self.right_motor.pulse(on_ms=80, off_ms=50, repeats=1, intensity=70)
            time.sleep(0.5)
            print(" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")
        
        # Front sector pattern (both motors)
        print("  Front sector pattern (both motors)...", end="", flush=True)
        try:
            # High interval, low pulse pattern
            pulse_ms = 20
            interval_ms = 100
            intensity = 40
            
            self.left_motor.pulse(on_ms=pulse_ms, off_ms=interval_ms, repeats=1, intensity=intensity)
            self.right_motor.pulse(on_ms=pulse_ms, off_ms=interval_ms, repeats=1, intensity=intensity)
            time.sleep(0.5)
            print(" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")

    def test_intensity_sweep(self):
        """Test intensity sweep from 0 to 100%."""
        print("Testing intensity sweep...")
        print("  Sweeping intensity from 0% to 100%...")
        
        try:
            for intensity in range(0, 101, 10):
                print(f"    {intensity:3d}%", end="", flush=True)
                self.left_motor.set_intensity(intensity)
                time.sleep(0.1)
            print()
            self.left_motor.stop()
            print("  ✓ Intensity sweep completed")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    def test_frequency_variation(self):
        """Test different frequencies."""
        print("Testing frequency variation...")
        
        frequencies = [100, 150, 200, 250, 300]
        
        for freq in frequencies:
            try:
                print(f"  Testing frequency {freq} Hz...", end="", flush=True)
                
                # Create temporary motor with different frequency
                temp_motor = HapticMotor(self.pi, pin=self.left_pin, frequency=freq)
                temp_motor.pulse(on_ms=100, off_ms=100, repeats=2, intensity=60)
                time.sleep(0.5)
                temp_motor.stop()
                
                print(" ✓")
            except Exception as e:
                print(f" ✗ Error: {e}")

    def continuous_test(self, duration=30, interval=2):
        """Run continuous test for specified duration."""
        print(f"Running continuous test for {duration} seconds...")
        print("Alternating between left and right motors every 2 seconds")
        print("Press Ctrl+C to stop early")
        print("-" * 50)
        
        start_time = time.time()
        motor_cycle = [("Left", self.left_motor), ("Right", self.right_motor)]
        cycle_index = 0
        
        try:
            while time.time() - start_time < duration:
                motor_name, motor = motor_cycle[cycle_index % 2]
                elapsed = time.time() - start_time
                
                print(f"[{elapsed:6.1f}s] Testing {motor_name} motor...", end="", flush=True)
                
                try:
                    motor.pulse(on_ms=100, off_ms=100, repeats=1, intensity=50)
                    time.sleep(interval)
                    print(" ✓")
                except Exception as e:
                    print(f" ✗ Error: {e}")
                
                cycle_index += 1
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        # Stop all motors
        self.left_motor.stop()
        self.right_motor.stop()

    def close(self):
        """Clean up resources."""
        try:
            self.left_motor.stop()
            self.right_motor.stop()
            self.pi.stop()
        except:
            pass

def main():
    """Main test function."""
    print("=" * 60)
    print("HAPTIC MOTOR TESTER")
    print("=" * 60)
    
    # Get configuration from environment or use defaults
    left_pin = int(os.getenv("LEFT_MOTOR_PIN", "2"))
    right_pin = int(os.getenv("RIGHT_MOTOR_PIN", "3"))
    frequency = int(os.getenv("MOTOR_FREQUENCY", "200"))
    
    try:
        tester = MotorTester(left_pin, right_pin, frequency)
        
        while True:
            print("\n" + "=" * 40)
            print("MOTOR TEST MENU:")
            print("1. Test left motor only")
            print("2. Test right motor only")
            print("3. Test both motors")
            print("4. Test pulse patterns (left)")
            print("5. Test pulse patterns (right)")
            print("6. Test sector patterns")
            print("7. Test intensity sweep")
            print("8. Test frequency variation")
            print("9. Continuous test (30s)")
            print("10. Custom continuous test")
            print("11. Exit")
            print("=" * 40)
            
            choice = input("Select test (1-11): ").strip()
            
            if choice == "1":
                tester.test_single_motor("Left", tester.left_motor)
                
            elif choice == "2":
                tester.test_single_motor("Right", tester.right_motor)
                
            elif choice == "3":
                tester.test_both_motors()
                
            elif choice == "4":
                tester.test_pulse_patterns("Left", tester.left_motor)
                
            elif choice == "5":
                tester.test_pulse_patterns("Right", tester.right_motor)
                
            elif choice == "6":
                tester.test_sector_patterns()
                
            elif choice == "7":
                tester.test_intensity_sweep()
                
            elif choice == "8":
                tester.test_frequency_variation()
                
            elif choice == "9":
                tester.continuous_test(30, 2)
                
            elif choice == "10":
                try:
                    duration = float(input("Duration in seconds: "))
                    interval = float(input("Interval between tests: "))
                    tester.continuous_test(duration, interval)
                except ValueError:
                    print("Invalid input. Using defaults.")
                    tester.continuous_test(10, 1)
                    
            elif choice == "11":
                break
                
            else:
                print("Invalid choice. Please select 1-11.")
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            tester.close()
        except:
            pass
        print("\nTest completed.")

if __name__ == "__main__":
    main()
