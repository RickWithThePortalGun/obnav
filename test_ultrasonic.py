#!/usr/bin/env python3
"""
Test script for ultrasonic sensors.
This script helps verify that the ultrasonic sensor is working correctly
and provides real-time distance readings.
"""

import os
import time
import pigpio
from improved_io import UltrasonicSensor

class UltrasonicTester:
    def __init__(self, trig_pin=23, echo_pin=24, timeout_s=0.05, calibration_cm=0.0):
        """Initialize the ultrasonic sensor tester."""
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio. Make sure pigpiod is running: sudo systemctl start pigpiod")
        
        self.sensor = UltrasonicSensor(
            self.pi, 
            trig=trig_pin, 
            echo=echo_pin, 
            timeout_s=timeout_s, 
            calibration_cm=calibration_cm
        )
        
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.timeout_s = timeout_s
        self.calibration_cm = calibration_cm
        
        print(f"Ultrasonic sensor initialized:")
        print(f"  TRIG pin: {trig_pin}")
        print(f"  ECHO pin: {echo_pin}")
        print(f"  Timeout: {timeout_s}s")
        print(f"  Calibration: {calibration_cm}cm")
        print()

    def single_reading(self):
        """Get a single distance reading."""
        try:
            distance = self.sensor.measure_distance()
            if distance is None:
                return None, "No echo received (timeout or no obstacle)"
            return distance, "Success"
        except Exception as e:
            return None, f"Error: {e}"

    def continuous_readings(self, duration=30, interval=0.5):
        """Take continuous readings for a specified duration."""
        print(f"Taking readings for {duration} seconds (every {interval}s)...")
        print("Press Ctrl+C to stop early")
        print("-" * 50)
        
        start_time = time.time()
        readings = []
        
        try:
            while time.time() - start_time < duration:
                distance, status = self.single_reading()
                timestamp = time.time() - start_time
                
                if distance is not None:
                    readings.append(distance)
                    print(f"[{timestamp:6.1f}s] Distance: {distance:6.2f} cm - {status}")
                else:
                    print(f"[{timestamp:6.1f}s] Distance:   None cm - {status}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        return readings

    def statistics_test(self, num_readings=20, interval=0.2):
        """Take multiple readings and show statistics."""
        print(f"Taking {num_readings} readings (every {interval}s)...")
        print("-" * 50)
        
        readings = []
        for i in range(num_readings):
            distance, status = self.single_reading()
            if distance is not None:
                readings.append(distance)
                print(f"Reading {i+1:2d}: {distance:6.2f} cm - {status}")
            else:
                print(f"Reading {i+1:2d}:   None cm - {status}")
            time.sleep(interval)
        
        if readings:
            import statistics
            print("\n" + "=" * 50)
            print("STATISTICS:")
            print(f"  Valid readings: {len(readings)}/{num_readings}")
            print(f"  Min distance:   {min(readings):.2f} cm")
            print(f"  Max distance:   {max(readings):.2f} cm")
            print(f"  Mean distance:  {statistics.mean(readings):.2f} cm")
            print(f"  Median:         {statistics.median(readings):.2f} cm")
            print(f"  Std deviation:  {statistics.stdev(readings):.2f} cm")
        else:
            print("\nNo valid readings obtained!")
        
        return readings

    def range_test(self, test_distances=[10, 20, 30, 50, 100, 150, 200]):
        """Test sensor at known distances."""
        print("RANGE TEST")
        print("Place objects at known distances and press Enter for each test.")
        print("Press Ctrl+C to skip remaining tests.")
        print("-" * 50)
        
        results = []
        
        for target_distance in test_distances:
            try:
                input(f"\nPlace object at {target_distance}cm and press Enter...")
                
                # Take 5 readings and average
                readings = []
                for _ in range(5):
                    distance, status = self.single_reading()
                    if distance is not None:
                        readings.append(distance)
                    time.sleep(0.1)
                
                if readings:
                    avg_distance = sum(readings) / len(readings)
                    error = avg_distance - target_distance
                    error_percent = (error / target_distance) * 100
                    
                    result = {
                        'target': target_distance,
                        'measured': avg_distance,
                        'error': error,
                        'error_percent': error_percent,
                        'readings': readings
                    }
                    results.append(result)
                    
                    print(f"  Target: {target_distance}cm")
                    print(f"  Measured: {avg_distance:.2f}cm")
                    print(f"  Error: {error:+.2f}cm ({error_percent:+.1f}%)")
                else:
                    print(f"  No valid readings at {target_distance}cm")
                    
            except KeyboardInterrupt:
                print("\nSkipping remaining tests...")
                break
        
        if results:
            print("\n" + "=" * 50)
            print("RANGE TEST SUMMARY:")
            for result in results:
                print(f"  {result['target']:3d}cm -> {result['measured']:6.2f}cm "
                      f"(error: {result['error']:+6.2f}cm, {result['error_percent']:+5.1f}%)")
        
        return results

    def close(self):
        """Clean up resources."""
        try:
            self.pi.stop()
        except:
            pass

def main():
    """Main test function."""
    print("=" * 60)
    print("ULTRASONIC SENSOR TESTER")
    print("=" * 60)
    
    # Get configuration from environment or use defaults
    trig_pin = int(os.getenv("TRIG_PIN", "23"))
    echo_pin = int(os.getenv("ECHO_PIN", "24"))
    timeout_s = float(os.getenv("TIMEOUT_S", "0.05"))
    calibration_cm = float(os.getenv("CALIBRATION_CM", "0.0"))
    
    try:
        tester = UltrasonicTester(trig_pin, echo_pin, timeout_s, calibration_cm)
        
        while True:
            print("\n" + "=" * 40)
            print("TEST MENU:")
            print("1. Single reading")
            print("2. Continuous readings (30s)")
            print("3. Statistics test (20 readings)")
            print("4. Range test (known distances)")
            print("5. Custom continuous test")
            print("6. Exit")
            print("=" * 40)
            
            choice = input("Select test (1-6): ").strip()
            
            if choice == "1":
                print("\nTaking single reading...")
                distance, status = tester.single_reading()
                if distance is not None:
                    print(f"Distance: {distance:.2f} cm - {status}")
                else:
                    print(f"Distance: None cm - {status}")
                    
            elif choice == "2":
                tester.continuous_readings(30, 0.5)
                
            elif choice == "3":
                tester.statistics_test(20, 0.2)
                
            elif choice == "4":
                tester.range_test()
                
            elif choice == "5":
                try:
                    duration = float(input("Duration in seconds: "))
                    interval = float(input("Interval between readings: "))
                    tester.continuous_readings(duration, interval)
                except ValueError:
                    print("Invalid input. Using defaults.")
                    tester.continuous_readings(10, 0.5)
                    
            elif choice == "6":
                break
                
            else:
                print("Invalid choice. Please select 1-6.")
                
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
