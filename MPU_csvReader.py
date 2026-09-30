#Connect to serial port and record accelerometer data and build a csv until told to stop
import serial
import time
import sys

PORT = "COM4"               #typically like COM4 or 3
BAUD_RATE = 115200          # must match Serial.begin(...) in the Arduino sketch
OUTPUT_FILE = "ztest7.csv"  # change this per recording, e.g. "fault_mild_run1.csv"
# -------------------------------------------------

def main():
    try:
        #get access to the serial port
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {PORT}: {e}")
        print("Check the port name in the Arduino IDE under Tools > Port, and make sure")
        print("the Serial Monitor in the Arduino IDE is closed (only one program can use the port at a time).")
        sys.exit(1)

    time.sleep(2)  # give the Arduino a moment to reset after the serial connection opens

    print(f"Logging from {PORT} to {OUTPUT_FILE}. Press Ctrl+C to stop.")
    line_count = 0

    with open(OUTPUT_FILE, "w", newline="") as f:
        try:
            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if raw_line:
                    f.write(raw_line + "\n")
                    f.flush()  # write to disk immediately so nothing is lost if you Ctrl+C
                    line_count += 1
                    if line_count % 50 == 0:
                        print(f"  {line_count} lines logged...")
        except KeyboardInterrupt:
            print(f"\nStopped. Logged {line_count} lines to {OUTPUT_FILE}.")
        finally:
            ser.close()

if __name__ == "__main__":
    main()