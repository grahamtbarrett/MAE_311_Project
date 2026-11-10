#Logs serial data from the Arduino (MPU6050 CSV stream) to a CSV file on disk.


import serial
import time
import sys

# ---- EDIT THESE THREE LINES FOR YOUR SETUP ----
PORT = "COM4"          # Windows looks like "COM5";
BAUD_RATE = 500000      # must match Serial.begin(...) in the Arduino sketch (500000 for MPU6050_fast_logger.ino)
OUTPUT_FILE = "B_low_batt_t6.csv"   # change this per recording
# -------------------------------------------------

def main():
    try:
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