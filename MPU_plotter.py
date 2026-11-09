#Loads CSV data from MPU_csvReader.py and creates raw time-domain acceleration plots
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set this to the CSV file you want to plot. If you'd rather pass it on the
# command line instead (MPU_plotter.py yourfile.csv), leave
# this as None and it will use the command-line argument instead.
FILEPATH = "battlow_t2.csv"

# Counts per g for the accelerometer range set in the sketch (+-4 g -> 8192).
# Only used for CSVs with ax_raw/ay_raw/az_raw columns.
ACCEL_COUNTS_PER_G = 8192.0

# Static-calibration results from your six-position test, per axis.
# Defaults (offset 0, scale 1) leave the data unchanged. Fill these in once you
# have your numbers. corrected = (raw - offset) / scale
# NOTE: this changes amplitudes only. It cannot move an FFT peak in frequency.
CALIBRATION = {
    "ax": {"offset": 0.0, "scale": 1.0},
    "ay": {"offset": 0.0, "scale": 1.0},
    "az": {"offset": 0.0, "scale": 1.0},
}

def load_csv(filepath):
    # skiprows=lambda handles the case where a stray non-CSV line (like a boot message)
    # sneaks into the file; on_bad_lines='skip' drops any malformed rows instead of crashing
    df = pd.read_csv(filepath, on_bad_lines='skip')
    df = df.dropna()                                    #deletes any rows that have a None or missing data
    return df

def compute_fft(signal, sample_rate):
    n = len(signal)
    # remove DC offset (mean) so the 0 Hz bin doesn't dominate the plot
    signal = signal - np.mean(signal)                   #Removing the 0Hz offset
    fft_vals = np.fft.rfft(signal)                      #1D DFT for real-valued input signals
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate) #Array of freq from FFT bins
    fft_mag = np.abs(fft_vals) / n                      #normalizing FFT vals
    return fft_freqs, fft_mag

def main():
    filepath = FILEPATH if FILEPATH else (sys.argv[1] if len(sys.argv) > 1 else None)
    #use terminal if filepath not specified
    if not filepath:
        print("Usage: python plot_vibration_data.py <path_to_csv>")
        print("(or set FILEPATH near the top of this script)")
        sys.exit(1)
    #check if path exist to file
    if not os.path.exists(filepath):
        print(f"File not found: {os.path.abspath(filepath)}")
        print("Check that the CSV is in the same folder as this script, or use a full path.")
        sys.exit(1)

    print(f"Loading {os.path.abspath(filepath)} ...")
    df = load_csv(filepath)

    # The fast logger sketch (MPU6050_fast_logger.ino) prints raw sensor counts in
    # ax_raw, ay_raw, az_raw. Convert to m/s^2 here (+-4 g range = 8192 counts/g).
    if {"ax_raw", "ay_raw", "az_raw"}.issubset(df.columns):
        for a in ("ax", "ay", "az"):
            df[a] = df[a + "_raw"] * 9.80665 / ACCEL_COUNTS_PER_G

    # The static-test sketch prints "timestamp_ms" (milliseconds); the dynamic
    # calibration sketch prints "timestamp_us" (microseconds, from micros()).
    # Support either one so this script works with both CSV types.
    if "timestamp_us" in df.columns:
        time_col = "timestamp_us"
        time_divisor = 1_000_000.0  # microseconds -> seconds
    elif "timestamp_ms" in df.columns:
        time_col = "timestamp_ms"
        time_divisor = 1_000.0  # milliseconds -> seconds
    else:
        print(f"Expected a timestamp_us or timestamp_ms column plus ax, ay, az. Found: {list(df.columns)}")
        sys.exit(1)

    if not {"ax", "ay", "az"}.issubset(df.columns):
        print(f"Expected columns ax, ay, az. Found: {list(df.columns)}")
        sys.exit(1)

    # Estimate the actual sample rate from the timestamps (more accurate than assuming)
    dt = df[time_col].diff().dropna()          #time diff between samples, in raw timestamp units
    avg_dt_sec = dt.mean() / time_divisor #convert average diff to seconds
    print(avg_dt_sec)
    sample_rate = 1.0 / avg_dt_sec
    print(f"Loaded {len(df)} samples. Estimated sample rate: {sample_rate:.1f} Hz")

    # Timing check: the FFT assumes evenly spaced samples. Large spread in the
    # time steps means the Arduino loop is falling behind and the frequency axis
    # cannot be trusted.
    dt_sec = dt / time_divisor
    jitter_pct = 100.0 * dt_sec.std() / dt_sec.mean()
    print(f"  Time-step spread: {jitter_pct:.1f}% of the mean (min {dt_sec.min()*1e3:.2f} ms, max {dt_sec.max()*1e3:.2f} ms)")
    if jitter_pct > 5:
        print("  WARNING: sampling is irregular. Lower TARGET_SAMPLE_RATE_HZ in the sketch or raise the baud rate.")

    # Apply static calibration (no effect until you fill in CALIBRATION above)
    for axis_name, cal in CALIBRATION.items():
        df[axis_name] = (df[axis_name] - cal["offset"]) / cal["scale"]
    #fix the phase offset so that it starts at t=0s
    t = (df[time_col] - df[time_col].iloc[0]) / time_divisor  # seconds, starting at 0
    #plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f"Vibration Data: {filepath}  (~{sample_rate:.1f} Hz)")

    axis_names = ["ax", "ay", "az"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, (axis, color) in enumerate(zip(axis_names, colors)):
        #Top row: raw time-domain signal
        axes[0, i].plot(t, df[axis], color=color, linewidth=0.7)
        axes[0, i].set_title(f"{axis} - time domain")
        axes[0, i].set_xlabel("Time (s)")
        axes[0, i].set_ylabel("Acceleration (m/s^2)")

        #Bottom row: FFT
        freqs, mag = compute_fft(df[axis].values, sample_rate)
        axes[1, i].plot(freqs, mag, color=color, linewidth=0.8)
        axes[1, i].set_title(f"{axis} - FFT")
        axes[1, i].set_xlabel("Frequency (Hz)")
        axes[1, i].set_ylabel("Magnitude")
        axes[1, i].set_xlim(0, sample_rate / 2)  #Nyquist limit

        #Print the dominant (non-zero) peak frequency for a quick sanity check
        peak_idx = np.argmax(mag[1:]) + 1  #skip the 0 Hz bin
        print(f"  {axis}: dominant peak at {freqs[peak_idx]:.2f} Hz")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()