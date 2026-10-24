"""
imu_calibration.py - static calibration + Allan deviation for the MPU-6050

    m = S * g_body + B + noise        (stationary sensor, so a = 0)

    S = scale-factor / misalignment matrix (3x3)
    B = bias (offset) vector

and characterizes the time-varying terms (noise N_t, bias drift dB_t) with an
Allan deviation of a long stationary recording.

------------------------------------------------------------------------------
1) SIX-POSITION CALIBRATION
------------------------------------------------------------------------------
Put the CSVs from your logger in one folder (default: calibration_data/) and
name them  <axis><up|down>_t<trial>.csv  e.g.

    xup_t1.csv  xup_t2.csv ... xdown_t1.csv ... zdown_t10.csv

"xup" means the sensor's +X arrow points straight up (so it reads about +9.81
on x). Do about 10 independent trials per pose, resetting the board between
trials. Any file named  val_*.csv  (sensor held at some other tilt, such as
your 30/60 degree setup) is used as an independent validation check.

    python imu_calibration.py calibrate
    python imu_calibration.py calibrate --dir my_folder --out my_cal.json

------------------------------------------------------------------------------
2) ALLAN DEVIATION
------------------------------------------------------------------------------
Use one long recording (30-60 min) with the board flat and untouched:

    python imu_calibration.py allan --file long_stationary.csv

------------------------------------------------------------------------------
3) USING THE RESULT IN YOUR OTHER SCRIPTS
------------------------------------------------------------------------------
    from imu_calibration import load_calibration, correct
    S, B = load_calibration("imu_calibration.json")
    corrected_xyz = correct(raw_xyz, S, B)     # raw_xyz is an (N, 3) array
"""

import argparse                      #command line arguments (calibrate / allan)
import glob                          #finds all the csv files in a folder
import json                          #saves/loads the calibration results
import os                            #file paths
import re                            #filename pattern matching (xup_t1.csv)
import sys                           #sys.argv, used for the default command in main()
from datetime import date            #stamps the calibration file with today's date

import numpy as np                   #matrix math (S, B, inverse)
import pandas as pd                  #reads the csv files
import matplotlib.pyplot as plt      #Allan deviation plot

G = 9.80665                      # m/s^2 (standard gravity, the known reference input)
AXES = ["x", "y", "z"]           #axis letters used in the file names
AXIS_COLS = ["ax", "ay", "az"]   #column names inside the csv
#raw counts per g for the accelerometer range in the sketch (+-4 g -> 8192, +-2 g -> 16384, +-8 g -> 4096)
#only used for csvs from the fast logger sketch that have ax_raw, ay_raw, az_raw columns
ACCEL_COUNTS_PER_G = 8192.0
TRIM_FRACTION = 0.10             # drop this fraction at each end of every file (handling/settling)

# Edit these to match the sketch you used. They are saved with the calibration
# because S depends on the range and filter settings.
SETTINGS = {"accel_range_g": 4, "filter_bandwidth_hz": 184}
#file naming convention used in gather_pose
#(axis letter)(up or down)_t(trial number).csv   ex: zdown_t3.csv
POSE_RE = re.compile(r"^([xyz])(up|down)_t(\d+)\.csv$", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_csv(path):
    #Load a logger CSV. Accepts timestamp_us or timestamp_ms. Returns df with a 't' column in seconds.
    df = pd.read_csv(path, on_bad_lines="skip")                 #skip garbled lines from the serial port
    df = df.apply(pd.to_numeric, errors="coerce").dropna()      #text -> numbers, drop any row that did not parse
    #the fast logger sketch prints raw counts, so convert to m/s^2 here and everything below can work in m/s^2
    if {"ax_raw", "ay_raw", "az_raw"}.issubset(df.columns):
        for a in AXIS_COLS:
            df[a] = df[a + "_raw"] * G / ACCEL_COUNTS_PER_G     #counts -> g -> m/s^2
    if "timestamp_us" in df.columns:
        df["t"] = df["timestamp_us"] / 1e6                      #microseconds -> seconds
    elif "timestamp_ms" in df.columns:
        df["t"] = df["timestamp_ms"] / 1e3                      #milliseconds -> seconds
    else: #exception cases
        raise ValueError(f"{path}: needs a timestamp_us or timestamp_ms column, found {list(df.columns)}")
    if not set(AXIS_COLS).issubset(df.columns):
        raise ValueError(f"{path}: needs columns ax, ay, az, found {list(df.columns)}")
    return df.reset_index(drop=True)                            #clean 0..n-1 row numbers after dropping rows


def trial_mean(path):
    #Mean (ax, ay, az) of one file after trimming 10% of each end to prevent any motion from stopping or starting
    #the recording to interrupt the data.
    df = load_csv(path)
    n = len(df)
    k = int(n * TRIM_FRACTION)                                  #number of samples to cut from each end
    core = df.iloc[k:n - k] if n - 2 * k >= 10 else df          #only trim if at least 10 samples would be left
    return core[AXIS_COLS].to_numpy().mean(axis=0)              #one number per axis for this trial


# ----------------------------------------------------------------------------
# Six-position calibration
# ----------------------------------------------------------------------------

#once all the data is collected, this function gathers and sorts data to be used in the calibration function
def gather_poses(folder):
    #Returns {(axis_index, +1/-1): array of shape (n_trials, 3)} of per-trial mean readings.
    poses = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):       #checks for all csvs
        m = POSE_RE.match(os.path.basename(path))                       #if it does not match template skip
        if not m:
            continue
        axis = AXES.index(m.group(1).lower())                           #convert xyz to 012
        sign = +1 if m.group(2).lower() == "up" else -1                 #+1 = axis pointing up, -1 = pointing down
        poses.setdefault((axis, sign), []).append(trial_mean(path))     #add this trial's mean to that pose's list
    return {k: np.array(v) for k, v in poses.items()}                   #lists -> arrays, one row per trial


def solve_calibration(poses):
    """
    With the sensor still, axis i pointing up reads  S[:, i]*g + B ; pointing down reads  -S[:, i]*g + B.
    So  S[:, i] = (m_up - m_down) / (2 g)   and   B = (m_up + m_down) / 2   (averaged over the three axes).
    Uncertainties are standard errors of the trial means, propagated through those formulas.
    """
    #make sure all six poses exist and each has enough trials before solving
    missing = [f"{AXES[a]}{'up' if s > 0 else 'down'}" for a in range(3) for s in (+1, -1) if (a, s) not in poses]
    if missing:
        raise SystemExit(f"Missing pose files for: {', '.join(missing)}")
    for key, arr in poses.items():
        if len(arr) < 2:                                                #1 trial has no spread, so no uncertainty
            raise SystemExit("Need at least 2 trials per pose to estimate uncertainty.")

    #initialize S matrix
    S = np.zeros((3, 3))
    S_se = np.zeros((3, 3))             #error matrix
    B_each, B_var = [], []              #biases and its variance
    pose_table = []                     #rows for the printout in cmd_calibrate
    for i in range(3):                  #one loop per axis (x, y, z)
        up, dn = poses[(i, +1)], poses[(i, -1)]
        mu_up, mu_dn = up.mean(axis=0), dn.mean(axis=0)                     #mean
        sd_up, sd_dn = up.std(axis=0, ddof=1), dn.std(axis=0, ddof=1)       #standard dev (ddof=1 = sample std dev)
        se_up, se_dn = sd_up / np.sqrt(len(up)), sd_dn / np.sqrt(len(dn))   #standard erors = stdev/sqrt(n)
        S[:, i] = (mu_up - mu_dn) / (2 * G)                                 #column i of S: full up-to-down swing / 2g
        print(f"S: {S}")                                                    #debug print: S as it fills in column by column
        S_se[:, i] = np.sqrt(se_up ** 2 + se_dn ** 2) / (2 * G)             #summing uncertainties se
        B_each.append((mu_up + mu_dn) / 2)                                  #average distance from zero
        B_var.append((se_up ** 2 + se_dn ** 2) / 4)                         #variance of that average
        pose_table.append((AXES[i] + "up", len(up), mu_up, sd_up))
        pose_table.append((AXES[i] + "down", len(dn), mu_dn, sd_dn))
    B_each = np.array(B_each)
    B = B_each.mean(axis=0)                                                 #final bias = average of the 3 axis estimates
    B_se = np.sqrt(np.sum(B_var, axis=0)) / 3                               #standard error of that average
    return S, S_se, B, B_se, B_each, pose_table


def correct(raw, S, B):
    """a_corrected = inv(S) @ (m - B), applied to an (N, 3) array (or a single length-3 vector)."""
    arr = np.asarray(raw, float)
    single = arr.ndim == 1                                      #remember if we were given one reading, not many
    arr = np.atleast_2d(arr)                                    #treat everything as rows of (x, y, z)
    out = (np.linalg.inv(S) @ (arr - B).T).T                    #subtract bias, then undo the scale/misalignment
    return out[0] if single else out


def validate(folder, S, B):
    files = sorted(glob.glob(os.path.join(folder, "val_*.csv")))        #tilted still recordings not used in the fit
    if not files:
        print("\nNo val_*.csv files found, skipping validation. (Record the sensor held still at a few")
        print("tilts NOT used in the fit; the corrected magnitude should equal 9.807 at any tilt.)")
        return None
    print("\nVALIDATION (stationary tilts not used in the fit; the magnitude should equal g = %.4f)" % G)
    print(f"  {'file':<22}{'|raw| error':>14}{'|corrected| error':>20}   (m/s^2)")
    err_raw, err_cor = [], []
    for p in files:
        mean = trial_mean(p)
        er = np.linalg.norm(mean) - G                           #length of the raw vector minus g
        ec = np.linalg.norm(correct(mean, S, B)) - G            #same after calibration (should be closer to 0)
        err_raw.append(er)
        err_cor.append(ec)
        print(f"  {os.path.basename(p):<22}{er:>14.4f}{ec:>20.4f}")
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))       #root mean square of a list of errors
    print(f"  RMS error: raw {rms(err_raw):.4f}  ->  corrected {rms(err_cor):.4f} m/s^2")
    return {"rms_raw": rms(err_raw), "rms_corrected": rms(err_cor), "n_files": len(files)}


def cmd_calibrate(args):
    poses = gather_poses(args.dir)                                  #read every pose file in the folder
    if not poses:
        raise SystemExit(f"No files like xup_t1.csv found in '{args.dir}'.")
    S, S_se, B, B_se, B_each, pose_table = solve_calibration(poses)

    #everything from here to the json is printing the results
    np.set_printoptions(precision=5, suppress=True)                 #no scientific notation in the arrays
    print("PER-POSE TRIAL MEANS (m/s^2): mean of the trial means, and the spread between trials (repeatability)")
    print(f"  {'pose':<8}{'n':>3}   {'mean ax, ay, az':<34}{'std across trials'}")
    for name, n, mu, sd in pose_table:
        print(f"  {name:<8}{n:>3}   {np.array2string(mu, precision=4):<34}{np.array2string(sd, precision=4)}")

    print("\nBIAS B (m/s^2):   ", np.array2string(B, precision=4), "  +/-", np.array2string(B_se, precision=4), "(standard error)")
    print("Bias estimated separately from each axis pair (they should agree; large disagreement = fixture tilt):")
    for i in range(3):
        print(f"   from {AXES[i]} up/down:", np.array2string(B_each[i], precision=4))

    print("\nSCALE/MISALIGNMENT MATRIX S (rows = measured axis, columns = input axis):")
    print(np.array2string(S, precision=5))
    print("Standard error of each element:")
    print(np.array2string(S_se, precision=5))
    print("\nDiagonal scale factors:  Sx = %.5f  Sy = %.5f  Sz = %.5f" % tuple(np.diag(S)))     #should be near 1.0
    off = S - np.diag(np.diag(S))                                   #S with the diagonal zeroed = cross-axis terms only
    print("Largest off-diagonal term: %.4f  (treat with caution: a tilted fixture shows up here)" % np.abs(off).max())

    val = validate(args.dir, S, B)                                  #independent check on the tilted recordings

    #everything needed to reuse this calibration later gets saved in one file
    result = {
        "date": str(date.today()),
        "settings": SETTINGS,
        "g": G,
        "B": B.tolist(),                                            #.tolist() because json cannot store numpy arrays
        "B_standard_error": B_se.tolist(),
        "S": S.tolist(),
        "S_standard_error": S_se.tolist(),
        "S_inverse": np.linalg.inv(S).tolist(),
        "trials_per_pose": {f"{AXES[a]}{'up' if s > 0 else 'down'}": len(v) for (a, s), v in sorted(poses.items())},
        "validation": val,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved calibration to {args.out}")


def load_calibration(path="imu_calibration.json"):
    """Returns (S, B) as numpy arrays from a saved calibration file."""
    with open(path) as f:
        cal = json.load(f)
    return np.array(cal["S"]), np.array(cal["B"])


# ----------------------------------------------------------------------------
# Allan deviation
# ----------------------------------------------------------------------------
def allan_deviation(y, fs, n_points=60):
    """Overlapping Allan deviation of a 1-D series y sampled at fs (Hz)."""
    y = np.asarray(y, float)
    n = len(y)
    theta = np.concatenate(([0.0], np.cumsum(y)))   #running sum: the sum of any block of samples is then one subtraction
    max_m = max(n // 9, 2)                          #longest block allowed, so the longest tau still has several blocks to compare
    ms = np.unique(np.floor(np.logspace(0, np.log10(max_m), n_points)).astype(int))     #block sizes spaced evenly on a log axis
    taus, adevs = [], []
    for m in ms:
        if 2 * m >= n:                              #need at least two blocks to compare
            break
        avg = (theta[m:] - theta[:-m]) / m          # overlapping cluster averages
        d = avg[m:] - avg[:-m]                      #difference between each block average and the next block
        taus.append(m / fs)                         #tau = how many seconds one block covers
        adevs.append(np.sqrt(0.5 * np.mean(d ** 2)))    #Allan deviation = sqrt(half the mean squared difference)
    return np.array(taus), np.array(adevs)


def cmd_allan(args):
    df = load_csv(args.file)
    n = len(df)
    duration = df["t"].iloc[-1] - df["t"].iloc[0]
    fs = (n - 1) / duration                         #sample rate estimated from the timestamps
    print(f"Loaded {n} samples over {duration/60:.1f} min, estimated rate {fs:.1f} Hz")
    if duration < 600:
        print("WARNING: under 10 minutes of data. Bias instability needs a long recording (30-60 min is better).")
    dt = np.diff(df["t"].to_numpy())                #time between samples
    if dt.std() / dt.mean() > 0.05:                 #more than 5% spread = loop falling behind
        print("WARNING: sampling is irregular; the Allan result assumes evenly spaced samples.")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for col, color in zip(AXIS_COLS, ["tab:blue", "tab:orange", "tab:green"]):      #one curve per axis
        taus, adev = allan_deviation(df[col].to_numpy(), fs)
        ax.loglog(taus, adev, color=color, label=col)

        #white noise region: short taus (about 10 samples up to 1 s) where the curve falls with a slope of -1/2
        lo, hi = 10.0 / fs, 1.0
        sel = (taus >= lo) & (taus <= hi)
        #on that slope adev = N/sqrt(tau), so adev*sqrt(tau) is flat and its median is the white noise coefficient N
        N = float(np.median(adev[sel] * np.sqrt(taus[sel]))) if sel.any() else float("nan")
        k = int(np.argmin(adev))                    #index of the bottom of the curve
        msg = (f"  {col}: white-noise coefficient N ~ {N:.5f} m/s/sqrt(s) "
               f"(one-sided noise density ~ {np.sqrt(2) * N:.5f} (m/s^2)/sqrt(Hz)); ")
        if k >= len(adev) - 2:                      #minimum is at the far right = it never turned back up
            msg += "curve still falling at the longest tau, so record longer to see bias instability."
        else:
            #bias instability = the curve minimum divided by 0.664 (standard factor for flicker-type bias drift)
            msg += (f"minimum {adev[k]:.5f} m/s^2 at tau = {taus[k]:.1f} s -> "
                    f"bias instability ~ {adev[k] / 0.664:.5f} m/s^2")
        print(msg)

    ax.set_xlabel("Averaging time tau (s)")
    ax.set_ylabel("Allan deviation (m/s^2)")
    ax.set_title(f"Allan deviation: {os.path.basename(args.file)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_png = os.path.splitext(args.file)[0] + "_allan.png"         #saves the png next to the csv
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")
    plt.show()                                                      #blocks until you close the plot window


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="MPU-6050 static calibration and Allan deviation")
    sub = p.add_subparsers(dest="cmd", required=True)               #the command word: calibrate or allan

    pc = sub.add_parser("calibrate", help="six-position calibration (+ validation if val_*.csv exist)")
    pc.add_argument("--dir", default="calibration_data", help="folder with xup_t1.csv, ... files")
    pc.add_argument("--out", default="imu_calibration.json", help="where to save the result")
    pc.set_defaults(func=cmd_calibrate)

    pa = sub.add_parser("allan", help="Allan deviation of one long stationary recording")
    pa.add_argument("--file", required=True, help="CSV of a long stationary recording")
    pa.set_defaults(func=cmd_allan)

    args = p.parse_args(sys.argv[1:] or ["calibrate"])              #plain Run button in PyCharm (no parameters) = calibrate
    args.func(args)


if __name__ == "__main__":
    main()