import numpy as np
import pandas as pd

def load_recording(filepath):
    #Returns ax, ay, az and sampling rate
    df = pd.read_csv(filepath, on_bad_lines='skip')

    df = df.dropna()                                    #removing rows and rows where time doesn't increase
    df = df[ df["timestamp_us"].diff().fillna(1) > 0]

    t = df["timestamp_us"].values * 1e-6                #[s]
    fs = (len(t)-1)/(t[-1]-t[0])

    raw = df[["ax_raw", "ay_raw", "az_raw"]].values.astype(float)
    acc = raw/8192 *9.80665                             # m/s^2

    return acc, fs

if __name__ == "__main__":
    acc, fs = load_recording('fanlowt1.csv')
    print("samples:", len(acc), " fs: %.1f Hz" % fs)
    print("mean per axis:", acc.mean(axis=0))
    print("std per axis:", acc.std(axis=0))

