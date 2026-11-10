import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

def load_recording(filepath):
    #input filepath for data
    #Returns ax, ay, az and sampling rate

    df = pd.read_csv(filepath, on_bad_lines='skip')
    df = df.dropna()                                    #removing rows and rows where time doesn't increase
    df = df[ df["timestamp_us"].diff().fillna(1) > 0]

    t = df["timestamp_us"].values * 1e-6                #[s]
    fs = (len(t)-1)/(t[-1]-t[0])

    raw = df[["ax_raw", "ay_raw", "az_raw"]].values.astype(float)
    acc = raw/8192 *9.80665                             # m/s^2
    trim = int(2 * fs)
    acc = acc[trim:-int(1 * fs)]
    return acc, fs

def make_windows(acc, fs, win_s=1, overlap = 0.5):
    #input: accelerations, sampling freq(fs), seconds in a window (win_s), overlap percentage
    #output: array of shape (# of windows, window length, 3)
    win_len = int(win_s*fs)
    step = int(win_len* (1-overlap))
    starts =range(0, len(acc) - win_len+1, step)
    return np.stack([acc[s:s+win_len] for s in starts])

def window_features(w, fs):
    #input: w - a single window; fs - sampling frequency
    #output: array of features
    w = w-w.mean(axis=0)

    #time domain features
    rms = np.sqrt((w**2).mean(axis=0))          #vibration "strength"
    kurt = kurtosis(w, axis=0)                  #peakiness
    crest = np.abs(w).max(axis=0)/rms           #largest peak rel to rms

    #freq domain features
    n = len(w)
    hann = np.hanning(n)                        #bell shaped weights to fade signal to zero at edge of windows
    spec = np.abs(np.fft.rfft(w*hann[:,None], axis=0)) * 2/ hann.sum()
    S= np.sqrt((spec**2).sum(axis=1))           #combines x,y,z
    freqs = np.fft.rfftfreq(n, 1/fs)

    band = (freqs > 8) & (freqs < 30)           #freq bins
    f1 = freqs[band][np.argmax(S[band])]        #tallest peak

    def amp_at(f0):
        near = np.abs(freqs-f0) <1.5
        return S[near].max()

    amps = [amp_at(k*f1) for k in (1, 2, 3, 4)]
    ratio_2x = amps[1]/ amps[0]
    Eband = (freqs >5) & (freqs <100)
    energy = ((S[Eband]**2)).sum()

    features = np.concatenate([rms, kurt, crest, [f1], amps, [ratio_2x, energy]])
    return features

def build_dataset(recordings):
    #input: recordings - [(path, label, group)]. Label is healthy(0) or faulty (1). groups is name of recording
    #returns: X, y, groups
    X, y, groups = [], [], []                       #initialize arrays

    for path, label, group in recordings:
        acc, fs = load_recording(path)              #load data
        windows = make_windows(acc, fs)             #make windows
        for w in windows:
            X.append(window_features(w,fs))         #collect featurs from windows
            y.append(label)
            groups.append(group)
    return np.array(X), np.array(y), np.array(groups)

if __name__ == "__main__":

    features = ["rms_x", "rms_y", "rms_z", "kurt_x", "kurt_y", "kurt_z",
                "crest_x", "crest_y", "crest_z", "f1",
                "amp_1x", "amp_2x", "amp_3x", "amp_4x", "ratio_2x_1x", "band_energy"]

    #acc, fs = load_recording('fanlowt1.csv')
    #print("samples:", len(acc), " fs: %.1f Hz" % fs)
    #print("mean per axis:", acc.mean(axis=0))
    #print("std per axis:", acc.std(axis=0))

    #w = make_windows(acc, fs)
    #print("windows: ", w.shape)

    #feat = window_features(w[0], fs)
    #print("n features:", len(feat))
    #print("f1 = %.1f Hz" % feat[9])
    #print(np.round(feat,3))
    #print(feat)
    data_dir = Path("model_data")
    recordings = [ (data_dir/ "A_low_healthy_t1.csv", 0, "A_low_healthy_1"),
                   (data_dir/ "A_low_healthy_t2.csv", 0, "A_low_healthy_2"),
                   (data_dir/ "A_low_healthy_t3.csv", 0, "A_low_healthy_3"),
                   (data_dir / "B_low_healthy_t1.csv", 0, "B_low_healthy_1"),
                   (data_dir / "B_low_healthy_t2.csv", 0, "B_low_healthy_2"),
                   (data_dir / "B_low_healthy_t3.csv", 0, "B_low_healthy_3"),
                   (data_dir/ "B_low_batt_t1.csv", 1, "B_low_batt_1"),
                   (data_dir/ "B_low_batt_t2.csv", 1, "B_low_batt_2"),
                   (data_dir/ "B_low_batt_t3.csv", 1, "B_low_batt_3")]
    X, y, groups = build_dataset(recordings)
    print("X:", X.shape, " y counts:", np.bincount(y), " recordings:", len(set(groups)))

    models = {
        "logistic regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0),
    }
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for name, model in models.items():
        y_predictions = cross_val_predict(model, X, y, cv=cv, groups=groups)
        print("\n" + name)
        print("accuracy:", accuracy_score(y, y_predictions))
        print("confusion matrix (rows = true, columns = predicted):")
        print(confusion_matrix(y, y_predictions))

    forest = models["random forest"].fit(X, y)
    order = np.argsort(forest.feature_importances_)[::-1]
    for i in order:
        print(f"{features[i]:<12} {forest.feature_importances_[i]:.3f}")
