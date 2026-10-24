import pandas as pd
import matplotlib.pyplot as plt

pos_filenames = [f"pos_ztest{i}.csv" for i in range(1, 11)]
neg_filenames = [f"neg_ztest{i}.csv" for i in range(1, 11)]
sensor_cols = ["ax", "ay", "az"]
test_axis = "az"  # the axis that actually sees +/-g in the z tests

# Use 9.80665 if your raw data is in m/s^2 (output stays in m/s^2).
# Use 1.0 if you want the output in g.
G = 9.80665


def read_csvs(filenames):
    dfs = [pd.read_csv(fn, on_bad_lines="skip").dropna() for fn in filenames]
    means = [df[sensor_cols].mean() for df in dfs]
    total_mean = pd.concat(means, axis=1).mean(axis=1)
    return dfs, total_mean


# Bias: the midpoint of the +g and -g readings (they should be symmetric about it)
def offset(mean_pos, mean_neg):
    return (mean_pos + mean_neg) / 2


# Sensitivity: half the span between +g and -g readings, divided by g
def scale_factor(mean_pos, mean_neg, g):
    return (mean_pos - mean_neg) / (2 * g)


def calibrate(df, off, scale):
    out = df.copy()
    out[sensor_cols] = (df[sensor_cols] - off) / scale
    return out


def plotter(raw_pos, cal_pos, raw_neg, cal_neg):
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex="col")
    cases = [("Positive z test", raw_pos, cal_pos), ("Negative z test", raw_neg, cal_neg)]

    for j, (title, raw, cal) in enumerate(cases):
        t = raw["timestamp_ms"]
        for i, col in enumerate(sensor_cols):
            ax = axes[i, j]
            ax.plot(t, raw[col], color="tab:red", alpha=0.6, label=f"Raw {col}")
            ax.plot(t, cal[col], color="tab:blue", linestyle="--", label=f"Calibrated {col}")
            ax.set_ylabel(f"{col} (m/s²)" if G != 1.0 else f"{col} (g)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            if i == 0:
                ax.set_title(title)
        axes[-1, j].set_xlabel("Timestamp (ms)")

    fig.suptitle("Raw vs calibrated accelerometer data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main():
    pos_dfs, pos_mean = read_csvs(pos_filenames)
    neg_dfs, neg_mean = read_csvs(neg_filenames)

    off = offset(pos_mean, neg_mean)

    # The z tests only excite az, so ax/ay have no usable +/-g span.
    # Dividing by their ~0 span would blow up, so only az gets a scale factor.
    # For ax/ay you need x and y tests too; until then they get offset only.
    scale = pd.Series(1.0, index=sensor_cols)
    scale[test_axis] = scale_factor(pos_mean[test_axis], neg_mean[test_axis], G)

    print("Offsets:\n", off, "\n")
    print("Scale factors:\n", scale, "\n")

    cal_pos = [calibrate(df, off, scale) for df in pos_dfs]
    cal_neg = [calibrate(df, off, scale) for df in neg_dfs]

    plotter(pos_dfs[0], cal_pos[0], neg_dfs[0], cal_neg[0])


if __name__ == "__main__":
    main()