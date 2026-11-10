import json
import joblib
from model_pipeline import diagnose

def show_summary():
    with open("measurement_summary.json") as f:
        rows = json.load(f)
    print("\n1x vibration amplitude by condition (calibrated, m/s^2)")
    print(f"  {'condition':<10}{'n':>3}{'mean':>9}{'+/- SE':>9}{'spread':>9}")
    for r in rows:
        print(f"  {r['condition']:<10}{r['n']:>3}{r['mean']:>9.3f}{r['se']:>9.3f}{r['sd']:>9.3f}")

def show_calibration():
    with open("imu_calibration.json") as f:
        c = json.load(f)
    print("\nCalibration (six-position test), +/- one standard error")
    for i, ax in enumerate("xyz"):
        print(f"  {ax}: bias {c['B'][i]:+.4f} +/- {c['B_standard_error'][i]:.4f} m/s^2,"
              f"  scale {c['S'][i][i]:.5f} +/- {c['S_standard_error'][i][i]:.5f}")

def show_noise():
    with open("noise_summary.json") as f:
        d = json.load(f)
    print(f"\nSensor noise (Allan deviation; {d['source']})")
    print(f"  {'axis':<6}{'white noise N':>15}{'density':>14}{'bias instability':>19}")
    print(f"  {'':<6}{'m/s/sqrt(s)':>15}{'(m/s2)/sqrtHz':>14}{'m/s2':>19}")
    for ax, v in d["axes"].items():
        bi = f"{v['bias_instability']:.5f}" if v["bias_instability"] is not None else "not reached"
        print(f"  {ax:<6}{v['N']:>15.5f}{v['density']:>14.5f}{bi:>19}")


model = joblib.load("fault_model.joblib")

while True:
    path = input("\nCSV to analyze ('summary', 'calibration', 'noise', or 'exit'): ").strip().strip('"')
    if path.lower() in ("exit", "quit"):
        print("Done.")
        break
    if path.lower() == "summary":
        show_summary()
    elif path.lower() == "calibration":
        show_calibration()
    elif path.lower() == "noise":
        show_noise()
    elif path:
        try:
            diagnose(path, model)
        except FileNotFoundError:
            print(f"Could not find '{path}'. Check the name or include the folder.")
        except Exception as e:
            print(f"Could not analyze '{path}': {e}")