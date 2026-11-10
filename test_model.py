import joblib
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import confusion_matrix
from model_pipeline import score_recording

levels = {"healthy": 0, "1tape": 1, "2tape": 2, "batt": 3}
level_names = ["healthy", "1 wrap", "2 wraps", "batt"]

model = joblib.load("fault_model.joblib")
test_dir = Path("model_test_data")

rows = []
tally = defaultdict(lambda: [0,0])

for p in sorted(test_dir.glob("*.csv")):
    parts = p.stem.split("_")
    if len(parts)<3 or parts[2] not in levels:
        continue
    fan, level = parts[0], levels[parts[2]]
    p_fault = score_recording(p, model)
    mean_p = p_fault.mean()
    correct = (mean_p >0.5) == (level>0)

    rows.append((p.name, fan, level, mean_p))
    tally[(fan, level_names[level])][0] += int(correct)
    tally[(fan, level_names[level])][1] += 1
    print(f"{p.name:<26} true: {'fault' if level else 'healthy':<8}"
          f" predicted: {'FAULTY' if mean_p >0.5 else 'HEALTHY':<8}"
          f" p(fault) = {mean_p:.2f} +/- {p_fault.std():.2f} {'OK' if correct else 'WRONG'}")

    if not rows:
        raise SystemExit(f"No usable files in {test_dir}")

    print("\n Recordings correct, by fan and level:")
    for (fan, name), (c, n) in sorted(tally.items()):
        print(f" fan {fan} {name:<8} {c} / {n}")

    y_true = np.array([int(r[2] > 0) for r in rows])
    y_pred = np.array([int(r[3] > 0.5) for r in rows])
    print("\n Confusion matrix (rows = true healthy / faulty, columns = predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0,1]))
    print(f"Overall: {int((y_true==y_pred).sum())} of {len(y_true)} recordings correct")