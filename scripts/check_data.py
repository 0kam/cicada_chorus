import os
from glob import glob
from pathlib import Path

wavs = sorted([Path(p).stem for p in glob("../data/train/wav/*.wav")])
labels = sorted([Path(p).stem for p in glob("../data/train/label/*.txt")])

for wav in wavs:
    if wav not in labels:
        print(f"Missing label for {wav}")
        label_path = f"../data/train/label/{wav}.txt"
        with open(label_path, "w") as f:
                f.write("")

for label in labels:
    if label not in wavs:
        print(f"Missing wav for {label}")