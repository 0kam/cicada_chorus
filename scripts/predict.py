import mlflow
from utils.data import AudioPredictionDataset
import omegaconf
from cnn.cnn import PretrainedCNNClassifier
import pandas as pd
import torch
import numpy as np
import torchaudio
from pathlib import Path
import os
import yaml
from glob import glob

d = "/home/okamoto/cicada_chorus/mlruns/347298962134178980/977c7e593c2c4476bc1c6a556d4f2edf/"

cfg = omegaconf.OmegaConf.load(d + "artifacts/config.yaml")
logged_model = d + "artifacts/best_model"

model = PretrainedCNNClassifier(cfg)
model.model = mlflow.pytorch.load_model(logged_model)

with open('{}/meta.yaml'.format(d), 'r') as yml:
    experiment_name = yaml.safe_load(yml)["run_name"]

out_dir = '/home/okamoto/cicada_chorus/predictions/{}'.format(experiment_name)
os.makedirs(out_dir, exist_ok=True)

def predict_dir(dir_path):
    # Dataset
    dataset = AudioPredictionDataset(
        source_dir = dir_path,
        win_sec = model.c.dataset.win_sec,
        stride_sec = model.c.dataset.win_sec / 2,
        sr = 16000
    )
    site_name = Path(dir_path).name
    print("processing {} ...".format(site_name))
    # Prediction
    preds = model.predict(dataset)
    #y = preds.max(axis=1).values
    #y = (y > 0.5).float()
    filenames = np.tile(np.array(dataset.source_files).reshape(-1, 1), (1, preds.shape[1])).reshape(-1)
    index = np.tile(np.arange(preds.shape[1]).reshape(1, -1), (preds.shape[0], 1)).reshape(-1)
    df = pd.DataFrame({'file_name': filenames, 'index': index})
    preds2 = preds.reshape(-1, preds.shape[2])
    for i, label in enumerate(model.c.dataset.label_names):
        df[label] = preds2[:, i]
    out_path = '{}/{}.csv'.format(out_dir, site_name)
    df.to_csv(out_path, index=False)

dirs = sorted(glob("/media/HDD10TB/cicadasong2023/NE*"))
for target_dir in dirs:
    predict_dir(target_dir + "/")