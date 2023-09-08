import mlflow
from data import AudioPredictionDataset
import omegaconf
from cnn.cnn import PretrainedCNNClassifier
import pandas as pd
import torch
import numpy as np
import torchaudio

d = "/home/okamoto/cicada_chorus/mlruns/347298962134178980/977c7e593c2c4476bc1c6a556d4f2edf/"

cfg = omegaconf.OmegaConf.load(d + "artifacts/config.yaml")
logged_model = d + "artifacts/best_model"

model = PretrainedCNNClassifier(cfg)
model.model = mlflow.pytorch.load_model(logged_model)

# Load data
dataset = AudioPredictionDataset(
    source_dir = '/home/okamoto/HDD4TB/Cicadasong_Detect/records/HS01',
    win_sec = model.c.dataset.win_sec,
    stride_sec = model.c.dataset.win_sec / 2,
    sr = 16000
)

# Prediction
model.c.dataset.label_names
preds = model.predict(dataset)

y_preds = preds.max(axis=1).values

# Classification scores
df = pd.read_csv('/home/okamoto/HDD4TB/Cicadasong_Detect/data/handwork/handwork_HS01.csv')
y_true = df.filter(['Aburazemi', 'Higurashi', 'Kumazemi', 'Minminzemi', 'Niiniizemi', 'Tsukutsukuboushi']).to_numpy()
y_true = torch.tensor(y_true, dtype=torch.float32)

model.classification_scores(y_true, y_preds, threshold=0.5)

y_pred = (y_preds > 0.5).float()
y_pred_segments = (preds > 0.5).float()

# 判別結果が間違っていた音源ファイルを種ごとに取得
import os
from pathlib import Path
os.makedirs('/home/okamoto/cicada_chorus/data/wrongs', exist_ok=True)

y_true = y_true.numpy()
y_pred = y_pred.numpy()
y_pred_segments = y_pred_segments.numpy()

for i, label in enumerate(model.c.dataset.label_names):
    fns_out_dir = '/home/okamoto/cicada_chorus/data/wrongs/{}/fns/'.format(label)
    fps_out_dir = '/home/okamoto/cicada_chorus/data/wrongs/{}/fps/'.format(label)
    os.makedirs(fns_out_dir, exist_ok=True)
    os.makedirs(fps_out_dir, exist_ok=True)
    # y_trueがlabelでy_predがlabelでないものを取得
    fns = np.array(dataset.source_files)[((y_true[:, i] == 1) & (y_pred[:, i] == 0)).astype(bool)].tolist()
     # y_trueがlabelでないがy_predがlabelなものを取得
    fps = np.array(dataset.source_files)[((y_true[:, i] == 0) & (y_pred[:, i] == 1)).astype(bool)].tolist()
    fps_segment = y_pred_segments[((y_true[:, i] == 0) & (y_pred[:, i] == 1)).astype(bool)]
    for fn in fns:
        out = fns_out_dir + Path(fn).parts[-3] + '_' + Path(fn).parts[-2] + '.wav'
        os.symlink(fn, out)
    for j, fp in enumerate(fps):
        a, sr = torchaudio.load(fp)
        a = a[0]
        for k, p in enumerate(fps_segment[j]):
            if p[i] == 1:
                out = fps_out_dir + Path(fp).parts[-3] + '_' + Path(fp).parts[-2] + '_' + str(k) + '.wav'
                start = int(k * model.c.dataset.win_sec / 2 * sr)
                end = int(start + model.c.dataset.win_sec * sr)
                torchaudio.save(out, a[start:end].unsqueeze(0), sr)