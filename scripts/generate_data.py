from utils.data import AudioPredictionDataset
from utils.utils import log_params_from_omegaconf_dict
from utils.chorus_generator import get_audio_datasets, generate
from cnn.cnn import PretrainedCNNClassifier

from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import pandas as pd
import torch
import os
from joblib import Parallel, delayed

cfg_path = "/home/okamoto/cicada_chorus/mlruns/514920413459784680/364cd1e7c2104506863fa464a34efb44/artifacts/config.yaml"
cfg = OmegaConf.load(cfg_path)
cfg.generation.n_train = 50000
cfg.generation.n_test = 10000

train_wav_dir = '/home/okamoto/cicada_chorus/data/tuning/tmp/train/source'
train_label_dir = '/home/okamoto/cicada_chorus/data/tuning/tmp/train/label'

test_wav_dir = '/home/okamoto/cicada_chorus/data/tuning/tmp/test/source'
test_label_dir = '/home/okamoto/cicada_chorus/data/tuning/tmp/test/label'

cicadas, bgs, others = get_audio_datasets(
    cfg.general.cicada_dir,
    cfg.general.background_dir,
    cfg.general.others_dir
)

cnn_cfg = OmegaConf.load('/home/okamoto/cicada_chorus/scripts/data_tuning/cnn_config.yaml')

def tune_dataset(cfg: DictConfig):
    bg_ratio = 1 - (cfg.general.cicada_ratio + cfg.general.others_ratio)
    if bg_ratio < 0:
        raise ValueError("The sum of cicada_ratio and others_ratio must be less than 1.")
    category_ratio = [cfg.general.cicada_ratio, cfg.general.others_ratio, bg_ratio]

    def generate_chorus(wav_path, label_path):            
        generate(
            wav_path,
            label_path,
            cicadas,
            bgs,
            others,
            cfg.general.sr,
            cfg.generation.length,
            category_ratio,
            cfg.generation.cicadas.weights,
            list(cfg.generation.cicadas.popsize.values()),
            list(cfg.generation.cicadas.distance.values()),
            list(cfg.generation.cicadas.n_species.values()),
            cfg.generation.bgs.weights,
            cfg.generation.others.weights,
            list(cfg.generation.others.popsize.values()),
            list(cfg.generation.others.distance.values()),
            list(cfg.generation.others.n_species.values()),
            list(cfg.augs.time_stretch.values()),
            list(cfg.augs.pitch_shift.values())
        )
    
    # Generate train data
    # if os.path.exists(train_wav_dir):
    #     os.system("rm -rf {}".format(train_wav_dir))
    # if os.path.exists(train_label_dir):
    #     os.system("rm -rf {}".format(train_label_dir))
    # Parallel(n_jobs=-1, verbose=10)([delayed(generate_chorus)(f"{train_wav_dir}/{i}.wav", f"{train_label_dir}/{i}.txt") for i in range(cfg.generation.n_train)])

    # # Generate test data
    # if os.path.exists(test_wav_dir):
    #     os.system("rm -rf {}".format(test_wav_dir))
    # if os.path.exists(test_label_dir):
    #     os.system("rm -rf {}".format(test_label_dir))
    # Parallel(n_jobs=-1, verbose=10)([delayed(generate_chorus)(f"{test_wav_dir}/{i}.wav", f"{test_label_dir}/{i}.txt") for i in range(cfg.generation.n_test)])
    
    # Model training
    best_val_f1_mean = 0
    model = PretrainedCNNClassifier(cnn_cfg)
    for epoch in range(cnn_cfg.general.epochs):
        log_params_from_omegaconf_dict(cfg)
        train_loss = model.train()
        val_loss, res = model.val()
        f1s = []
        for c in model.c.dataset.label_names:
            for m in ["precision", "recall", "f1"]:
                value = res[res["label"]==c][m].values[0]
                if m == "f1":
                    f1s.append(value)
        f1_mean = sum(f1s) / len(f1s)
        if f1_mean >= best_val_f1_mean:
            best_val_f1_mean = f1_mean
            best_model = model.model
    
    torch.save(best_model.state_dict(), "/home/okamoto/cicada_chorus/data/tuning/tmp/best_model.pth")
    
    # Prediction of HS01
    dataset = AudioPredictionDataset(
        source_dir = '/home/okamoto/HDD4TB/Cicadasong_Detect/records/HS01',
        win_sec = model.c.dataset.win_sec,
        stride_sec = model.c.dataset.win_sec / 2,
        sr = 16000
    )

    # Using best weights to validate model on real-world data
    model.model = best_model
    preds = model.predict(dataset)

    y_preds = preds.max(axis=1).values

    df = pd.read_csv('/home/okamoto/HDD4TB/Cicadasong_Detect/data/handwork/handwork_HS01.csv')
    y_true = df.filter(['Aburazemi', 'Higurashi', 'Kumazemi', 'Minminzemi', 'Niiniizemi', 'Tsukutsukuboushi']).to_numpy()
    y_true = torch.tensor(y_true, dtype=torch.float32)

    res = model.classification_scores(y_true, y_preds, threshold=0.5)
    f1s = []
    for c in model.c.dataset.label_names:
        for m in ["precision", "recall", "f1"]:
            value = res[res["label"]==c][m].values[0]
            if (m == "f1") and (c != "higurashi") and (c != "kumazemi"): # higurashi and kumazemi were not observed in HS01
                f1s.append(value)
    res.to_csv("/home/okamoto/cicada_chorus/data/tuning/tmp/res_HS01.csv")
    f1_mean = sum(f1s) / len(f1s)
    return f1_mean

if __name__ == "__main__":
    tune_dataset(cfg)
