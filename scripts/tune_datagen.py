from utils.data import AudioPredictionDataset
from utils.utils import log_params_from_omegaconf_dict
from utils.chorus_generator import get_audio_datasets, generate
from cnn.cnn import PretrainedCNNClassifier

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import mlflow
from datetime import datetime
import pandas as pd
import torch
import os
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent import futures

cfg = OmegaConf.load('/home/okamoto/cicada_chorus/scripts/data_tuning/config.yaml')

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

@hydra.main(config_path="data_tuning", config_name="config")
def tune_dataset(cfg: DictConfig):
    # MLFlow settings
    mlflow.set_tracking_uri("file:// /home/okamoto/cicada_chorus/mlruns")
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="{}".format(datetime.now().strftime("%Y%m%d%H%M%S"))):
        mlflow.log_artifact(".hydra/config.yaml")
        mlflow.log_artifact(".hydra/hydra.yaml")
        mlflow.log_artifact(".hydra/overrides.yaml")
        mlflow.log_artifact("/home/okamoto/cicada_chorus/scripts/data_tuning/cnn_config.yaml")
        # Data generation
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
        if os.path.exists(train_wav_dir):
            os.system("rm -rf {}".format(train_wav_dir))
        if os.path.exists(train_label_dir):
            os.system("rm -rf {}".format(train_label_dir))
        Parallel(n_jobs=-1, verbose=10)([delayed(generate_chorus)(f"{train_wav_dir}/{i}.wav", f"{train_label_dir}/{i}.txt") for i in range(cfg.generation.n_train)])

        # Generate test data
        if os.path.exists(test_wav_dir):
            os.system("rm -rf {}".format(test_wav_dir))
        if os.path.exists(test_label_dir):
            os.system("rm -rf {}".format(test_label_dir))
        Parallel(n_jobs=-1, verbose=10)([delayed(generate_chorus)(f"{test_wav_dir}/{i}.wav", f"{test_label_dir}/{i}.txt") for i in range(cfg.generation.n_test)])
        
        # Model training
        best_val_f1_mean = 0
        model = PretrainedCNNClassifier(cnn_cfg)
        mlflow.log_artifact(".hydra/config.yaml")
        mlflow.log_artifact(".hydra/hydra.yaml")
        mlflow.log_artifact(".hydra/overrides.yaml") 
        for epoch in range(cnn_cfg.general.epochs):
            log_params_from_omegaconf_dict(cfg)
            train_loss = model.train()
            val_loss, res = model.val()
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            f1s = []
            for c in model.c.dataset.label_names:
                for m in ["precision", "recall", "f1"]:
                    value = res[res["label"]==c][m].values[0]
                    mlflow.log_metric('val_{}_{}'.format(m, c), value, step=epoch)
                    if m == "f1":
                        f1s.append(value)
            f1_mean = sum(f1s) / len(f1s)
            mlflow.log_table(res, artifact_file='val_result.csv')
            mlflow.pytorch.log_model(model.model, "model_epoch_{}".format(epoch))
            if f1_mean >= best_val_f1_mean:
                best_val_f1_mean = f1_mean
                best_model = model.model
                mlflow.pytorch.log_model(model.model, "best_model")
                mlflow.log_metric('best_test_f1', best_val_f1_mean, step=epoch)

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
                mlflow.log_metric('HS01_{}_{}'.format(m, c), value)
                if (m == "f1") and (c != "higurashi") and (c != "kumazemi"): # higurashi and kumazemi were not observed in HS01
                    f1s.append(value)
        f1_mean = sum(f1s) / len(f1s)
        mlflow.log_metric('HS01_last_f1_mean', f1_mean)
    return f1_mean

if __name__ == "__main__":
    tune_dataset()