from siamese.siamese import SiameseCNNModel
from utils.utils import log_params_from_omegaconf_dict
from utils.data import AudioPredictionDataset

import hydra
from omegaconf import DictConfig
import mlflow
from datetime import datetime
import pandas as pd
import omegaconf
import torch

cfg = omegaconf.OmegaConf.load('/home/okamoto/cicada_chorus/scripts/siamese/config.yaml')

@hydra.main(config_path="siamese", config_name="config")
def train(cfg: DictConfig):
    model = SiameseCNNModel(cfg)
    mlflow.set_tracking_uri("file:// /home/okamoto/cicada_chorus/mlruns")
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    best_val_loss = 1000.0
    with mlflow.start_run(run_name="{}_{}_{}".format(cfg.model.model_name, cfg.feature.feature, datetime.now().strftime("%Y%m%d%H%M%S"))):
        mlflow.log_artifact(".hydra/config.yaml")
        mlflow.log_artifact(".hydra/hydra.yaml")
        mlflow.log_artifact(".hydra/overrides.yaml") 
        for epoch in range(cfg.general.epochs):
            log_params_from_omegaconf_dict(cfg)
            train_loss = model.train()
            val_loss, res, feature_fig, h_mean, h_std = model.val()
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
            mlflow.log_table(res, artifact_file='val_result_epoch_{}.csv'.format(epoch))
            mlflow.log_figure(feature_fig, artifact_file='feature_epoch_{}.png'.format(epoch))
            mlflow.pytorch.log_model(model.model, "model_epoch_{}".format(epoch))
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model = model.model
                best_h_mean = h_mean
                best_h_std = h_std
                mlflow.pytorch.log_model(model.model, "best_model")
                mlflow.log_metric('best_val_loss', best_val_loss, step=epoch)
                mlflow.log_table(res, artifact_file='best_val_result.csv'.format(epoch))
                mlflow.log_figure(feature_fig, artifact_file='best_feature.png'.format(epoch))

        # Prediction of HS01
        dataset = AudioPredictionDataset(
            source_dir = '/home/okamoto/HDD4TB/Cicadasong_Detect/records/HS01',
            win_sec = model.c.dataset.win_sec,
            stride_sec = model.c.dataset.win_sec / 2,
            sr = 16000
        )

        model.model = best_model
        preds = model.predict(dataset, best_h_mean, best_h_std, outlier_detection=model.c.model.use_contrastive_loss)

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
                if (m == "f1") and (c != "higurashi") and (c != "kumazemi"):
                    f1s.append(value)
        f1_mean = sum(f1s) / len(f1s)
        mlflow.log_metric('HS01_f1_mean', f1_mean)
    return best_val_loss

if __name__ == '__main__':
    train()