from data import AudioSegmentationDataset
from utils.utils import Transpose, Unsqueeze, Squeeze

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.transforms import Resize
from tensorboardX import SummaryWriter

from torchaudio import transforms
from torchaudio.functional import highpass_biquad, lowpass_biquad
from torch_audiomentations import Gain, AddBackgroundNoise, AddColoredNoise, Compose, PitchShift, Shift, PeakNormalization, HighPassFilter, LowPassFilter

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score

from dataclasses import dataclass
import yaml
from tqdm import tqdm
from glob import glob
from pathlib import Path
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

@dataclass(init=True)
class Config:
    # General information for model training
    source_dir:  str
    label_dir: str
    model_name: str
    val_ratio: float
    batch_size: int
    num_workers: int
    device: str
    # Information for model testing
    test_source_dir: str
    test_label_dir: str
    # Dataset parameters
    label_names: list
    sr: int
    win_size: float
    overlap_rate: float
    # Hyperparameters of the model
    model_name: str
    feature: str # "spectrogram", or "melspectrogram"
    threshold: float
    sr: int
    highpass_cutoff: int
    lowpass_cutoff: int
    # Configurations of STFT
    n_fft: int #512
    # Configurations of data augumentations
    gain: list
    pitch_shift: list
    background_noise: list
    colored_noise: list
    time_masking: list
    peak_norm: float
    # Hyperparameters of the model
    n_layers: int
    h_dims: int
    batch_norm: bool
    drop_out: float
    learning_rate: float
    freeze_base: bool

class PretrainedCNN(nn.Module):
    def __init__(self, model_name, y_dims, n_layers, h_dims, batch_norm, drop_out, freeze = False):
        super().__init__()
        self.model = getattr(models, model_name)(pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        if "resnet" in model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, h_dims)
        elif "efficientnet" in model_name:
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, h_dims)
        if n_layers == 1:
            self.fc = nn.Linear(h_dims, y_dims)
        else:
            self.fc = [nn.Linear(h_dims, h_dims)]
            for i in range(n_layers-1):
                self.fc.append(nn.Linear(h_dims, h_dims))
                self.fc.append(nn.GELU())
                if batch_norm:
                    self.fc.append(nn.BatchNorm1d(h_dims))
                self.fc.append(nn.Dropout(drop_out))
            self.fc.append(nn.Linear(h_dims, y_dims))
            self.fc = nn.Sequential(*self.fc)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
    
    def forward(self, x):
        h = self.cnn1(x)
        h = self.model(h)
        return self.fc(h)

class PretrainedCNNClassifier:
    def __init__(self, config_path):
        print("Getting pretrained model...")
        # Loading configuration file
        self.config_path = config_path
        with open(config_path) as file:
            self.c = Config(**yaml.safe_load(file))
        # Setting input size of each CNN
        if self.c.model_name == "efficientnet_b0":
            self.input_size = (224, 224)
        elif self.c.model_name == "efficientnet_b1":
            self.input_size = (240, 240)
        elif self.c.model_name == "efficientnet_b2":
            self.input_size = (260, 260)
        elif self.c.model_name == "efficientnet_b3":
            self.input_size = (300, 300)
        elif self.c.model_name == "efficientnet_b4":
            self.input_size = (380, 380)
        elif self.c.model_name == "efficientnet_b5":
            self.input_size = (456, 456)
        elif self.c.model_name == "efficientnet_b6":
            self.input_size = (528, 528)
        elif self.c.model_name == "efficientnet_b7":
            self.input_size = (600, 600)
        elif self.c.model_name == "efficientnet_v2_s":
            self.input_size = (384, 384)
        elif self.c.model_name == "efficientnet_v2_m":
            self.input_size = (480, 480)
        elif self.c.model_name == "efficientnet_v2_l":
            self.input_size = (480, 480)
        elif "resnet" in self.c.model_name:
            self.input_size = (224, 224)
        else:
            raise ValueError("Invalid model name!")
        print("constructing dataloaders...")
        # Constructing Dataloaders
        self.ds = AudioSegmentationDataset(
            source_dir = self.c.source_dir,
            label_dir = self.c.label_dir,
            label_names = self.c.label_names,
            win_size = self.c.win_size,
            sr = self.c.sr,
            overlap_rate = self.c.overlap_rate,
        )
        train_index, val_index = train_test_split(range(len(self.ds)), test_size=self.c.val_ratio, shuffle=True)
        self.train_loader = DataLoader(Subset(self.ds, train_index), batch_size=self.c.batch_size, \
            num_workers=self.c.num_workers, shuffle=True)
        self.val_loader = DataLoader(Subset(self.ds, val_index), batch_size=self.c.batch_size, \
            num_workers=self.c.num_workers, shuffle=False)
        
        self.build_transforms("training")

        # Constructing models
        print("constructing models...")
        self.model = PretrainedCNN(self.c.model_name, len(self.c.label_names), self.c.n_layers, \
            self.c.h_dims, self.c.batch_norm, self.c.drop_out, self.c.freeze_base)
        self.model.to(self.c.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.learning_rate)
    
    def load_config(self, config_path):
        cp = Path(config_path)
        @hydra.main(config_path=str(cp.parent), config_name=cp.stem, version_base=None)
        def _load_config(cfg: DictConfig):
            self.c2 = cfg
        _load_config()
    
    def _train(self, epoch):
        self.model.train()
        running_loss = 0.0
        for audio, distances in tqdm(self.train_loader):
            x = audio.reshape(-1, 1, audio.shape[2])
            #x = self.transforms_train(x)
            x = self.transforms_base(x.to(self.c.device))
            y = distances.max(dim=-2).values
            y[y > 0] = 1
            y = y.to(self.c.device).reshape(-1, len(self.c.label_names)).float()
            
            self.optimizer.zero_grad()
            y2 = self.model(x)
            loss = self.criterion(y2, y)
            running_loss += loss

            loss.backward()
            self.optimizer.step()
        running_loss = running_loss / len(self.train_loader.dataset)
        print("Epoch {} train_loss: {:.4f}".format(epoch, running_loss))
        return running_loss
    
    def _val(self, epoch):
        self.model.eval()
        running_loss = 0.0
        y_trues = []
        y_preds = []
        for audio, distances in tqdm(self.val_loader):
            x = audio.reshape(-1, 1, audio.shape[2])
            x = self.transforms_base(x.to(self.c.device))
            y = distances.max(dim=-2).values
            y[y > 0] = 1
            y = y.to(self.c.device).reshape(-1, len(self.c.label_names)).to(torch.long)

            with torch.no_grad():
                y2 = self.model(x)
            loss = self.criterion(y2, y)
            running_loss += loss
            y_trues.append(y.detach().cpu())
            y_preds.append(y2.detach().cpu())

        running_loss = running_loss / len(self.val_loader.dataset)
        res = self.classification_scores(torch.cat(y_trues), torch.cat(y_preds))
        print("Epoch {} val_loss: {:.4f}".format(epoch, running_loss))
        print("Epoch {} classification scores".format(epoch))
        print("--------------------------------------------------------------")
        print(res)
        print("--------------------------------------------------------------")
        return running_loss, res
    
    def test(self):
        test_ds = AudioSegmentationDataset(
            source_dir = self.c.test_source_dir,
            label_dir = self.c.test_label_dir,
            label_names = self.c.label_names,
            win_size = self.c.win_size,
            sr = self.c.sr,
            overlap_rate = self.c.overlap_rate,
        )
        self.test_loader = DataLoader(test_ds, batch_size=self.c.batch_size, \
            num_workers=self.c.num_workers, shuffle=False)
        self.model.eval()
        running_loss = 0.0
        y_trues = []
        y_preds = []
        for audio, distances in tqdm(self.test_loader):
            x = audio.reshape(-1, 1, audio.shape[2])
            x = self.transforms_base(x.to(self.c.device))
            y = distances.max(dim=-2).values
            y[y > 0] = 1
            y = y.to(self.c.device).reshape(-1, len(self.c.label_names)).to(torch.long)

            with torch.no_grad():
                y2 = self.model(x)
            loss = self.criterion(y2, y)
            running_loss += loss
            y_trues.append(y.detach().cpu())
            y_preds.append(y2.detach().cpu())

        running_loss = running_loss / len(self.val_loader.dataset)
        res = self.classification_scores(torch.cat(y_trues), torch.cat(y_preds))
        print("Test_loss: {:.4f}".format(running_loss))
        print("Test classification scores")
        print("--------------------------------------------------------------")
        print(res)
        print("--------------------------------------------------------------")
        return running_loss, res
        
    
    def train(self, epochs):
        for i in range(epochs):
            self._train(i)
            self._val(i)

    def build_transforms(self, mode):
        if self.c.feature == "mfcc":
            self.transforms_base = nn.Sequential(
                transforms.MFCC(self.c.sr, self.c.n_mfcc+1),
                Transpose(1,2)
            ).to(self.c.device)
        elif self.c.feature == "spectrogram":
            self.transforms_base = nn.Sequential(
                Squeeze(),
                transforms.Spectrogram(n_fft=self.c.n_fft, normalized=False),
                transforms.AmplitudeToDB(),
                Transpose(1,2),
                Resize(self.input_size),
                Unsqueeze(1)
            ).to(self.c.device)
        elif self.c.feature == "melspectrogram":
            self.transforms_base = nn.Sequential(
                Squeeze(),
                transforms.MelSpectrogram(sample_rate=self.c.sr, n_fft=self.c.n_fft, normalized=False),
                transforms.AmplitudeToDB(),
                Transpose(1,2),
                Resize(self.input_size),
                Unsqueeze(1)
            ).to(self.c.device)
        if mode == "training":
            self.transforms_train = Compose([
                Gain(p=self.c.gain[0], \
                    min_gain_in_db=self.c.gain[1], \
                    max_gain_in_db=self.c.gain[2], \
                    mode = "per_example"),
                PitchShift(p = self.c.pitch_shift[0],\
                    sample_rate=self.c.sr, \
                    min_transpose_semitones=self.c.pitch_shift[1], \
                    max_transpose_semitones=self.c.pitch_shift[2], \
                    mode = "per_batch"),
                AddColoredNoise(p = self.c.colored_noise[0], \
                    min_snr_in_db=self.c.colored_noise[1], \
                    max_snr_in_db=self.c.colored_noise[2], \
                    mode = "per_example"),
                transforms.TimeMasking(p = self.c.time_masking[0], \
                    time_mask_param = int(self.c.sr*self.c.win_size*self.c.time_masking[1])),
                PeakNormalization(p = self.c.peak_norm, sample_rate=self.c.sr),
                Squeeze()
            ])
    
    def classification_scores(self, y_true, y_pred, threshold=0.5):
        """
        y_pred is the output of the model
        """
        y_pred = (y_pred.detach().cpu() >= threshold).float()
        y_true = y_true.detach().cpu()

        labels = []
        accs = []
        f1s = []
        recalls = []
        precisions = []
        for i, l in enumerate(self.c.label_names):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            labels.append(l)
            accs.append(accuracy_score(yt, yp))
            f1s.append(f1_score(yt, yp, average='binary'))
            recalls.append(recall_score(yt, yp, average='binary'))
            precisions.append(precision_score(yt, yp, average='binary'))

        res = pd.DataFrame({
            "label": labels,
            "accuracy": accs,
            "f1": f1s,
            "recall": recalls,
            "precision": precisions
        })
        return res

self = PretrainedCNNClassifier("old/config.yaml")
self.load_config("cnn/config.yaml")