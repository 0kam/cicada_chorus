from data import AudioSegmentationDataset, AudioPredictionDataset
from utils import Transpose, Unsqueeze, Squeeze

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.transforms import Resize

from torchaudio import transforms
from torchaudio.functional import highpass_biquad, lowpass_biquad
from torch_audiomentations import Gain, AddColoredNoise, Compose, PitchShift, Shift, PeakNormalization

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from tqdm import tqdm
import pandas as pd
from omegaconf import DictConfig

class PretrainedCNN(nn.Module):
    def __init__(self, model_name, y_dims, n_layers, h_dims, batch_norm, drop_out, pretrained = True, freeze = False):
        super().__init__()
        self.model = getattr(models, model_name)(pretrained=pretrained)
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
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
    
    def forward(self, x):
        h = self.cnn1(x)
        h = self.model(h)
        return self.fc(h)
        #return F.sigmoid(self.fc(h))

class PretrainedCNNClassifier:
    def __init__(self, cfg: DictConfig):
        print("Getting pretrained model...")
        self.c = cfg
        # Setting input size of each CNN
        if self.c.model.model_name == "efficientnet_b0":
            self.input_size = (224, 224)
        elif self.c.model.model_name == "efficientnet_b1":
            self.input_size = (240, 240)
        elif self.c.model.model_name == "efficientnet_b2":
            self.input_size = (260, 260)
        elif self.c.model.model_name == "efficientnet_b3":
            self.input_size = (300, 300)
        elif self.c.model.model_name == "efficientnet_b4":
            self.input_size = (380, 380)
        elif self.c.model.model_name == "efficientnet_b5":
            self.input_size = (456, 456)
        elif self.c.model.model_name == "efficientnet_b6":
            self.input_size = (528, 528)
        elif self.c.model.model_name == "efficientnet_b7":
            self.input_size = (600, 600)
        elif self.c.model.model_name == "efficientnet_v2_s":
            self.input_size = (384, 384)
        elif self.c.model.model_name == "efficientnet_v2_m":
            self.input_size = (480, 480)
        elif self.c.model.model_name == "efficientnet_v2_l":
            self.input_size = (480, 480)
        elif "resnet" in self.c.model.model_name:
            self.input_size = (224, 224)
        else:
            raise ValueError("Invalid model name!")
        print("constructing dataloaders...")
        # Constructing Dataloaders
        self.train_ds = AudioSegmentationDataset(
            source_dir = self.c.general.source_dir,
            label_dir = self.c.general.label_dir,
            label_names = self.c.dataset.label_names,
            win_sec = self.c.dataset.win_sec,
            stride_sec = self.c.dataset.stride_sec,
            sr = self.c.dataset.sr
        )
        self.train_loader = DataLoader(self.train_ds, batch_size=self.c.general.batch_size, \
            num_workers=self.c.general.num_workers, shuffle=True)
        
        self.val_ds = AudioSegmentationDataset(
            source_dir = self.c.general.val_source_dir,
            label_dir = self.c.general.val_label_dir,
            label_names = self.c.dataset.label_names,
            win_sec = self.c.dataset.win_sec,
            sr = self.c.dataset.sr,
            stride_sec = self.c.dataset.stride_sec,
        )
        self.val_loader = DataLoader(self.val_ds, batch_size=self.c.general.batch_size, \
            num_workers=self.c.general.num_workers, shuffle=False)

        self.build_transforms("training")

        # Constructing models
        print("constructing models...")
        self.model = PretrainedCNN(self.c.model.model_name, len(self.c.dataset.label_names), self.c.model.n_layers, \
            self.c.model.h_dims, self.c.model.batch_norm, self.c.model.drop_out, self.c.model.freeze_base)
        self.model.to(self.c.general.device)
        if self.c.model.loss == "mse":
            self.criterion = nn.MSELoss()
        elif self.c.model.loss == "bce":
            self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.model.learning_rate)
    
    def train(self):
        self.model.train()
        running_loss = 0.0
        for audio, distances in tqdm(self.train_loader):
            x = audio.reshape(-1, 1, audio.shape[2])
            #x = self.transforms_train(x)
            if self.c.feature.highpass_cutoff is not None:
                x = highpass_biquad(x, self.c.dataset.sr, self.c.feature.highpass_cutoff)
            if self.c.feature.lowpass_cutoff is not None:
                x = lowpass_biquad(x, self.c.dataset.sr, self.c.feature.lowpass_cutoff)
            x = self.transforms_base(x.to(self.c.general.device))
            y = distances.max(dim=-2).values
            y[y > 0] = 1
            y = y.to(self.c.general.device).reshape(-1, len(self.c.dataset.label_names)).float()
            
            self.optimizer.zero_grad()
            y2 = self.model(x)
            loss = self.criterion(y2, y)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        running_loss = running_loss / len(self.train_loader.dataset)
        return running_loss
    
    def val(self):
        self.model.eval()
        running_loss = 0.0
        y_trues = []
        y_preds = []
        for audio, distances in tqdm(self.val_loader):
            x = audio.reshape(-1, 1, audio.shape[2])
            if self.c.feature.highpass_cutoff is not None:
                x = highpass_biquad(x, self.c.dataset.sr, self.c.feature.highpass_cutoff)
            if self.c.feature.lowpass_cutoff is not None:
                x = lowpass_biquad(x, self.c.dataset.sr, self.c.feature.lowpass_cutoff)
            x = self.transforms_base(x.to(self.c.general.device))
            y = distances.max(dim=-2).values
            y[y > 0] = 1
            y = y.to(self.c.general.device).reshape(-1, len(self.c.dataset.label_names)).float()

            with torch.no_grad():
                y2 = self.model(x)
            loss = self.criterion(y2, y.float())
            running_loss += loss.item()
            y_trues.append(y.detach().cpu())
            y_preds.append(y2.detach().cpu())

        running_loss = running_loss / len(self.val_loader.dataset)
        res = self.classification_scores(torch.cat(y_trues), torch.cat(y_preds))
        return running_loss, res
    
    def predict(self, dataset: AudioPredictionDataset):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, \
            num_workers=self.c.general.num_workers, shuffle=False)
        predictions = []
        for audio in tqdm(loader):
            x = audio.reshape(-1, 1, audio.shape[2])
            if self.c.feature.highpass_cutoff is not None:
                x = highpass_biquad(x, self.c.dataset.sr, self.c.feature.highpass_cutoff)
            if self.c.feature.lowpass_cutoff is not None:
                x = lowpass_biquad(x, self.c.dataset.sr, self.c.feature.lowpass_cutoff)
            x = self.transforms_base(x.to(self.c.general.device))
        
            with torch.no_grad():
                y = self.model(x)
            
            predictions.append(y.detach().cpu())

        return torch.stack(predictions).detach().cpu()

    def build_transforms(self, mode):
        if self.c.feature.feature == "mfcc":
            self.transforms_base = nn.Sequential(
                transforms.MFCC(self.c.dataset.sr, self.c.n_mfcc+1),
                Transpose(1,2)
            ).to(self.c.general.device)
        elif self.c.feature.feature == "spectrogram":
            self.transforms_base = nn.Sequential(
                Squeeze(),
                transforms.Spectrogram(n_fft=self.c.feature.n_fft, normalized=False),
                transforms.AmplitudeToDB(),
                Transpose(1,2),
                Resize(self.input_size),
                Unsqueeze(1)
            ).to(self.c.general.device)
        elif self.c.feature.feature == "melspectrogram":
            self.transforms_base = nn.Sequential(
                Squeeze(),
                transforms.MelSpectrogram(sample_rate=self.c.dataset.sr, n_fft=self.c.feature.n_fft, normalized=False, n_mels=self.c.feature.n_mels),
                transforms.AmplitudeToDB(),
                Transpose(1,2),
                Resize(self.input_size),
                Unsqueeze(1)
            ).to(self.c.general.device)
        if mode == "training":
            self.transforms_train = Compose([
                Gain(p=self.c.augumentation.gain[0], \
                    min_gain_in_db=self.c.augumentation.gain[1], \
                    max_gain_in_db=self.c.augumentation.gain[2], \
                    mode = "per_example"),
                PitchShift(p = self.c.augumentation.pitch_shift[0],\
                    sample_rate=self.c.dataset.sr, \
                    min_transpose_semitones=self.c.augumentation.pitch_shift[1], \
                    max_transpose_semitones=self.c.augumentation.pitch_shift[2], \
                    mode = "per_batch"),
                AddColoredNoise(p = self.c.augumentation.colored_noise[0], \
                    min_snr_in_db=self.c.augumentation.colored_noise[1], \
                    max_snr_in_db=self.c.augumentation.colored_noise[2], \
                    mode = "per_example"),
                transforms.TimeMasking(p = self.c.augumentation.time_masking[0], \
                    time_mask_param = int(self.c.dataset.sr*self.c.dataset.win_sec*self.c.augumentation.time_masking[1])),
                PeakNormalization(p = self.c.augumentation.peak_norm, sample_rate=self.c.dataset.sr),
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
        for i, l in enumerate(self.c.dataset.label_names):
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