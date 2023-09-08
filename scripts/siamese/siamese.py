import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.transforms import Resize

from utils.contrastive import ContrastiveAudioDataset
from utils.data import AudioPredictionDataset
from utils.utils import Transpose, Unsqueeze, Squeeze

from torchaudio import transforms
from torchaudio.functional import highpass_biquad, lowpass_biquad
from torch_audiomentations import Gain, AddColoredNoise, Compose, PitchShift, Shift, PeakNormalization
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.manifold import TSNE

from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import math
from omegaconf import DictConfig

class MultilabelSiameseCNN(nn.Module):
    def __init__(self, backbone, y_dims, h_dims, pretrained = True):
        super().__init__()
        self.y_dims = y_dims
        self.h_dims = h_dims
        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))
        
        # Create a backbone network from the pretrained models provided in torchvision.models 
        self.backbone = models.__dict__[backbone](pretrained=pretrained, progress=True)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features + self.y_dims

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        # Since this model implies that the data has multiple labels, the output of the MLP is a vector of size y_dims.

        self.labelwise_feature_extractor = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Conv1d(out_features, 512, 1, stride=1),
            nn.BatchNorm1d(512),
            nn.SiLU(),

            #nn.Dropout(p=0.5),
            nn.Conv1d(512, 256, 1, stride=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            #nn.Dropout(p=0.5),
            nn.Conv1d(256, 256, 1, stride=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            
            nn.Conv1d(256, self.h_dims, 1, stride=1)
        )

        self.cls_head = nn.Sequential(
            nn.Conv1d(self.h_dims, 32, 1, stride=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 1, 1, stride=1),
            nn.Sigmoid()
        )

        # Create a CNN to convert the 3 channel image to 1 channel image.
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.SiLU()
        )

        self.backbone = nn.Sequential(self.cnn1, self.backbone)

    def forward(self, x):
        '''
        Returns the extracted features and label predictions.

            Parameters:
                    x (torch.Tensor): shape=[b, 1, 224, 224] where b = batch size

            Returns:
                    h (torch.Tensor): shape=[b, y_dims, h_dims], Extracted features of each image.
                    y (torch.Tensor): shape=[b, y_dims], Predicted labels.
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        h = self.backbone(x).unsqueeze(1).repeat(1,self.y_dims,1)
        # self.y_dimsの次元のone-hotラベルを作る
        labels = torch.eye(self.y_dims, requires_grad=True).to(x.device).repeat(h.shape[0], 1, 1)
        h = torch.cat([h, labels], dim=-1).transpose(1,2)
        h = self.labelwise_feature_extractor(h).transpose(1,2) # shape=[n_batch, y_dims, 64]
        y = self.cls_head(h.transpose(1,2)).squeeze()        
        return h, y 

class SiameseCNNModel:
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
        self.train_ds = ContrastiveAudioDataset(
            source_dir = self.c.general.source_dir,
            label_dir = self.c.general.label_dir,
            win_sec = self.c.dataset.win_sec,
            stride_sec = self.c.dataset.stride_sec,
            label_names = self.c.dataset.label_names,
            audio_sec=self.c.dataset.audio_sec,
            shuffle_pairs=True,
            sr=self.c.dataset.sr
        )
        
        self.train_loader = DataLoader(self.train_ds, batch_size=self.c.general.batch_size, \
            num_workers=self.c.general.num_workers)
        
        self.val_ds = ContrastiveAudioDataset(
            source_dir = self.c.general.val_source_dir,
            label_dir = self.c.general.val_label_dir,
            win_sec = self.c.dataset.win_sec,
            stride_sec = self.c.dataset.stride_sec,
            label_names = self.c.dataset.label_names,
            audio_sec=self.c.dataset.audio_sec,
            shuffle_pairs=False,
            sr=self.c.dataset.sr
        )

        self.val_loader = DataLoader(self.val_ds, batch_size=self.c.general.batch_size, \
            num_workers=self.c.general.num_workers)
        self.y_dims = len(self.c.dataset.label_names)

        self.build_transforms("training")

        # Constructing models
        print("constructing models...")
        self.model = MultilabelSiameseCNN(backbone=self.c.model.model_name, y_dims=self.y_dims, h_dims = self.c.model.h_dims, pretrained=self.c.model.pretrained)
        self.model.to(self.c.general.device)
        # Loss function for classifier
        if self.c.model.classifier_loss == "mse":
            self.criterion = nn.MSELoss()
        elif self.c.model.classifier_loss == "bce":
            self.criterion = nn.BCELoss()
        # Distance metric for siamese network
        if self.c.model.distance_metric == "l1":
            self.distance_metric = nn.L1Loss(reduction="none")
        elif self.c.model.distance_metric == "l2":
            self.distance_metric = nn.MSELoss(reduction="none")
        elif self.c.model.distance_metric == "cosine":
            self.distance_metric = nn.CosineSimilarity(reduction="none")
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.model.learning_rate)
    
    def loss(self, matches, h1, h2, y1_true, y2_true, y1_pred, y2_pred):
        """
        matches: shape=[n_batch, y_dims]
        h1: shape=[n_batch, y_dims, 64]
        y1_true: shape=[n_batch, y_dims]
        y2_true: shape=[n_batch, y_dims]
        y1_pred: shape=[n_batch, y_dims]
        y2_pred: shape=[n_batch, y_dims]
        """
        y1_loss = self.criterion(y1_pred.to(torch.double), y1_true.to(torch.double))
        y2_loss = self.criterion(y2_pred.to(torch.double), y2_true.to(torch.double))
        
        distance = self.distance_metric(h1, h2).mean(-1) # shape=[n_batch, y_dims, h_dims]
        m = matches.reshape(-1, self.y_dims).to(self.c.general.device)

        # contrastive loss
        # m is the label of whether the pair is the same or not (if same, 1, else 0)
        contrastive_loss = (m) * torch.pow(distance, 2) + (1 - m) * torch.pow(torch.clamp(self.c.model.margin - distance, min=0.0), 2) # shape=[n_batch, y_dims]
        contrastive_loss = contrastive_loss.mean() # shape=[n_batch]

        if self.c.model.use_contrastive_loss:
            loss = y1_loss + y2_loss + contrastive_loss
        else:
            loss = y1_loss + y2_loss
        return loss

    def train(self):
        self.model.train()
        running_loss = 0.0
        for audios, matches, labels in tqdm(self.train_loader):
            x1 = audios[0].reshape(-1, 1, audios[0].shape[2])
            x2 = audios[1].reshape(-1, 1, audios[1].shape[2])
            if self.c.feature.highpass_cutoff is not None:
                x1 = highpass_biquad(x1, self.c.dataset.sr, self.c.feature.highpass_cutoff)
                x2 = highpass_biquad(x2, self.c.dataset.sr, self.c.feature.highpass_cutoff)
            if self.c.feature.lowpass_cutoff is not None:
                x1 = lowpass_biquad(x1, self.c.dataset.sr, self.c.feature.lowpass_cutoff)
                x1 = lowpass_biquad(x1, self.c.dataset.sr, self.c.feature.lowpass_cutoff)

            x1 = self.transforms_base(x1.to(self.c.general.device))
            x2 = self.transforms_base(x2.to(self.c.general.device))
            
            y1_true = labels[0].reshape(-1, self.y_dims).to(self.c.general.device)
            y2_true = labels[1].reshape(-1, self.y_dims).to(self.c.general.device)
            m = matches.reshape(-1, self.y_dims).to(self.c.general.device).float()
            
            self.optimizer.zero_grad()

            h1, y1_pred = self.model(x1)
            h2, y2_pred = self.model(x2)

            loss = self.loss(m, h1, h2, y1_true, y2_true, y1_pred, y2_pred)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        running_loss = running_loss / len(self.train_loader)
        return running_loss
    
    def val(self):
        self.model.eval()
        running_loss = 0.0
        y_trues = []
        y_preds = []
        hs = []
        for audios, matches, labels in tqdm(self.val_loader):
            x1 = audios[0].reshape(-1, 1, audios[0].shape[2])
            x2 = audios[1].reshape(-1, 1, audios[1].shape[2])
            if self.c.feature.highpass_cutoff is not None:
                x1 = highpass_biquad(x1, self.c.dataset.sr, self.c.feature.highpass_cutoff)
                x2 = highpass_biquad(x2, self.c.dataset.sr, self.c.feature.highpass_cutoff)
            if self.c.feature.lowpass_cutoff is not None:
                x1 = lowpass_biquad(x1, self.c.dataset.sr, self.c.feature.lowpass_cutoff)
                x1 = lowpass_biquad(x1, self.c.dataset.sr, self.c.feature.lowpass_cutoff)

            x1 = self.transforms_base(x1.to(self.c.general.device))
            x2 = self.transforms_base(x2.to(self.c.general.device))
            
            y1_true = labels[0].reshape(-1, self.y_dims).to(self.c.general.device)
            y2_true = labels[1].reshape(-1, self.y_dims).to(self.c.general.device)
            m = matches.reshape(-1, self.y_dims).to(self.c.general.device).float()
            
            with torch.no_grad():
                h1, y1_pred = self.model(x1)
                h2, y2_pred = self.model(x2)
            
            loss = self.loss(m, h1, h2, y1_true, y2_true, y1_pred, y2_pred)

            y_trues.append(y1_true.detach().cpu())
            y_trues.append(y2_true.detach().cpu())
            y_preds.append(y1_pred.detach().cpu())
            y_preds.append(y2_pred.detach().cpu())
            hs.append(h1.detach().cpu())
            hs.append(h2.detach().cpu())

            running_loss += loss.item()

        running_loss = running_loss / len(self.val_loader)
        y_trues = torch.cat(y_trues)
        y_preds = torch.cat(y_preds)
        hs = torch.cat(hs)

        res = self.classification_scores(y_trues, y_preds)
        fig = self.draw_features(hs, y_trues)

        # ラベルごとの平均と標準偏差を計算
        h_mean = hs.mean(axis=0)
        h_std = hs.std(axis=0)

        return running_loss, res, fig, h_mean, h_std
    
    def draw_features(self, hs, y_trues, n_sample = 1000):
        """
        hs: shape=[n_batch, y_dims, h_dims]
        y_trues: shape=[n_batch, y_dims]
        y_preds: shape=[n_batch, y_dims]
        """
        idx = np.random.choice(len(hs), n_sample)
        hs = hs[idx]
        y_trues = y_trues[idx]

        hs = hs.transpose(0,1)
        y_trues = y_trues.transpose(0,1)

        hs_reduced = []
        for h in hs:
            hs_reduced.append(TSNE(n_components=2, random_state=0).fit_transform(h))
        hs_reduced = np.stack(hs_reduced)

        # Plot hs for each label.
        ncol = math.floor(self.y_dims**0.5)
        nrow = math.ceil(self.y_dims/ncol)
        fig = plt.figure(figsize=(ncol*5, nrow*5))
        plt.rcParams["font.size"] = 20
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for i, l in enumerate(self.c.dataset.label_names):
            ax = plt.subplot(nrow, ncol, i+1)
            ax.set_title(l)
            ax.scatter(hs_reduced[i, :, 0], hs_reduced[i, :, 1], c=y_trues[i].detach().numpy(), cmap="jet")
        return fig

    def predict(self, dataset: AudioPredictionDataset, h_mean, h_std, outlier_detection=True):
        """
        Predict the labels of the dataset.
        Args:
            dataset: AudioPredictionDataset
            h_mean: shape=[y_dims, h_dims] used for outlier detection
            h_std: shape=[y_dims, h_dims] used for outlier detection
        Returns:
        """
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
                h, y = self.model(x)
                # outlier detection
            h = h.detach().cpu()
            y = y.detach().cpu()
            if outlier_detection:
                outliers = (torch.abs((h - h_mean) / h_std) > 3).max(axis=-1).values # outlier threshold = 3 sigma
                y[outliers] = 0

            predictions.append(y)

        return torch.stack(predictions)

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