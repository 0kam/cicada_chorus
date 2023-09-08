from glob import glob
import torch
import torchaudio
from torch.utils.data import IterableDataset
from torchaudio import transforms
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from pathlib import Path
import math

class ContrastiveAudioDataset(IterableDataset):
    def __init__(self, source_dir, label_dir, win_sec, stride_sec, label_names, audio_sec, shuffle_pairs = True, sr = 16000):
        """A Dataset class for multi-labeled audio contrastive learning tasks. This modules assumes that all audio sources have same length.
        Attributes:
            source_dir (str): Source directory.
            label_dir (str): Label directory.
            win_sec (float): Window size in seconds.
            stride_sec (float): Stride size in seconds.
            label_names (list): List of label names.
            audio_sec (float): Audio length in seconds. Since all audio sources are assumed to have same length, all audio sources will be cut at this length.
            shuffle_pairs (bool): Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
            sr (int): Sampling rate.
        """
        self.win_sec = win_sec
        self.sr = sr
        self.stride_sec = stride_sec
        # self.overlap_rate = overlap_rate
        self.label_names = label_names
        self.audio_length =  math.floor(audio_sec * sr)
        self.source_files = sorted(glob(source_dir+"/*.wav"))
        self.label_files = sorted(glob(label_dir+"/*.txt"))
        self.labels_dict = {}
        for i in range(len(label_names)):
            label = label_names[i]
            self.labels_dict[label] = i + 1 # 0 for no label
        self.shuffle_pairs = shuffle_pairs
        self._create_pairs()
    
    def _create_pairs(self):
        """
        Create two lists of indices that can be used to create pairs of audios.
        """
        self.indices1 = []
        self.indices2 = []
        for i in range(2):
            indices = list(range(len(self.source_files)))
            indices = random.sample(indices, len(indices))
            if len(indices) % 2 == 1:
                indices = indices[:-1]
            self.indices1 = self.indices1 + indices[:len(indices)//2]
            self.indices2 = self.indices2 + indices[len(indices)//2:]
        self.indices1 = torch.tensor(self.indices1)
        self.indices2 = torch.tensor(self.indices2)
        assert sum(self.indices1 == self.indices2) == 0

    def _load_sample(self, index):
        """
        Given an index, load the corresponding audio and label.
        """
        source_file = self.source_files[index]
        label_file = self.label_files[index]
        if Path(source_file).stem != Path(label_file).stem:
            raise ValueError("wav name and csv name don't match! {} {}".format(source_file, label_file))
        a, sr = torchaudio.load(source_file)
        a = a[0]
        if a.shape[0] < self.audio_length:
            raise ValueError("Audio length is shorter than audio_sec. {} {}, Try shorter value as audio_sec.".format(a.shape[0], self.audio_length))
        if sr != self.sr:
            raise ValueError("Sampling rate is different from sr. {} {}".format(sr, self.sr))
        
        a = a[:self.audio_length]

        label_df = pd.read_table(label_file, header=None, names=["start", "end", "label", "distance"])
        label_df["start"] = (label_df["start"] * self.sr).astype(int)
        label_df["end"] = (label_df["end"] * self.sr).astype(int)
        label = np.zeros((len(a), len(self.label_names)))
        for _, s, e, c, _ in label_df.itertuples():
            if c in self.label_names:
                c = self.labels_dict[c]
                label[s:e, c-1] = 1

        w = int(self.win_sec * self.sr)
        stride = int(self.stride_sec * self.sr)
        steps = ((a.shape[0] - w) // stride) - 1

        audios = []
        labels = []
        
        for step in range(steps):
            start = step * stride
            end = start + w
            assert(a[start:end].shape[0] == w)
            labels.append(torch.tensor(label[start:end]))
            audios.append(a[start:end])

        labels = torch.stack(labels)
        labels = labels.max(dim=1)[0]
        audios = torch.stack(audios)
        return audios, labels
    
    def __iter__(self):
        """
        Create a generator that iterates over the dataset.
        """
        if self.shuffle_pairs:
            self._create_pairs()

        for idx1, idx2 in zip(self.indices1, self.indices2):
            audio1, label1 = self._load_sample(idx1)
            audio2, label2 = self._load_sample(idx2)
            
            label_matches = (label1 == label2).to(torch.float32)
            label_matches[(label1.sum(dim=1) == 0)] = 0 # if no label, then label_matches = 0
            yield (audio1, audio2), label_matches, (label1, label2)

    def __len__(self):
        return len(self.indices1)