from pyroomacoustics import room as pra
from scipy.io import wavfile
import math
from scipy.signal import resample
from glob import glob
from pathlib import Path
import numpy as np
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchaudio import transforms

class ChorusGenerator():
    def __init__(
            self, song_dir, bg_dir,
            audio_sec = 300, sr = 16000,
            distance_range = [20, 30],
            species_range = [1, 3],
            popsize_range = [1, 20]
        ):
        self.audio_sec = audio_sec
        self.sr = sr
        self.song_dir = song_dir
        self.bg_dir = bg_dir
        self.song_files = glob(song_dir + '/**/*.wav')
        self.bg_files = glob(bg_dir + '/*.wav')
        self.distance_range = distance_range
        self.species_range = species_range
        self.popsize_range = popsize_range
        self.species = [Path(p).name for p in  glob(song_dir + "/*")]
    
    # bg_filesからランダムに選んだ音声に、song_filesからランダムに選んだ音声を挿入する
    # song_filesはbgのランダムな位置に挿入される。
    def generate(self, wav_path, label_path):
        # シミュレーション環境を設定
        room = pra.AnechoicRoom(
            dim=2,
            fs=16000,
            sigma2_awgn=10,
            temperature=30,
            humidity=0.7,
            air_absorption=True
        )
        room.add_microphone(loc=[0, 0])
        
        bg_file = np.random.choice(self.bg_files)
        # self.speciesからmax_sp以下のランダムな種を選ぶ    
        sp = np.random.choice(self.species, np.random.randint(self.species_range[0], self.species_range[1] + 1)).tolist()
        # spの中からmax_pop以下のランダムな個数を選ぶ
        pop = np.random.choice(sp, np.random.randint(self.popsize_range[0], self.popsize_range[1] + 1)).tolist()
        song_files = [np.random.choice(glob(f"{self.song_dir}/{p}/*.wav")) for p in pop]
        sr, bg = wavfile.read(bg_file)
        # bgをaudio_sec分にランダムな位置で切り出す。audio_secより短い場合は、audio_sec分になるまでbgを繰り返す。
        bg = bg[np.random.randint(0, len(bg) - sr * self.audio_sec):][:sr * self.audio_sec]
        bg = resample(bg, math.floor(
            bg.shape[0] / sr * self.sr
        ))
        room.add_source([0,1], bg)

        # song_filesを読み込み、self.srにリサンプリング
        sps = []
        starts = []
        distances = []
        ends = []
        for song_file in song_files:
            print(song_file)
            sps.append(Path(song_file).parent.name)
            song = np.zeros(bg.shape)
            sr, s = wavfile.read(song_file)
            s = resample(s, math.floor(
                s.shape[0] / sr * self.sr
            ))
            # self.distance_rangeの範囲でランダムに音源からマイクまでの距離を決める
            dist = np.random.randint(self.distance_range[0], self.distance_range[1])
            print("Distance: {} m".format(dist))
            distances.append(dist)
            # songのランダムな位置にsを埋め込む
            start = np.random.randint(0, len(song) - len(s))
            end = start + len(s)
            starts.append(start)
            ends.append(end)
            song[start:end] += s
            room.add_source([0,dist], song)
        premix = room.simulate(return_premix=True)
        if not os.path.exists(Path(wav_path).parent):
            os.makedirs(Path(wav_path).parent)
        room.mic_array.to_wav(
            wav_path,
            bitdepth=np.int16,
        )
        # Save label file
        if not os.path.exists(Path(label_path).parent):
            os.makedirs(Path(label_path).parent)
        if os.path.exists(label_path):
            os.remove(label_path)
        for sp, start,end, distance in zip(sps, starts, ends, distances):
            s = start / self.sr
            e = end / self.sr
            with open(label_path, mode='a') as f:
                f.write("{}\t{}\t{}\t{}\n".format(s, e, sp, distance))
        
        df = pd.DataFrame({
            "species": sps,
            "start": starts,
            "end": ends,
            "distance": distances
        })

class AudioSegmentationDataset(Dataset):
    def __init__(self, source_dir, label_dir, win_size, label_names, sr = 16000, overlap_rate=0.5):
        """A Dataset class for multi-labeled audio segmentation tasks.
        Attributes:
            source_dir (str): Source directory.
            win_size (float): Window size in sec.
            label_names (list of str): Label names.
            overlap_rate (float): Overlap rate of each neighbour windows.
            normalize (boolean): Normalize?
        """
        self.win_size = win_size
        self.sr = sr
        self.label_names = label_names
        source_files = sorted(glob(source_dir+"/*.wav"))
        label_files = sorted(glob(label_dir+"/*.txt"))
        self.labels_dict = {}
        for i in range(len(label_names)):
            label = label_names[i]
            self.labels_dict[label] = i + 1 # 0 for no label
        
        print("Measuring dataset size......")
        idx = []
        source_files_list = []
        label_files_list = []
        starts = []
        ends = []
        # For data summarization
        labels = []
        max_distances = []
        
        i = 0
        for source_file, label_file in tqdm(zip(source_files, label_files), total=len(source_files)):
            if Path(source_file).stem != Path(label_file).stem:
                raise ValueError("wav name and csv name don't match! {} {}".format(source_file, label_file))
            a, sr_from = torchaudio.load(source_file)
            a = transforms.Resample(sr_from, sr)(a)[0] # サンプリングレートを揃えて1チャンネル目だけを取る

            label = pd.read_table(label_file, header=None, names=["start", "end", "label", "distance"])
            label["start"] = (label["start"] * sr).astype(int)
            label["end"] = (label["end"] * sr).astype(int)
            distances = np.zeros((len(a), len(label_names)))
            for _, s, e, c, d in label.itertuples():
                if c in label_names:
                    c = self.labels_dict[c]
                    distances[s:e, c-1] = d

            w = int(win_size*sr)
            steps = a.shape[0] // int(w*(1-overlap_rate)) - 1
            for step in range(steps):
                start = int(step * w * (1-overlap_rate))
                end = start + w
                assert(a[start:end].shape[0] == w)
                idx.append(i)
                i += 1
                source_files_list.append(source_file)
                label_files_list.append(label_file)
                starts.append(start)
                ends.append(end)
                distance = distances[start:end]
                for l in label_names:
                    d = distance[:, self.labels_dict[l]-1]
                    if d.max() > 0:
                        labels.append(l)
                        max_distances.append(d.max())
                    else:
                        labels.append("no_label")
                        max_distances.append(0)               
        
        self.ds_df = pd.DataFrame({
            "idx": idx,
            "source_file": source_files_list,
            "label_file": label_files_list,
            "start": starts,
            "end": ends
        })

        self.summary_df = pd.DataFrame({
            "label": labels,
            "max_distance": max_distances
        })

        print("Dataset size: {}".format(len(self.ds_df)))
        print("Dataset summary: ")
        print(self.summary_df.groupby("label").agg(["count", "mean", "max", "min", "std"]))

    def __len__(self):
        return len(self.summary_df)
    
    def __getitem__(self, idx):
        source_file = self.ds_df.iloc[idx]["source_file"]
        label_file = self.ds_df.iloc[idx]["label_file"]
        start = self.ds_df.iloc[idx]["start"]
        end = self.ds_df.iloc[idx]["end"]
        if Path(source_file).stem != Path(label_file).stem:
            raise ValueError("wav name and csv name don't match! {} {}".format(source_file, label_file))
        a, sr_from = torchaudio.load(source_file)
        a = transforms.Resample(sr_from, self.sr)(a)[0] # サンプリングレートを揃えて1チャンネル目だけを取る
        label = pd.read_table(label_file, header=None, names=["start", "end", "label", "distance"])
        label["start"] = (label["start"] * self.sr).astype(int)
        label["end"] = (label["end"] * self.sr).astype(int)
        distances = np.zeros((len(a), len(self.label_names)))
        for _, s, e, c, d in label.itertuples():
            if c in self.label_names:
                c = self.labels_dict[c]
                distances[s:e, c-1] = d
        audio = a[start:end]
        distance = distances[start:end]
        return audio, distance

ds = AudioSegmentationDataset(
    source_dir = "data/train/wav",
    label_dir = "data/train/label",
    label_names = ["aburazemi", "higurashi", "kumazemi", "minminzemi", "niiniizemi", "tsukutsukuboushi"],
    win_size = 0.96,
    sr = 16000,
    overlap_rate = 0.5
)

d = iter(ds)
next(d)