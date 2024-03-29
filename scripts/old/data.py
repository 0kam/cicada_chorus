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
from torch.utils.data import Dataset
from torchaudio import transforms
from audiomentations import Compose, TimeStretch, PitchShift
from tqdm import tqdm
import random

class ChorusGenerator():
    def __init__(
            self, song_dir, bg_dir,
            audio_sec = 300, sr = 16000,
            distance_range = [20, 30],
            species_range = [1, 3],
            popsize_range = [1, 20],
            time_stretch_range = [0.8, 1.2],
            pitch_shift_range = [-1, 1]
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
        self.audiomentation = Compose([
            TimeStretch(min_rate=time_stretch_range[0], max_rate=time_stretch_range[1], p=0.5),
            PitchShift(min_semitones=pitch_shift_range[0], max_semitones=pitch_shift_range[1], p=0.5)
        ])
    
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
        sr, bg = wavfile.read(bg_file)
        # bgをaudio_sec分にランダムな位置で切り出す。audio_secより短い場合は、audio_sec分になるまでbgを繰り返す。
        bg = bg[np.random.randint(0, len(bg) - sr * self.audio_sec):][:sr * self.audio_sec]
        bg = resample(bg, math.floor(
            bg.shape[0] / sr * self.sr
        )).astype(np.float32)
        room.add_source([0,1], bg)
        # self.speciesからmax_sp以下のランダムな種を選ぶ 
        sp = np.random.choice(self.species, np.random.randint(self.species_range[0], self.species_range[1] + 1), replace=False).tolist()
        # spの中からmax_pop以下のランダムな個数を選ぶ
        if len(sp) == 0:
            if not os.path.exists(Path(wav_path).parent):
                os.makedirs(Path(wav_path).parent)
            premix = room.simulate(return_premix=True)
            room.mic_array.to_wav(
                wav_path,
                bitdepth=np.int16,
            )
            if not os.path.exists(Path(label_path).parent):
                os.makedirs(Path(label_path).parent)
            if os.path.exists(label_path):
                os.remove(label_path)
            #  空のファイルを作成
            with open(label_path, "w") as f:
                f.write("")
        else:
            pop = np.random.choice(sp, np.random.randint(self.popsize_range[0], self.popsize_range[1] + 1)).tolist()
            song_files = [np.random.choice(glob(f"{self.song_dir}/{p}/*.wav")) for p in pop]

            # song_filesを読み込み、self.srにリサンプリング
            sps = []
            starts = []
            distances = []
            ends = []
            for song_file in song_files:
                # print(song_file)
                species = Path(song_file).parent.name
                sps.append(species)
                song = np.zeros(bg.shape)
                sr, s = wavfile.read(song_file)
                s = resample(s, math.floor(
                    s.shape[0] / sr * self.sr
                )).astype(np.float32)
                if len(s.shape) > 1:
                    s = s[:, 0]
                s = self.audiomentation(samples=s, sample_rate=self.sr)
                # self.distance_rangeの範囲でランダムに音源からマイクまでの距離を決める
                dist = np.random.randint(self.distance_range[0], self.distance_range[1])
                # print("Distance: {} m".format(dist))
                distances.append(dist)
                # songのランダムな位置にsを埋め込む
                if len(s) > len(song):
                    start = np.random.randint(0, len(s) - len(song))
                    end = start + len(song)
                    starts.append(0)
                    ends.append(len(song))
                    s = s[start:end]
                    song += s
                else:
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
    def __init__(self, source_dir, label_dir, win_sec, stride_sec, label_names, sr = 16000):
        """A Dataset class for multi-labeled audio segmentation tasks.
        Attributes:
            source_dir (str): Source directory.
            win_sec (float): Window size in sec.
            label_names (list of str): Label names.
            overlap_rate (float): Overlap rate of each neighbour windows.
            normalize (boolean): Normalize?
        """
        self.win_sec = win_sec
        self.sr = sr
        self.stride_sec = stride_sec
        # self.overlap_rate = overlap_rate
        self.label_names = label_names
        self.source_files = sorted(glob(source_dir+"/*.wav"))
        self.label_files = sorted(glob(label_dir+"/*.txt"))
        self.labels_dict = {}
        for i in range(len(label_names)):
            label = label_names[i]
            self.labels_dict[label] = i + 1 # 0 for no label
        
    def __getitem__(self, index):
        source_file = self.source_files[index]
        label_file = self.label_files[index]
        if Path(source_file).stem != Path(label_file).stem:
            raise ValueError("wav name and csv name don't match! {} {}".format(source_file, label_file))
        a, sr_from = torchaudio.load(source_file)
        a = transforms.Resample(sr_from, self.sr)(a)[0] # サンプリングレートを揃えて1チャンネル目だけを取る

        label = pd.read_table(label_file, header=None, names=["start", "end", "label", "distance"])
        label["start"] = (label["start"] * self.sr).astype(int)
        label["end"] = (label["end"] * self.sr).astype(int)
        # distanceが大きい順にソート (同じラベルが重なっている場合、近い方の距離に上書きされる)
        label = label.sort_values("distance", ascending=False)
        distance = np.zeros((len(a), len(self.label_names)))
        for _, s, e, c, d in label.itertuples():
            if c in self.label_names:
                c = self.labels_dict[c]
                distance[s:e, c-1] = d

        w = int(self.win_sec * self.sr)
        stride = int(self.stride_sec * self.sr)
        steps = ((a.shape[0] - w) // stride) - 1

        # audios = torch.as_strided(a, (steps, w), (stride,1))
        # distances = torch.as_strided(
        #     torch.tensor(distance), 
        #     (steps, w, len(self.label_names)), 
        #     (stride,1,1))

        audios = []
        distances = []
        
        for step in range(steps):
            start = step * stride
            end = start + w
            assert(a[start:end].shape[0] == w)
            distances.append(torch.tensor(distance[start:end]))
            audios.append(a[start:end])
        distances = torch.stack(distances)
        audios = torch.stack(audios)
        return audios, distances

    def __len__(self):
        return len(self.source_files)

class AudioPredictionDataset(Dataset):
    def __init__(self, source_dir, win_sec, stride_sec, sr = 16000):
        """A Dataset class for audio prediction.
        Attributes:
            source_dir (str): Source directory.
            win_sec (float): Window size in sec.
            stride_sec (float): Window stride size in sec.
            sr (int): Sampling rate.
        """
        self.win_sec = win_sec
        self.sr = sr
        self.stride_sec = stride_sec
        self.source_files = sorted(glob(source_dir+"/**/*.wav"))
        
    def __getitem__(self, index):
        source_file = self.source_files[index]
        a, sr_from = torchaudio.load(source_file)
        a = transforms.Resample(sr_from, self.sr)(a)[0] # サンプリングレートを揃えて1チャンネル目だけを取る

        w = int(self.win_sec * self.sr)
        stride = int(self.stride_sec * self.sr)
        steps = ((a.shape[0] - w) // stride) - 1

        audios = []
        
        for step in range(steps):
            start = step * stride
            end = start + w
            assert(a[start:end].shape[0] == w)
            audios.append(a[start:end])
        audios = torch.stack(audios)
        return audios

    def __len__(self):
        return len(self.source_files)

class ChorusGenerator2():
    """
    ChorusGenerator module for fast generation of chorus data.
    This module loads all the audio files in the song_dir and bg_dir at the initialization.
    """
    def __init__(
            self, song_dir, bg_dir,
            audio_sec = 300, sr = 16000,
            distance_range = [20, 30],
            species_range = [1, 3],
            popsize_range = [1, 20],
            time_stretch_range = [0.8, 1.2],
            pitch_shift_range = [-1, 1]
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
        self.audiomentation = Compose([
            TimeStretch(min_rate=time_stretch_range[0], max_rate=time_stretch_range[1], p=0.5),
            PitchShift(min_semitones=pitch_shift_range[0], max_semitones=pitch_shift_range[1], p=0.5)
        ])

        # song_filesを読み込み、self.srにリサンプリング
        # song_file の親ディレクトリの名前を種名として取得
        self.songs = []
        self.species = []
        print("Loading song files...")
        for song_file in tqdm(self.song_files):
            sr, s = wavfile.read(song_file)
            s = resample(s, math.floor(
                s.shape[0] / sr * self.sr
            )).astype(np.float32)
            if len(s.shape) > 1:
                s = s[:, 0]
            self.songs.append(s)
            self.species.append(Path(song_file).parent.name)
        
        self.songs = pd.DataFrame({
            "species": self.species,
            "song": self.songs
        })
        
        # bg_filesを読み込み、self.srにリサンプリング
        self.bgs = []
        for bg_file in tqdm(self.bg_files):
            sr, bg = wavfile.read(bg_file)
            bg = resample(bg, math.floor(
                bg.shape[0] / sr * self.sr
            )).astype(np.float32)
            self.bgs.append(bg)
    
    # self.bgsからランダムに選んだ音声に、self.songsからランダムに選んだ音声を挿入する
    # songはbgのランダムな位置に挿入される。
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
        
        bg = random.choice(self.bgs)
        # bgをaudio_sec分にランダムな位置で切り出す。audio_secより短い場合は、audio_sec分になるまでbgを繰り返す。
        bg = bg[np.random.randint(0, len(bg) - self.sr * self.audio_sec):][:self.sr * self.audio_sec]
        room.add_source([0,1], bg)
        # self.speciesからmax_sp以下のランダムな種を選ぶ 
        sp = np.random.choice(self.species, np.random.randint(self.species_range[0], self.species_range[1] + 1), replace=False).tolist()
        # spの中からmax_pop以下のランダムな個数を選ぶ
        if len(sp) == 0:
            if not os.path.exists(Path(wav_path).parent):
                os.makedirs(Path(wav_path).parent)
            premix = room.simulate(return_premix=True)
            room.mic_array.to_wav(
                wav_path,
                bitdepth=np.int16,
            )
            if not os.path.exists(Path(label_path).parent):
                os.makedirs(Path(label_path).parent)
            if os.path.exists(label_path):
                os.remove(label_path)
            #  空のファイルを作成
            with open(label_path, "w") as f:
                f.write("")
        else:
            pop = np.random.choice(sp, np.random.randint(self.popsize_range[0], self.popsize_range[1] + 1)).tolist()
            # self.songsのうち、speciesがpopに含まれるものをランダムに選ぶ
            sps = []
            starts = []
            distances = []
            ends = []
            for p in pop:
                song_data = self.songs[self.songs["species"]==p].sample()
                sps.append(song_data["species"].values[0])
                song = np.zeros(bg.shape)
                s = self.audiomentation(samples=song_data["song"].values[0], sample_rate=self.sr)
                # self.distance_rangeの範囲でランダムに音源からマイクまでの距離を決める
                dist = np.random.randint(self.distance_range[0], self.distance_range[1])
                # print("Distance: {} m".format(dist))
                distances.append(dist)
                # songのランダムな位置にsを埋め込む
                if len(s) > len(song):
                    start = np.random.randint(0, len(s) - len(song))
                    end = start + len(song)
                    starts.append(0)
                    ends.append(len(song))
                    s = s[start:end]
                    song += s
                else:
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


def load_audio(file, sr_to=16000):
    sr, s = wavfile.read(file)
    s = resample(s, math.floor(
        s.shape[0] / sr * sr_to
    )).astype(np.float32)
    if len(s.shape) > 1:
        s = s[:, 0]
    return s

def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class ChorusGenerator3():
    """
    合唱データを自動生成するモジュール。
    ChorusGenerator2と同じで最初にすべての音源を読みこむ。
    音源のカテゴリごとに、用いる頻度や距離を調整する。
    """
    def __init__(self, cicada_dir, bg_dir, others_dir, sr = 16000):
        self.cicada_dir = cicada_dir
        self.bg_dir = bg_dir
        self.others_dir = others_dir
        self.sr = sr

        self.cicadas = pd.DataFrame({
            "file": glob(f"{cicada_dir}/**/*.wav")
        })
        self.cicadas["label"] = self.cicadas.file.apply(lambda x: Path(x).parent.name)
        print("Loading cicada audios...")
        self.cicadas["audio"] = [load_audio(file, self.sr) for file in tqdm(self.cicadas.file)]

        self.others = pd.DataFrame({
            "file": glob(f"{others_dir}/**/*.wav")
        })
        self.others["label"] = self.others.file.apply(lambda x: Path(x).parent.name)
        print("Loading other audios...")
        self.others["audio"] = [load_audio(file, self.sr) for file in tqdm(self.others.file)]

        self.bgs = pd.DataFrame({
            "file": glob(f"{bg_dir}/**/*.wav")
        })
        self.bgs["label"] = self.bgs.file.apply(lambda x: Path(x).parent.name)
        print("Loading bg audios...")
        self.bgs["audio"] = [load_audio(file, self.sr) for file in tqdm(self.bgs.file)]

    def set_params(self, audio_sec = 30, category_ratio = [0.5, 0.3, 0.2],
            # Cicadas
            cicada_weights = {"aburazemi": 1, "higurashi": 1, "kumazemi": 1, "minminzemi": 1, 
                              "niiniizemi": 1, "tsukutsukuboushi": 1, },
            cicada_popsize_range = [1, 20],
            cicada_distance_range = [10, 60],
            cicada_species_range = [1, 4],
            # Backgrounds
            bg_weights = {"city": 1, "real_rec_2022": 1, "real_rec_2023": 1, "roadside": 1, "windy": 1},
            # Other sounds
            others_weights = {"music": 1, "birds": 1, "esc50": 1, "insects": 1, "speech": 1},
            others_popsize_range = [1, 20],
            others_distance_range = [10, 60],
            others_species_range = [1, 4],
            # Audio augmentation
            time_stretch_range = [0.8, 1.2],
            pitch_shift_range = [-1, 1]
        ):
        self.audio_sec = audio_sec
        self.category_ratio = category_ratio

        self.cicada_weights = cicada_weights
        self.bg_weights = bg_weights
        self.others_weights = others_weights

        self.cicada_popsize_range = cicada_popsize_range
        self.cicada_distance_range = cicada_distance_range
        self.cicada_species_range = cicada_species_range
        self.others_popsize_range = others_popsize_range
        self.others_distance_range = others_distance_range
        self.others_species_range = others_species_range
        
        self.audiomentation = Compose([
            TimeStretch(min_rate=time_stretch_range[0], max_rate=time_stretch_range[1], p=0.5),
            PitchShift(min_semitones=pitch_shift_range[0], max_semitones=pitch_shift_range[1], p=0.5)
        ])
    
    # self.bgsからランダムに選んだ音声に、self.songsからランダムに選んだ音声を挿入する
    # songはbgのランダムな位置に挿入される。
    def generate(self, wav_path, label_path):
        """
        Generate chorus data.
        Parameters
        ----------
        wav_path : str
            Path to save the generated audio.
        label_path : str
            Path to save the generated label.
        """
        # シミュレーション環境を設定
        room = pra.AnechoicRoom(
            dim=2,
            fs=16000,
            sigma2_awgn=100,
            temperature=30,
            humidity=70,
            air_absorption=True
        )
        room.add_microphone(loc=[0, 0])
        # 音源のカテゴリーを決める
        category = np.random.choice(["cicada", "bg", "others"], p=self.category_ratio)
        # 背景音を決める
        bg_weights = softmax(np.array(list(self.bg_weights.values())))
        bg_category = np.random.choice(list(self.bg_weights.keys()), p=bg_weights)
        bg = random.choice(self.bgs[self.bgs["label"]==bg_category]["audio"].values)
        # bgをaudio_sec分にランダムな位置で切り出す。audio_secより短い場合は、audio_sec分になるまでbgを繰り返す。
        bg = bg[np.random.randint(0, len(bg) - self.sr * self.audio_sec):][:self.sr * self.audio_sec]
        room.add_source([0,1], bg)

        sps = []
        starts = []
        distances = []
        ends = []

        if category == "bg":
            pass
        else:
            if category == "cicada":
                species_range = self.cicada_species_range
                popsize_range = self.cicada_popsize_range
                distance_range = self.cicada_distance_range
                weights = self.cicada_weights
                df = self.cicadas
            elif category == "others":
                species_range = self.others_species_range
                popsize_range = self.others_popsize_range
                distance_range = self.others_distance_range
                weights = self.others_weights
                df = self.others

            # 種数を決める
            n_species = np.random.randint(species_range[0], species_range[1] + 1)
            # 種を決める
            species = np.random.choice(list(weights.keys()), n_species, p=softmax(np.array(list(weights.values()))), replace=False)
            # 個体数を決める
            popsize = np.random.randint(popsize_range[0], popsize_range[1] + 1)
            # self.cicada_weightsのうち、speciesにふくまれるものだけを絞り込む
            species_weights = {k: v for k, v in weights.items() if k in species}

            for _ in range(popsize):
                sp = np.random.choice(species, 1, p=softmax(np.array(list(species_weights.values()))))
                song_data = df[df["label"]==sp[0]].sample()
                sps.append(song_data["label"].values[0])
                song = np.zeros(bg.shape)
                s = self.audiomentation(samples=song_data["audio"].values[0], sample_rate=self.sr)
                # self.distance_rangeの範囲でランダムに音源からマイクまでの距離を決める
                dist = np.random.randint(distance_range[0], distance_range[1])
                # print("Distance: {} m".format(dist))
                distances.append(dist)
                # songのランダムな位置にsを埋め込む
                if len(s) > len(song):
                    start = np.random.randint(0, len(s) - len(song))
                    end = start + len(song)
                    starts.append(0)
                    ends.append(len(song))
                    s = s[start:end]
                    song += s
                else:
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
        
        if category == "bg":
            with open(label_path, "w") as f:
                f.write("")


def generate(self, wav_path, label_path):
    """
    Generate chorus data.
    Parameters
    ----------
    wav_path : str
        Path to save the generated audio.
    label_path : str
        Path to save the generated label.
    """
    # シミュレーション環境を設定
    room = pra.AnechoicRoom(
        dim=2,
        fs=16000,
        sigma2_awgn=100,
        temperature=30,
        humidity=70,
        air_absorption=True
    )
    room.add_microphone(loc=[0, 0])
    # 音源のカテゴリーを決める
    category = np.random.choice(["cicada", "bg", "others"], p=self.category_ratio)
    # 背景音を決める
    bg_weights = softmax(np.array(list(self.bg_weights.values())))
    bg_category = np.random.choice(list(self.bg_weights.keys()), p=bg_weights)
    bg = random.choice(self.bgs[self.bgs["label"]==bg_category]["audio"].values)
    # bgをaudio_sec分にランダムな位置で切り出す。audio_secより短い場合は、audio_sec分になるまでbgを繰り返す。
    bg = bg[np.random.randint(0, len(bg) - self.sr * self.audio_sec):][:self.sr * self.audio_sec]
    room.add_source([0,1], bg)

    sps = []
    starts = []
    distances = []
    ends = []

    if category == "bg":
        pass
    else:
        if category == "cicada":
            species_range = self.cicada_species_range
            popsize_range = self.cicada_popsize_range
            distance_range = self.cicada_distance_range
            weights = self.cicada_weights
            df = self.cicadas
        elif category == "others":
            species_range = self.others_species_range
            popsize_range = self.others_popsize_range
            distance_range = self.others_distance_range
            weights = self.others_weights
            df = self.others

        # 種数を決める
        n_species = np.random.randint(species_range[0], species_range[1] + 1)
        # 種を決める
        species = np.random.choice(list(weights.keys()), n_species, p=softmax(np.array(list(weights.values()))), replace=False)
        # 個体数を決める
        popsize = np.random.randint(popsize_range[0], popsize_range[1] + 1)
        # self.cicada_weightsのうち、speciesにふくまれるものだけを絞り込む
        species_weights = {k: v for k, v in weights.items() if k in species}

        for _ in range(popsize):
            sp = np.random.choice(species, 1, p=softmax(np.array(list(species_weights.values()))))
            song_data = df[df["label"]==sp[0]].sample()
            sps.append(song_data["label"].values[0])
            song = np.zeros(bg.shape)
            s = self.audiomentation(samples=song_data["audio"].values[0], sample_rate=self.sr)
            # self.distance_rangeの範囲でランダムに音源からマイクまでの距離を決める
            dist = np.random.randint(distance_range[0], distance_range[1])
            # print("Distance: {} m".format(dist))
            distances.append(dist)
            # songのランダムな位置にsを埋め込む
            if len(s) > len(song):
                start = np.random.randint(0, len(s) - len(song))
                end = start + len(song)
                starts.append(0)
                ends.append(len(song))
                s = s[start:end]
                song += s
            else:
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
    
    if category == "bg":
        with open(label_path, "w") as f:
            f.write("")