from pyroomacoustics import room as pra
from scipy.io import wavfile
import math
from scipy.signal import resample
from glob import glob
from pathlib import Path
import numpy as np
import os
import pandas as pd
from audiomentations import Compose, TimeStretch, PitchShift
import scipy
import warnings

def load_audio(file, sr_to=16000):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sr, s = wavfile.read(file)
    if s.max() < 1:
        s = s * 2**16
    s = resample(s, math.floor(
        s.shape[0] / sr * sr_to
    )).astype(np.float32)
    if len(s.shape) > 1:
        s = s[:, 0]
    return s

def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def get_audio_datasets(cicada_dir, bg_dir, others_dir):
    cicadas = pd.DataFrame({
        "file": glob(f"{cicada_dir}/**/*.wav")
    })
    cicadas["label"] = cicadas.file.apply(lambda x: Path(x).parent.name)
    #print("Loading cicada audios...")
    #cicadas["audio"] = [load_audio(file, sr) for file in tqdm(cicadas.file)]

    others = pd.DataFrame({
        "file": glob(f"{others_dir}/**/*.wav")
    })
    others["label"] = others.file.apply(lambda x: Path(x).parent.name)
    #print("Loading other audios...")
    #others["audio"] = [load_audio(file, sr) for file in tqdm(others.file)]

    bgs = pd.DataFrame({
        "file": glob(f"{bg_dir}/**/*.wav")
    })
    bgs["label"] = bgs.file.apply(lambda x: Path(x).parent.name)
    #print("Loading bg audios...")
    #bgs["audio"] = [load_audio(file, sr) for file in tqdm(bgs.file)]
    return cicadas, bgs, others

def generate(wav_path, label_path, cicadas, bgs, others, sr = 16000, audio_sec = 30, category_ratio = [0.5, 0.3, 0.2],
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
    """
    Generate chorus data.
    Parameters
    ----------
    wav_path : str
        Path to save the generated audio.
    label_path : str
        Path to save the generated label.
    """
    np.random.seed()
    # オーディオオーグメンテーションを設定
    audiomentation = Compose([
        TimeStretch(min_rate=time_stretch_range[0], max_rate=time_stretch_range[1], p=0.5),
        PitchShift(min_semitones=pitch_shift_range[0], max_semitones=pitch_shift_range[1], p=0.5)
    ])
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
    category = np.random.choice(["cicada", "bg", "others"], p=category_ratio)
    # 背景音を決める
    bg_category = np.random.choice(list(bg_weights.keys()), p=softmax(np.array(list(bg_weights.values()))))
    #bg = random.choice(bgs[bgs["label"]==bg_category]["audio"].values)
    bg = load_audio(bgs[bgs["label"]==bg_category].sample()["file"].values[0])
    # bgをaudio_sec分にランダムな位置で切り出す。audio_secより短い場合は、audio_sec分になるまでbgを繰り返す。
    if len(bg) != sr * audio_sec:
        bg = bg[np.random.randint(0, len(bg) - sr * audio_sec):][:sr * audio_sec]
    room.add_source([0,1], bg)

    sps = []
    starts = []
    distances = []
    ends = []

    if category == "bg":
        pass
    else:
        if category == "cicada":
            species_range = cicada_species_range
            popsize_range = cicada_popsize_range
            distance_range = cicada_distance_range
            weights = cicada_weights
            df = cicadas
        elif category == "others":
            species_range = others_species_range
            popsize_range = others_popsize_range
            distance_range = others_distance_range
            weights = others_weights
            df = others

        # 種数を決める
        n_species = np.random.randint(species_range[0], species_range[1] + 1)
        # 種を決める
        species = np.random.choice(list(weights.keys()), n_species, p=softmax(np.array(list(weights.values()))), replace=False)
        # 個体数を決める
        popsize = np.random.randint(popsize_range[0], popsize_range[1] + 1)
        # cicada_weightsのうち、speciesにふくまれるものだけを絞り込む
        species_weights = {k: v for k, v in weights.items() if k in species}

        for _ in range(popsize):
            sp = np.random.choice(species, 1, p=softmax(np.array(list(species_weights.values()))))
            song_data = df[df["label"]==sp[0]].sample()
            sps.append(song_data["label"].values[0])
            song = np.zeros(bg.shape)
            s = load_audio(song_data["file"].values[0])
            # s = audiomentation(samples=s, sample_rate=sr)
            # distance_rangeの範囲でランダムに音源からマイクまでの距離を決める
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
            elif len(s) == len(song):
                starts.append(0)
                ends.append(len(song))
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
        s = start / sr
        e = end / sr
        with open(label_path, mode='a') as f:
            f.write("{}\t{}\t{}\t{}\n".format(s, e, sp, distance))
    
    if category == "bg":
        with open(label_path, "w") as f:
            f.write("")