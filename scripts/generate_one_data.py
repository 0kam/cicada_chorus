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

def generate_one_chorus(out_dir, cicadas, bg, others, sr = 16000, audio_sec = 30, distance_range = [10, 30],
            time_stretch_range = [0.8, 1.2], # Audio augmentation
            pitch_shift_range = [-1, 1]
        ):
    """
    Generate one chorus data.
    Parameters
    ----------
    wav_path : str
        Path to save the generated audio.
    cicadas : list of str
        List of cicada audio paths.
    bg : str
        Path to background audio.
    others : list of str
        List of other audio paths.
    sr : int
        Sampling rate.
    audio_sec : int
        Length of the generated audio in seconds.
    distance_range : list of int
        Distance range.
    time_stretch_range : list of float
        Time stretch range.
    pitch_shift_range : list of float
        Pitch shift range.
    """
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
    bg = load_audio(bg)
    # bgをaudio_sec分にランダムな位置で切り出す。audio_secより短い場合は、audio_sec分になるまでbgを繰り返す。
    if len(bg) != sr * audio_sec:
        bg = bg[np.random.randint(0, len(bg) - sr * audio_sec):][:sr * audio_sec]
    room.add_source([0,1], bg)
    song_paths = cicadas + others
    for song_path in song_paths:
        song = np.zeros(bg.shape)
        s = load_audio(song_path)
        # s = audiomentation(samples=s, sample_rate=sr)
        # distance_rangeの範囲でランダムに音源からマイクまでの距離を決める
        dist = np.random.randint(distance_range[0], distance_range[1])
        # print("Distance: {} m".format(dist))
        # songのランダムな位置にsを埋め込む
        if len(s) > len(song):
            start = np.random.randint(0, len(s) - len(song))
            end = start + len(song)
            s = s[start:end]
            song += s
        elif len(s) == len(song):
            song += s
        else:
            start = np.random.randint(0, len(song) - len(s))
            end = start + len(s)
            song[start:end] += s
        room.add_source([0,dist], song)
    
    premix = room.simulate(return_premix=True)
    # 混ぜ合わせる前の音源を保存
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        os.system("rm -rf {}".format(out_dir))
        os.makedirs(out_dir)
    for i, src in enumerate(room.sources):
        signal = src.signal
        if signal.max() < 1:
            signal = signal * 2**16
        if i == 0:
            wavfile.write(os.path.join(out_dir, "bg.wav"), sr, signal.astype(np.int16))
        else:
            out_name = Path(song_paths[i-1]).stem
            wavfile.write(os.path.join(out_dir, f"{out_name}.wav"), sr, signal.astype(np.int16))
    room.mic_array.to_wav(
        os.path.join(out_dir, 'mixed.wav'),
        bitdepth=np.int16,
    )


out_dir = '/home/okamoto/cicada_chorus/data/examples/0'
cicadas = [
    '/home/okamoto/cicada_chorus/data/cicada_song/train/aburazemi/アブラゼミaudiostock_81685_01.wav',
    '/home/okamoto/cicada_chorus/data/cicada_song/train/minminzemi/ミンミンゼミaudiostock_81707_01.wav',
    '/home/okamoto/cicada_chorus/data/cicada_song/train/higurashi/ヒグラシaudiostock_81702_01.wav',
]
others = []
bg = '/home/okamoto/cicada_chorus/data/background/real_rec_2023/NE01_biohazard_20230610_1150.wav'

generate_one_chorus(out_dir, cicadas, bg, others)