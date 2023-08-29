from glob import glob
import torchaudio
import os
from torchaudio.functional import highpass_biquad, lowpass_biquad
from pathlib import Path
import torch

def preprocess_segments(path, out):
    wav, sr = torchaudio.load(path)
    # ステレオであればモノラルに変換
    if wav.shape[0] == 2:
        wav = wav.mean(axis=0, keepdim=True)
    # ハイパスをかける
    out_dir = Path(out).parent
    sp = out_dir.parts[-1]
    if sp != 'other_sounds':
        wav = highpass_biquad(wav, sr, cutoff_freq=500)
    # 正規化
    wav = wav / wav.abs().max() * 0.8
    # 16bitに変換
    wav = (wav * 32767).to(torch.int16)
    out_dir = Path(out).parent
    if out_dir.exists() == False:
        os.makedirs(out_dir)
    torchaudio.save(out, wav, sr)

files = glob('/home/okamoto/cicada_chorus/data/cicada_song/segments/*/*')
out_files = [f.replace('segments', 'segments_preprocessed') for f in files]
out_files = [f.replace(Path(f).suffix, '.wav') for f in out_files]

for f, out in zip(files, out_files):
    preprocess_segments(f, out)
