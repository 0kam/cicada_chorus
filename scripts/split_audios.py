import torchaudio
from glob import glob
import torchaudio
import os
from scipy.signal import resample
import math
import torch
from tqdm import tqdm

def split_audio(in_path, out_dir, length=45, normalize=False, sr_to = 16000):
    wav, sr = torchaudio.load(in_path)
    # ステレオであればモノラルに変換
    if wav.shape[0] == 2:
        wav = wav.mean(axis=0, keepdim=True)
    # 正規化
    wav = wav.numpy()[0]
    wav = resample(wav, math.floor(
        wav.shape[0] / sr * sr_to
    ))
    wav = torch.tensor(wav).unsqueeze(0)
    if normalize:
        wav = wav / wav.abs().max() * 0.8
    # 16bitに変換
    wav = (wav * 32767).to(torch.int16)
    # out_dirがなければ作成
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 30秒ごとに分割してout_dir以下に本の名前＋連番で保存
    w = length * sr_to
    n = wav.shape[1] // w + 1
    for i in range(n):
        out = out_dir + f"/{in_path.split('/')[-1].replace('.wav', '')}_{i}.wav"
        torchaudio.save(out, wav[:, i*w:(i+1)*w], sr_to)

in_dirs = glob("/home/okamoto/cicada_chorus/data/source/background/*")

for in_dir in in_dirs:
    print(in_dir)
    files = glob(f"{in_dir}/*.wav")
    out_dir = in_dir.replace("source", "tuning")
    for file in tqdm(files):
        split_audio(file, out_dir)


in_dirs = glob("/home/okamoto/cicada_chorus/data/source/others/*")

for in_dir in in_dirs:
    files = glob(f"{in_dir}/*.wav")
    out_dir = in_dir.replace("source", "tuning")
    for file in files:
        split_audio(file, out_dir, normalize=True)

in_dirs = glob("/home/okamoto/cicada_chorus/data/cicada_song/segments_preprocessed/*")

for in_dir in in_dirs:
    files = glob(f"{in_dir}/*.wav")
    out_dir = in_dir.replace("cicada_song/segments_preprocessed", "tuning/cicada_song")
    for file in files:
        split_audio(file, out_dir, normalize=False)