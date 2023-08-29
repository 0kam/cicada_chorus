from data import ChorusGenerator
from tqdm import tqdm

# Generate train data
cg_train = ChorusGenerator(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/train',
    bg_dir = '/home/okamoto/cicada_chorus/data/background',
    audio_sec = 10,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 4],
    popsize_range = [2, 10]
)

for i in tqdm(range(10000)):
    cg_train.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/train/wav/{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/train/label/{i}.txt'
    )

# Generate test data
cg_test = ChorusGenerator(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/test',
    bg_dir = '/home/okamoto/cicada_chorus/data/background',
    audio_sec = 10,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 4],
    popsize_range = [2, 10]
)

for i in tqdm(range(2000)):
    cg_test.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/test/wav/{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/test/label/{i}.txt'
    )

# Generate others data
cg_others_train = ChorusGenerator(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/segments_preprocessed/others',
    bg_dir = '/home/okamoto/cicada_chorus/data/background',
    audio_sec = 10,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 10]
)

for i in tqdm(range(5000)):
    cg_others_train.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/train/wav/others_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/train/label/others_{i}.txt'
    )

cg_others_test = ChorusGenerator(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/segments_preprocessed/others',
    bg_dir = '/home/okamoto/cicada_chorus/data/background',
    audio_sec = 10,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 10]
)

for i in tqdm(range(1000)):
    cg_others_test.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/test/wav/others_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/test/label/others_{i}.txt'
    )

cg_esc_train = ChorusGenerator(
    song_dir = '/home/okamoto/HDD4TB/ESC-50/',
    bg_dir = '/home/okamoto/cicada_chorus/data/background',
    audio_sec = 10,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 15]
)

for i in tqdm(range(5000)):
    cg_esc_train.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/train/wav/esc50_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/train/label/esc50_{i}.txt'
    )

cg_esc_test = ChorusGenerator(
    song_dir = '/home/okamoto/HDD4TB/ESC-50/',
    bg_dir = '/home/okamoto/cicada_chorus/data/background',
    audio_sec = 10,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 15]
)

for i in tqdm(range(1000)):
    cg_esc_test.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/test/wav/esc50_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/test/label/esc50_{i}.txt'
    )