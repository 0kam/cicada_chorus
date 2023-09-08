from utils.data import ChorusGenerator2
from tqdm import tqdm

# Generate train data
cg_train = ChorusGenerator2(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/train',
    bg_dir = '/home/okamoto/cicada_chorus/data/uncicada',
    audio_sec = 30,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 4],
    popsize_range = [2, 60]
)

for i in tqdm(range(2000)):
    cg_train.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/train_small/wav/{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/train_small/label/{i}.txt'
    )

# Generate test data
cg_test = ChorusGenerator2(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/test',
    bg_dir = '/home/okamoto/cicada_chorus/data/uncicada',
    audio_sec = 30,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 4],
    popsize_range = [2, 60]
)

for i in tqdm(range(1000)):
    cg_test.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/test_small/wav/{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/test_small/label/{i}.txt'
    )

# Generate others data
cg_others_train = ChorusGenerator2(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/segments_preprocessed/others',
    bg_dir = '/home/okamoto/cicada_chorus/data/uncicada',
    audio_sec = 30,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 30]
)

for i in tqdm(range(1000)):
    cg_others_train.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/train_small/wav/others_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/train_small/label/others_{i}.txt'
    )

cg_others_test = ChorusGenerator2(
    song_dir = '/home/okamoto/cicada_chorus/data/cicada_song/segments_preprocessed/others',
    bg_dir = '/home/okamoto/cicada_chorus/data/uncicada',
    audio_sec = 30,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 30]
)

for i in tqdm(range(500)):
    cg_others_test.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/test_small/wav/others_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/test_small/label/others_{i}.txt'
    )

cg_esc_train = ChorusGenerator2(
    song_dir = '/home/okamoto/HDD4TB/ESC-50/',
    bg_dir = '/home/okamoto/cicada_chorus/data/uncicada',
    audio_sec = 30,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 60]
)

for i in tqdm(range(1000)):
    cg_esc_train.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/train_small/wav/esc50_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/train_small/label/esc50_{i}.txt'
    )

cg_esc_test = ChorusGenerator2(
    song_dir = '/home/okamoto/HDD4TB/ESC-50/',
    bg_dir = '/home/okamoto/cicada_chorus/data/uncicada',
    audio_sec = 30,
    sr = 16000,
    distance_range = [20, 60],
    species_range = [1, 1],
    popsize_range = [1, 60]
)

for i in tqdm(range(500)):
    cg_esc_test.generate(
        wav_path = f'/home/okamoto/cicada_chorus/data/test_small/wav/esc50_{i}.wav',
        label_path = f'/home/okamoto/cicada_chorus/data/test_small/label/esc50_{i}.txt'
    )