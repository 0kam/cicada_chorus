import numpy as np
import torch
import torchaudio.transforms as ta_trans
from torch_audioset.params import CommonParams, YAMNetParams
from torchvision.transforms import Resize
# This program is based on https://github.com/w-hc/torch_audioset/blob/master/torch_audioset/data/torch_input_processing.py

class YAMNestFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        audio_sample_rate = CommonParams.TARGET_SAMPLE_RATE
        window_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_WINDOW_LENGTH_SECONDS
        ))
        hop_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_HOP_LENGTH_SECONDS
        ))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        assert window_length_samples == 400
        assert hop_length_samples == 160
        assert fft_length == 512
        self.mel_trans_ope = VGGishLogMelSpectrogram(
            CommonParams.TARGET_SAMPLE_RATE, n_fft=fft_length,
            win_length=window_length_samples, hop_length=hop_length_samples,
            f_min=CommonParams.MEL_MIN_HZ,
            f_max=CommonParams.MEL_MAX_HZ,
            n_mels=CommonParams.NUM_MEL_BANDS
        )
        self.resize = Resize((96, 64))
        # note that the STFT filtering logic is exactly the same as that of a
        # conv kernel. It is the center of the kernel, not the left edge of the
        # kernel that is aligned at the start of the signal.

    def __call__(self, waveform):
        '''
        Args:
            waveform: torch tensor [win_sec * sample_rate]
        Returns:
            torch tensor of shape [N, C, T]
        '''
        x = self.mel_trans_ope(waveform)
        x = x.transpose(-1,-2)  # # [1, C, T] -> [T, C]
        x = self.resize(x)
        
        return x

class VGGishLogMelSpectrogram(ta_trans.MelSpectrogram):
    '''
    This is a _log_ mel-spectrogram transform that adheres to the transform
    used by Google's vggish model input processing pipeline
    '''

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time)
        """
        specgram = self.spectrogram(waveform)
        # NOTE at mel_features.py:98, googlers used np.abs on fft output and
        # as a result, the output is just the norm of spectrogram raised to power 1
        # For torchaudio.MelSpectrogram, however, the default
        # power for its spectrogram is 2.0. Hence we need to sqrt it.
        # I can change the power arg at the constructor level, but I don't
        # want to make the code too dirty
        specgram = specgram ** 0.5

        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + CommonParams.LOG_OFFSET)
        return mel_specgram