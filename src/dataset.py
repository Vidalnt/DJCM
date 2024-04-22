import os
import librosa
import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob
from .constants import *


class MIR1K(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None):
        self.path = path
        self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
        self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
        self.num_class = N_CLASS
        self.data = []

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.extend(self.load(*input_files))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def availabe_groups():
        return ['test']

    def files(self, group):
        audio_m_files = glob(os.path.join(self.path, group, '*_m.wav'))
        audio_v_files = [f.replace('_m.wav', '_v.wav') for f in audio_m_files]
        label_files = [f.replace('_m.wav', '.pv') for f in audio_m_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_v_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_m_files, audio_v_files, label_files))

    def load(self, audio_m_path, audio_v_path, label_path):
        data = []
        audio_m, _ = librosa.load(audio_m_path, sr=SAMPLE_RATE)
        if audio_m.ndim == 1:
            audio_m = np.array([audio_m])
        audio_m = torch.from_numpy(audio_m)

        audio_v, _ = librosa.load(audio_v_path, sr=SAMPLE_RATE)
        if audio_v.ndim == 1:
            audio_v = np.array([audio_v])
        audio_v = torch.from_numpy(audio_v)

        audio_l = audio_m.shape[-1]
        audio_steps = audio_l // self.HOP_LENGTH + 1

        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                i += 1
                if float(line) != 0:
                    freq = 440 * (2.0 ** ((float(line) - 69.0) / 12.0))
                    cent = 1200 * np.log2(freq/10)
                    index = int(round((cent-CONST)/20))
                    pitch_label[i][index] = 1
                    voice_label[i] = 1

        if self.seq_len is not None:
            n_steps = self.seq_len // self.HOP_LENGTH
            for i in range(audio_l // self.seq_len):
                begin_t = i * self.seq_len
                end_t = begin_t + self.seq_len
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                data.append(dict(audio_m=audio_m[:, begin_t:end_t], audio_v=audio_v[:, begin_t:end_t],
                                 pitch=pitch_label[begin_step:end_step], voice=voice_label[begin_step:end_step],
                                 file=os.path.basename(audio_m_path)))
            data.append(dict(audio_m=audio_m[:, -self.seq_len:], audio_v=audio_v[:, -self.seq_len:],
                             pitch=pitch_label[-n_steps:], voice=voice_label[-n_steps:],
                             file=os.path.basename(audio_m_path)))
        else:
            data.append(dict(audio_m=audio_m, audio_v=audio_v, pitch=pitch_label, voice=voice_label,
                             file=os.path.basename(audio_m_path)))
        return data
