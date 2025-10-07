import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob
from .constants import *


class MIR1K(Dataset):
    """
    MIR-1K Dataset for VPE (Vocal Pitch Estimation) only
    Only loads mixture audio and pitch labels - no separated vocal audio
    """
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
    def available_groups():
        return ['train', 'test']

    def files(self, group):
        """
        Get audio mixture and label files (no vocal separation files needed)
        """
        audio_mir1k_files = glob(os.path.join(self.path, group, '*_m.wav'))
        audio_ptdb_files = glob(os.path.join(self.path, group, '*_p.wav'))
        all_audio_files = sorted(audio_mir1k_files + audio_ptdb_files)

        label_files = []
        for f in all_audio_files:
            if f.endswith('_m.wav'):
                label_files.append(f.replace('_m.wav', '.pv'))
            elif f.endswith('_p.wav'):
                label_files.append(f.replace('_p.wav', '.pv'))
        
        # Verify label files exist
        assert all(os.path.isfile(label_file) for label_file in label_files), \
            "Some label files are missing"
        
        return zip(all_audio_files, label_files)

    def load(self, audio_path, label_path):
        """
        Load mixture audio and pitch labels only
        
        Args:
            audio_path: Path to mixture audio file
            label_path: Path to pitch label file (.pv)
            
        Returns:
            data: List of dictionaries with audio_m, pitch, voice, and file
        """
        data = []
        
        # Load mixture audio
        audio_m, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        if audio_m.ndim == 1:
            audio_m = np.array([audio_m])
        audio_m = torch.from_numpy(audio_m)
        
        audio_l = audio_m.shape[-1]
        audio_steps = audio_l // self.HOP_LENGTH + 1
        
        # Initialize pitch and voice labels
        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)
        
        # Load pitch annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if float(line) != 0:
                    if audio_path.endswith('_m.wav'):
                        freq = 440 * (2.0 ** ((float(line) - 69.0) / 12.0))
                    elif audio_path.endswith('_p.wav'):
                        freq = float(line)
                    # Convert frequency to cents
                    cent = 1200 * np.log2(freq / 10)
                    # Quantize to pitch bin
                    index = int(round((cent - CONST) / 20))
                    
                    # Ensure index is within valid range
                    if 0 <= index < self.num_class and i < audio_steps:
                        pitch_label[i][index] = 1
                        voice_label[i] = 1
        
        # Split into sequences if sequence_length is specified
        if self.seq_len is not None:
            n_steps = (self.seq_len // self.HOP_LENGTH) + 1
            
            # Create non-overlapping sequences
            for i in range(audio_l // self.seq_len):
                begin_t = i * self.seq_len
                end_t = begin_t + self.seq_len
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                
                data.append(dict(
                    audio_m=audio_m[:, begin_t:end_t],
                    pitch=pitch_label[begin_step:end_step],
                    voice=voice_label[begin_step:end_step],
                    file=os.path.basename(audio_path)
                ))
            
            # Add last sequence (may overlap with previous)
            data.append(dict(
                audio_m=audio_m[:, -self.seq_len:],
                pitch=pitch_label[-n_steps:],
                voice=voice_label[-n_steps:],
                file=os.path.basename(audio_path)
            ))
        else:
            # Full-length sample
            data.append(dict(
                audio_m=audio_m,
                pitch=pitch_label,
                voice=voice_label,
                file=os.path.basename(audio_path)
            ))
        
        return data