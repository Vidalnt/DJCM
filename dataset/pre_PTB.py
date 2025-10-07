import os
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

df_info = pd.read_csv(r'D:\ICASSP_2024\SVSDT\dataset\INFO\ptb.csv')
path_in_root = r'D:\Dataset\SPEECH_DATA_ZIPPED\SPEECH DATA'
path_out = r'D:\ICASSP_2024\SVSDT\dataset\PTDB'

for _, row in tqdm(df_info.iterrows()):
    filename, f0, split = row['name'], row['label_path'], row['split']

    audio_p, sr = librosa.load(os.path.join(path_in_root, filename), sr=16000, mono=True)
    out_filename = os.path.splitext(os.path.basename(filename))[0]

    os.makedirs(os.path.join(path_out, split), exist_ok=True)

    sf.write(os.path.join(path_out, split, f"{out_filename}_p.wav"),
             audio_p.T, sr, 'PCM_24')

    f0_hz = np.loadtxt(os.path.join(path_in_root, f0), usecols=(0,))
    f0_hz[f0_hz < 0] = 0.0
    pv_out = os.path.join(path_out, split, f"{out_filename}.pv")
    np.savetxt(pv_out, f0_hz, fmt="%.9f")