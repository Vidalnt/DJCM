import os
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

df_info = pd.read_csv(r'D:\ICASSP_2024\SVSDT\dataset\INFO\mir1k.csv')
hop_length = 160
path_in = r'D:\联合模型\Data\MIR-1K\Wavfile'
path_label_in = r'D:\联合模型\Data\MIR-1K\PitchLabel'
path_out = r'D:\ICASSP_2024\SVSDT\dataset\MIR1K'

for _, row in tqdm(df_info.iterrows()):
    filename, _, split = row.iloc[0], row.iloc[1], row.iloc[2]
    
    audio_m, sr = librosa.load(os.path.join(path_in, filename), sr=16000, mono=True)

    os.makedirs(os.path.join(path_out, split), exist_ok=True)
    sf.write(os.path.join(path_out, split, filename.replace('.wav', '_m.wav')), 
             audio_m.T, sr, 'PCM_24')

    pv_in = os.path.join(path_label_in, filename.replace('.wav', '.pv'))
    pv_out = os.path.join(path_out, split, filename.replace('.wav', '.pv'))
    f0 = np.loadtxt(pv_in)
    old_times = 0.020 + np.arange(len(f0)) * 0.02
    new_times = np.arange(0.020, old_times[-1] + 0.01, 0.01)
    f0_interp = np.interp(new_times, old_times, f0)
    f0_interp[np.isnan(f0_interp)] = 0.0
    np.savetxt(pv_out, f0_interp, fmt="%.6f")