import os
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

df_info = pd.read_csv(r'D:\ICASSP_2024\SVSDT\dataset\INFO\vocadito.csv')
sr = 16000
path_in = r'D:\Dataset\Vocadito'
path_out = r'D:\ICASSP_2024\SVSDT\dataset\Vocadito'

for _, row in tqdm(df_info.iterrows(), total=len(df_info)):
    filename, label_path, split = row['name'], row['label_path'], row['split']
   
    audio_m, _ = librosa.load(os.path.join(path_in, filename), sr=sr, mono=True)
   
    os.makedirs(os.path.join(path_out, split), exist_ok=True)
   
    out_filename = os.path.splitext(os.path.basename(filename))[0]
    sf.write(os.path.join(path_out, split, f"{out_filename}_p.wav"),
             audio_m.T, sr, 'PCM_24')
   
    pv_in = os.path.join(path_in, label_path)
    pv_out = os.path.join(path_out, split, f"{out_filename}.pv")
   
    f0 = np.loadtxt(pv_in, delimiter=',', usecols=1)
   
    f0 = np.nan_to_num(f0)
    f0[f0 < 0] = 0.0
   
    np.savetxt(pv_out, f0, fmt="%.9f")