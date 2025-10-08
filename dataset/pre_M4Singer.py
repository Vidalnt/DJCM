import os
import json
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

csv_file = r'D:\ICASSP_2024\SVSDT\dataset\INFO\m4singer.csv'
path_in_root = r'D:\Dataset\m4singer\m4singer'
path_out = r'D:\ICASSP_2024\SVSDT\dataset\M4Singer'
hop_length = 160
sr = 16000

with open(os.path.join(path_in_root, 'meta.json'), 'r', encoding='utf-8') as f:
    metadata = {item['item_name']: item for item in json.load(f)}

df_info = pd.read_csv(csv_file)
frame_time = hop_length / sr

for _, row in tqdm(df_info.iterrows()):
    relative_path, split = row['name'], row['split']
    
    folder_name = relative_path.split('/')[0]
    file_index = os.path.splitext(relative_path.split('/')[1])[0]
    item_name = f"{folder_name}#{file_index}"
    base_filename = item_name.replace('#', '_')
    
    audio_m, _ = librosa.load(os.path.join(path_in_root, relative_path), sr=sr, mono=True)
    os.makedirs(os.path.join(path_out, split), exist_ok=True)
    sf.write(os.path.join(path_out, split, f"{base_filename}_p.wav"),
             audio_m.T, sr, 'PCM_24')
    
    item_data = metadata[item_name]
    n_frames = int(np.ceil(len(audio_m) / hop_length))
    f0_sequence = np.zeros(n_frames)
    
    current_time = 0.0
    for ph_dur, note_pitch in zip(item_data['ph_dur'], item_data['notes']):
        start_frame = int(round(current_time / frame_time))
        end_frame = int(round((current_time + ph_dur) / frame_time))
        
        if note_pitch > 0:
            freq = 440 * (2 ** ((note_pitch - 69) / 12))
            f0_sequence[start_frame:min(end_frame, n_frames)] = freq
        
        current_time += ph_dur
    
    f0_sequence[f0_sequence < 0] = 0.0
    f0_sequence[np.isnan(f0_sequence)] = 0.0
    
    np.savetxt(os.path.join(path_out, split, f"{base_filename}.pv"), 
               f0_sequence, fmt="%.6f")