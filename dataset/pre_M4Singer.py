import os
import json
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
import torch
import torchcrepe

csv_file = r'D:\ICASSP_2024\SVSDT\dataset\INFO\m4singer.csv'
path_in_root = r'D:\Dataset\m4singer\m4singer'
path_out = r'D:\ICASSP_2024\SVSDT\dataset\M4Singer'
hop_length = 160
sr = 16000
fmin = librosa.note_to_hz('E2')
fmax = librosa.note_to_hz('C7')
model = 'full'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 12:
        batch_size = 8192
    elif vram_gb >= 8:
        batch_size = 4096
    elif vram_gb >= 6:
        batch_size = 2048
    else:
        batch_size = 1024
else:
    batch_size = 512

periodicity_threshold = 0.21

with open(os.path.join(path_in_root, 'meta.json'), 'r', encoding='utf-8') as f:
    metadata = {item['item_name']: item for item in json.load(f)}

df_info = pd.read_csv(csv_file)
frame_time = hop_length / sr

for _, row in tqdm(df_info.iterrows(), total=len(df_info)):
    relative_path, split = row['name'], row['split']
    folder_name = relative_path.split('/')[0]
    file_index = os.path.splitext(relative_path.split('/')[1])[0]
    item_name = f"{folder_name}#{file_index}"
    base_filename = item_name.replace('#', '_')
    
    audio_m, _ = librosa.load(os.path.join(path_in_root, relative_path), sr=sr, mono=True)
    os.makedirs(os.path.join(path_out, split), exist_ok=True)
    sf.write(os.path.join(path_out, split, f"{base_filename}_p.wav"),
             audio_m.T, sr, 'PCM_24')
    audio_tensor = torch.from_numpy(audio_m).to(device).unsqueeze(0)
    
    pitch, periodicity = torchcrepe.predict(
        audio_tensor, sr, hop_length, fmin, fmax, model,
        batch_size=batch_size, device=device, return_periodicity=True
    )
    
    f0_sequence = pitch.squeeze(0).cpu().numpy()
    confidence = periodicity.squeeze(0).cpu().numpy()
    f0_sequence[confidence < periodicity_threshold] = 0.0
    
    item_data = metadata[item_name]
    n_frames = f0_sequence.shape[0]
    current_time = 0.0
    
    for note, duration in zip(item_data['notes'], item_data['notes_dur']):
        if note == 0:
            start_frame = int(round(current_time / frame_time))
            end_frame = int(round((current_time + duration) / frame_time))
            start_frame = max(0, start_frame)
            end_frame = min(n_frames, end_frame)
            if start_frame < end_frame:
                f0_sequence[start_frame:end_frame] = 0.0
        current_time += duration
    
    f0_sequence[np.isnan(f0_sequence)] = 0.0
    f0_sequence[f0_sequence < 0] = 0.0
    
    np.savetxt(os.path.join(path_out, split, f"{base_filename}.pv"), 
               f0_sequence, fmt="%.6f")