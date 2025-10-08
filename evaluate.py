import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from src import to_local_average_cents, Inference
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, overall_accuracy, raw_chroma_accuracy
from mir_eval.melody import voicing_false_alarm, voicing_recall
from src import SAMPLE_RATE


def evaluate(dataset, model, batch_size, hop_length, seq_l, device, path=None, pitch_th=0.5):
    """
    Evaluate VPE (Vocal Pitch Estimation) only - no SVS
    
    Args:
        dataset: Dataset to evaluate
        model: DJCM model (VPE only)
        batch_size: Batch size for inference
        hop_length: Hop length in ms
        seq_l: Sequence length in seconds
        device: torch device
        path: Optional path to save pitch predictions
        pitch_th: Threshold for pitch detection
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    metrics = defaultdict(list)
    seq_l = int(seq_l * SAMPLE_RATE)
    hop_length = int(hop_length / 1000 * SAMPLE_RATE)
    seg_frames = seq_l // hop_length
    infer = Inference(model, seq_l, seg_frames, hop_length, batch_size, device)
    
    for data in tqdm(dataset):
        audio_m = data['audio_m'].to(device)
        pitch_label = data['pitch'].to(device)
        
        # Inference - solo pitch estimation
        pitch_pred = infer.inference(audio_m)
        
        # Loss - solo pitch
        loss_pitch = F.binary_cross_entropy(pitch_pred, pitch_label)
        metrics['loss_pe'].append(loss_pitch.item())
        
        # Convert to cents
        cents = to_local_average_cents(pitch_label.detach().cpu().numpy(), None, pitch_th)
        cents_pred = to_local_average_cents(pitch_pred.detach().cpu().numpy(), None, pitch_th)
        
        # Convert cents to frequencies
        freqs = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents])
        freqs_pred = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents_pred])
        
        # Time array
        time_slice = np.array([i * hop_length / SAMPLE_RATE for i in range(len(freqs))])
        
        # Pitch accuracy metrics
        try:
            ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freqs, time_slice, freqs_pred)
            rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
            rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
            oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
            vfa = voicing_false_alarm(ref_v, est_v)
            vr = voicing_recall(ref_v, est_v)
            
            metrics['RPA'].append(rpa)
            metrics['RCA'].append(rca)
            metrics['OA'].append(oa)
            metrics['VFA'].append(vfa)
            metrics['VR'].append(vr)

        except Exception as e:
            print(f'ERROR [{filename}]: {str(e)}')
            continue
        
        # Save predictions if path provided
        if path is not None:
            os.makedirs(path, exist_ok=True)
            df_pitch = pd.DataFrame(columns=['times', 'freqs', 'confi'])
            df_pitch['times'] = time_slice
            df_pitch['freqs'] = freqs_pred
            df_pitch['confi'] = torch.max(pitch_pred, dim=-1).values.cpu().numpy()
            
            # Save with appropriate filename
            filename = data.get('file', f'prediction_{len(metrics["RPA"])}.csv')
            if filename.endswith('.wav'):
                filename = filename.replace('.wav', '_pitch.csv')
            else:
                filename = f'{filename}_pitch.csv'
            
            df_pitch.to_csv(os.path.join(path, filename), index=False)
        
        # Print metrics for this sample
        print(f'RPA: {rpa:.4f}\tRCA: {rca:.4f}\tOA: {oa:.4f}')
    
    return metrics