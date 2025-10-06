import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from src import MIR1K, cycle, summary, DJCM, FL
from evaluate import evaluate

def train(weight_pe):
    alpha = 10
    gamma = 0
    in_channels = 1
    n_blocks = 1
    latent_layers = 1
    seq_l = 2.56
    hop_length = 20
    seq_frames = int(seq_l * 1000 / hop_length)
    logdir = 'runs/MIR1K_VPE_Only/' + 'nblocks' + str(n_blocks) + '_latent' + str(latent_layers) + '_frames' + str(seq_frames) \
             + '_alpha' + str(alpha) + '_gamma' + str(gamma) + '_pe' + str(weight_pe) + '_gateT'

    pitch_th = 0.5
    learning_rate = 5e-4
    batch_size = 16
    clip_grad_norm = 3
    learning_rate_decay_rate = 0.95
    learning_rate_decay_epochs = 5
    train_epochs = 250
    early_stop_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MIR1K(path='./dataset/MIR1K', hop_length=hop_length, groups=['train'], sequence_length=seq_l)
    print('train nums:', len(train_dataset))
    valid_dataset = MIR1K(path='./dataset/MIR1K', hop_length=hop_length, groups=['test'], sequence_length=None)
    print('valid nums:', len(valid_dataset))
    data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    epoch_nums = len(data_loader)
    print('epoch_nums:', epoch_nums)
    learning_rate_decay_steps = len(data_loader) * learning_rate_decay_epochs
    iterations = epoch_nums * train_epochs

    resume_iteration = None
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    if resume_iteration is None:
        model = DJCM(in_channels, n_blocks, hop_length, latent_layers, seq_frames)
        model = nn.DataParallel(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    summary(model)

    best_sdr, best_rpa, best_gnsdr, best_rca, best_it = 0, 0, 0, 0, 0
    loop = tqdm(range(resume_iteration + 1, iterations + 1))

    for i, data in zip(loop, cycle(data_loader)):
        model.train()
        
        audio_m = data['audio_m'].to(device)
        pitch_label = data['pitch'].to(device)

        # Forward pass - only needs audio_m for VPE
        out_pitch = model(audio_m)
        
        # Loss - solo pitch estimation
        loss_pitch = FL(out_pitch, pitch_label, alpha, gamma)
        loss_total = weight_pe * loss_pitch

        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step()

        # Logging
        if i % 10 == 0:  # Log every 10 iterations to reduce output
            print(f'{i}\tloss_total: {loss_total.item():.6f}\tloss_pe: {loss_pitch.item():.6f}')

        writer.add_scalar('loss/loss_total', loss_total.item(), global_step=i)
        writer.add_scalar('loss/loss_pe', loss_pitch.item(), global_step=i)

        # Validation
        if i % epoch_nums == 0:
            print('*' * 50)
            print(f'Epoch validation at iteration {i}')
            model.eval()
            with torch.no_grad():
                metrics = evaluate(valid_dataset, model, batch_size, hop_length, seq_l, device, None, pitch_th)
                
                # Log all metrics
                for key, value in metrics.items():
                    if key != 'LENGTH' and key != 'NSDR_W':
                        writer.add_scalar('validation/' + key, np.mean(value), global_step=i)
                
                # Calculate metrics
                gnsdr = np.round((np.sum(metrics["NSDR_W"]) / np.sum(metrics["LENGTH"])), 2)
                writer.add_scalar('validation/GNSDR', gnsdr, global_step=i)
                sdr = np.round(np.mean(metrics['SDR']), 2)
                rpa = np.round(np.mean(metrics['RPA']) * 100, 2)
                rca = np.round(np.mean(metrics['RCA']) * 100, 2)
                oa = np.round(np.mean(metrics['OA']) * 100, 2)
                
                print(f'SDR: {sdr}, GNSDR: {gnsdr}, RPA: {rpa}%, RCA: {rca}%, OA: {oa}%')
                
                # Save best model
                if sdr + rpa >= best_sdr + best_rpa:
                    best_sdr, best_gnsdr, best_rpa, best_rca, best_it = sdr, gnsdr, rpa, rca, i
                    print(f'New best model at iteration {i}!')
                    
                    with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                        f.write(f'{i}\t')
                        f.write(f'{best_sdr}±{np.round(np.std(metrics["SDR"]), 2)}\t')
                        f.write(f'{best_gnsdr}\t')
                        f.write(f'{best_rpa}±{np.round(np.std(metrics["RPA"]) * 100, 2)}\t')
                        f.write(f'{best_rca}±{np.round(np.std(metrics["RCA"]) * 100, 2)}\t')
                        f.write(f'{oa}±{np.round(np.std(metrics["OA"]) * 100, 2)}\n')
                    
                    torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

        # Early stopping
        if i - best_it >= epoch_nums * early_stop_epochs:
            print(f'Early stopping at iteration {i}')
            print(f'Best iteration: {best_it}, SDR: {best_sdr}, RPA: {best_rpa}%')
            break


weight_pe_values = [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
for weight_pe in weight_pe_values:
    print(f'\n{"="*60}')
    print(f'Training with weight_pe = {weight_pe}')
    print(f'{"="*60}\n')
    train(weight_pe)