import torch.nn as nn
from .modules import Encoder, LatentBlocks, PE_Decoder
from .modules import init_bn
from .constants import WINDOW_LENGTH, SAMPLE_RATE, N_MELS, MEL_FMIN, MEL_FMAX
from .spec import MelSpectrogram


class DJCM(nn.Module):
    def __init__(self, n_blocks, hop_length, latent_layers, seq_frames, gate=False, seq='gru', seq_layers=1):
        super(DJCM, self).__init__()
        self.to_spec = MelSpectrogram(
            n_mel_channels=N_MELS,
            sampling_rate=SAMPLE_RATE,
            win_length=WINDOW_LENGTH,
            hop_length=hop_length,
            n_fft=WINDOW_LENGTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )
        self.bn = nn.BatchNorm2d(N_MELS, momentum=0.01)
        init_bn(self.bn)
        self.pe_encoder = Encoder(N_MELS, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks, seq_frames, seq, seq_layers, gate)

    def forward(self, audio_m):
        log_mel_spec = self.to_spec(audio_m.squeeze(1))
        x = log_mel_spec.unsqueeze(-1)
        x, concat_tensors = self.pe_encoder(x)
        x = self.pe_latent(x)
        pe_out = self.pe_decoder(x, concat_tensors)
        return pe_out
