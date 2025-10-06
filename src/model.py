import torch.nn as nn
from .modules import Encoder, LatentBlocks, PE_Decoder
from .modules import Wav2Spec, init_bn
from .constants import WINDOW_LENGTH, SAMPLE_RATE

class DJCM(nn.Module):
    def __init__(self, in_channels, n_blocks, hop_length, latent_layers, seq_frames, gate=False, seq='gru', seq_layers=1):
        super(DJCM, self).__init__()
        self.to_spec = Wav2Spec(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.bn = nn.BatchNorm2d(2048 // 2 + 1, momentum=0.01)
        init_bn(self.bn)
        self.pe_encoder = Encoder(in_channels, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks, seq_frames, seq, seq_layers, gate)

    def forward(self, audio_m, audio_v=None):
        spec_m, cos_m, sin_m = self.to_spec(audio_m)
        x = self.bn(spec_m.transpose(1, 3)).transpose(1, 3)[..., :-1]
        x, concat_tensors = self.pe_encoder(x)
        x = self.pe_latent(x)
        pe_out = self.pe_decoder(x, concat_tensors)
        return pe_out
