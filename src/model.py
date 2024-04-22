import torch.nn as nn
import torch.nn.functional as F
from .modules import Encoder, LatentBlocks, SVS_Decoder, PE_Decoder
from .modules import Wav2Spec, Spec2Wav, SVS_PE_Base, SVS_PE_MMOE, init_bn
from .constants import WINDOW_LENGTH, SAMPLE_RATE


class JM_Base(nn.Module):
    def __init__(self, in_channels, n_blocks, hop_length, latent_layers, seq_frames, seq='gru', seq_layers=1):
        super(JM_Base, self).__init__()
        self.to_spec = Wav2Spec(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.to_wav = Spec2Wav(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.bn = nn.BatchNorm2d(2048 // 2 + 1, momentum=0.01)
        init_bn(self.bn)
        # in_channels, n_blocks, latent_layers, seq_frames, seq='gru', seq_layers=1
        self.model = SVS_PE_Base(in_channels, n_blocks, latent_layers, seq_frames, seq, seq_layers)

    def forward(self, audio_m, audio_v=None):
        spec_m, cos_m, sin_m = self.to_spec(audio_m)
        x = self.bn(spec_m.transpose(1, 3)).transpose(1, 3)[..., :-1]
        pe_out, svs_out = self.model(x)
        out_audio, out_spec = self.to_wav(svs_out, spec_m, cos_m, sin_m, audio_m.shape[-1])
        if audio_v is None:
            return out_audio, pe_out
        else:
            spec_v, _, _ = self.to_spec(audio_v)
            loss_spec = F.l1_loss(out_spec[..., :-1], spec_v[..., :-1])
            return out_audio, pe_out, loss_spec


class JM_MMOE(nn.Module):
    def __init__(self, in_channels, n_blocks, hop_length, latent_layers, seq_frames, expert_num=2, seq='gru',
                 seq_layers=1):
        super(JM_MMOE, self).__init__()
        self.to_spec = Wav2Spec(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.to_wav = Spec2Wav(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.bn = nn.BatchNorm2d(2048 // 2 + 1, momentum=0.01)
        init_bn(self.bn)
        # in_channels, n_blocks, latent_layers, seq_frames, expert_num=2, seq='gru', seq_layers=1
        self.model = SVS_PE_MMOE(in_channels, n_blocks, latent_layers, seq_frames, expert_num, seq, seq_layers)

    def forward(self, audio_m, audio_v=None):
        spec_m, cos_m, sin_m = self.to_spec(audio_m)
        x = self.bn(spec_m.transpose(1, 3)).transpose(1, 3)[..., :-1]
        pe_out, svs_out = self.model(x)
        out_audio, out_spec = self.to_wav(svs_out, spec_m, cos_m, sin_m, audio_m.shape[-1])
        if audio_v is None:
            return out_audio, pe_out
        else:
            spec_v, _, _ = self.to_spec(audio_v)
            loss_spec = F.l1_loss(out_spec[..., :-1], spec_v[..., :-1])
            return out_audio, pe_out, loss_spec


class DJCM(nn.Module):
    def __init__(self, in_channels, n_blocks, hop_length, latent_layers, seq_frames, gate=False, seq='gru', seq_layers=1):
        super(DJCM, self).__init__()
        self.to_spec = Wav2Spec(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.to_wav = Spec2Wav(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.bn = nn.BatchNorm2d(2048 // 2 + 1, momentum=0.01)
        init_bn(self.bn)
        self.svs_encoder = Encoder(in_channels, n_blocks)
        self.svs_latent = LatentBlocks(n_blocks, latent_layers)
        self.svs_decoder = SVS_Decoder(in_channels, n_blocks, gate)

        self.pe_encoder = Encoder(in_channels, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks, seq_frames, seq, seq_layers, gate)

    def forward(self, audio_m, audio_v=None):
        spec_m, cos_m, sin_m = self.to_spec(audio_m)
        x = self.bn(spec_m.transpose(1, 3)).transpose(1, 3)[..., :-1]
        x, concat_tensors = self.svs_encoder(x)
        x = self.svs_latent(x)
        x = self.svs_decoder(x, concat_tensors)
        svs_out = F.pad(x, pad=(0, 1))
        out_audio, out_spec = self.to_wav(svs_out, spec_m, cos_m, sin_m, audio_m.shape[-1])
        x, concat_tensors = self.pe_encoder(out_spec[..., :-1])
        x = self.pe_latent(x)
        pe_out = self.pe_decoder(x, concat_tensors)
        # pe_out, svs_out = self.model(x)
        if audio_v is None:
            return out_audio, pe_out
        else:
            spec_v, _, _ = self.to_spec(audio_v)
            loss_spec = F.l1_loss(out_spec[..., :-1], spec_v[..., :-1])
            return out_audio, pe_out, loss_spec
