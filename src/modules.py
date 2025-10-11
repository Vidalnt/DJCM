import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from .seq import BiGRU, BiLSTM , TransformerBlock
from .constants import N_CLASS, N_MELS
from einops.layers.torch import Rearrange

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block.
    Source: "Squeeze-and-Excitation Networks" (Hu et al., 2018).
    Purpose: Improves channel interdependencies at negligible computational cost. It acts as
             an attention mechanism that adaptively recalibrates channel-wise feature
             responses, allowing the model to focus on more informative channels.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DilatedConvBlock(nn.Module):
    """
    Block of dilated convolutions with residual connections.
    Source: Inspired by "WaveNet" (van den Oord et al., 2016) and Temporal
            Convolutional Networks from "An Empirical Evaluation..." (Bai et al., 2018).
    Purpose: Efficiently captures long-range temporal dependencies. The exponentially
             increasing dilation rate allows the receptive field to grow rapidly, enabling
             the model to understand broad temporal context (like melodies and phrases)
             without the computational cost of recurrent layers.
    """
    def __init__(self, channels, n_blocks=3, bias=False):
        super(DilatedConvBlock, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** i
            self.convs.append(nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=bias),
            ))
            
    def forward(self, x):
        for conv in self.convs:
            res = x
            x = conv(x)
            x = x + res
        return x


def init_layer(layer: nn.Module):
    r"""Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module):
    r"""Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)


class Wav2Spec(nn.Module):
    def __init__(self, hop_length, window_size):
        super(Wav2Spec, self).__init__()
        self.hop_length = hop_length
        self.stft = STFT(window_size, hop_length, window_size)

    def forward(self, audio):
        bs, c, segment_samples = audio.shape
        audio = audio.reshape(bs * c, segment_samples)
        real, imag = self.stft(audio[:, :-1])
        mag = torch.clamp(real ** 2 + imag ** 2, 1e-10, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        _, _, time_steps, freq_bins = mag.shape
        mag = mag.reshape(bs, c, time_steps, freq_bins)
        cos = cos.reshape(bs, c, time_steps, freq_bins)
        sin = sin.reshape(bs, c, time_steps, freq_bins)
        return mag, cos, sin


class Spec2Wav(nn.Module):
    def __init__(self, hop_length, window_size):
        super(Spec2Wav, self).__init__()
        self.istft = ISTFT(window_size, hop_length, window_size)

    def forward(self, x, spec_m, cos_m, sin_m, audio_len):
        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)
        mask_spec = torch.sigmoid(x[:, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        linear_spec = x[:, :, 3, :, :]
        out_cos = cos_m * mask_cos - sin_m * mask_sin
        out_sin = sin_m * mask_cos + cos_m * mask_sin
        out_spec = F.relu(spec_m.detach() * mask_spec + linear_spec)
        out_real = (out_spec * out_cos).reshape(bs * c // 4, 1, time_steps, freqs_steps)
        out_imag = (out_spec * out_sin).reshape(bs * c // 4, 1, time_steps, freqs_steps)
        audio = self.istft(out_real, out_imag, audio_len).reshape(bs, c // 4, audio_len)
        return audio, out_spec


class ResConvBlock(nn.Module):
    def __init__(self, in_planes, planes, bias=False, use_se=False):
        super(ResConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, (3, 3), padding=(1, 1), bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, (3, 3), padding=(1, 1), bias=bias)
        self.se = SELayer(planes) if use_se else None
        if in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False
        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        if self.se is not None:
            out = self.se(out)
        if self.is_shortcut:
            return self.shortcut(x) + out
        else:
            return out + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, kernel_size, bias, use_se=False):
        super(EncoderBlock, self).__init__()
        self.conv = nn.ModuleList([
            ResConvBlock(in_channels, out_channels, bias, use_se)
        ])
        for i in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels, bias, use_se))
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size)
        else:
            self.pool = None

    def forward(self, x):
        for each_layer in self.conv:
            x = each_layer(x)
        if self.pool is not None:
            return x, self.pool(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride, bias, gate=False, use_se=False):
        super(DecoderBlock, self).__init__()
        self.gate = gate
        if self.gate:
            self.W_g = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, (1, 1)),
                nn.BatchNorm2d(out_channels // 2)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, (1, 1)),
                nn.BatchNorm2d(out_channels // 2)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(out_channels // 2, 1, (1, 1)),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride, (0, 0), bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv = nn.ModuleList([
            ResConvBlock(out_channels * 2, out_channels, bias, use_se)
        ])
        for i in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels, bias, use_se))
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, x, concat):
        x = self.conv1(F.relu_(self.bn1(x)))
        if self.gate:
            concat = x * self.psi(F.relu_(self.W_g(x) + self.W_x(concat)))
        x = torch.cat((x, concat), dim=1)
        for each_layer in self.conv:
            x = each_layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, n_blocks, use_se=False):
        super(Encoder, self).__init__()
        self.en_blocks = nn.ModuleList([
            EncoderBlock(in_channels, 32, n_blocks, (1, 2), False, use_se),
            EncoderBlock(32, 64, n_blocks, (1, 2), False, use_se),
            EncoderBlock(64, 128, n_blocks, (1, 2), False, use_se),
            EncoderBlock(128, 256, n_blocks, (1, 2), False, use_se),
            EncoderBlock(256, 384, n_blocks, (1, 2), False, use_se),
            EncoderBlock(384, 384, n_blocks, (1, 2), False, use_se)
        ])

    def forward(self, x):
        concat_tensors = []
        for layer in self.en_blocks:
            _, x = layer(x)
            concat_tensors.append(_)
        return x, concat_tensors


class Decoder(nn.Module):
    def __init__(self, n_blocks, gate=False, use_se=False):
        super(Decoder, self).__init__()
        self.de_blocks = nn.ModuleList([
            DecoderBlock(384, 384, n_blocks, (1, 2), False, gate, use_se),
            DecoderBlock(384, 384, n_blocks, (1, 2), False, gate, use_se),
            DecoderBlock(384, 256, n_blocks, (1, 2), False, gate, use_se),
            DecoderBlock(256, 128, n_blocks, (1, 2), False, gate, use_se),
            DecoderBlock(128, 64, n_blocks, (1, 2), False, gate, use_se),
            DecoderBlock(64, 32, n_blocks, (1, 2), False, gate, use_se),
        ])

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.de_blocks):
            x = layer(x, concat_tensors[-1-i])
        return x


class LatentBlocks(nn.Module):
    def __init__(self, n_blocks, latent_layers, use_dilated_conv=False):
        super(LatentBlocks, self).__init__()
        self.use_dilated_conv = use_dilated_conv
        if use_dilated_conv:
            self.latent_blocks = DilatedConvBlock(channels=384, n_blocks=latent_layers)
        else:
            self.latent_blocks = nn.ModuleList([])
            for i in range(latent_layers):
                self.latent_blocks.append(EncoderBlock(384, 384, n_blocks, None, False))

    def forward(self, x):
        if self.use_dilated_conv:
            return self.latent_blocks(x)
        else:
            for layer in self.latent_blocks:
                x = layer(x)
            return x

class TimbreFilter(nn.Module):
    def __init__(self, latent_rep_channels, use_se=False):
        super(TimbreFilter, self).__init__()
        self.layers = nn.ModuleList()
        for latent_rep in latent_rep_channels:
            self.layers.append(ResConvBlock(latent_rep[0], latent_rep[0], use_se=use_se))

    def forward(self, x_tensors):
        out_tensors = []
        for i, layer in enumerate(self.layers):
            out_tensors.append(layer(x_tensors[i]))
        return out_tensors

class PE_Decoder(nn.Module):
    def __init__(self, n_blocks, seq_frames, seq='gru', seq_layers=1, gate=False, use_se=False, output_channels=1):
        super(PE_Decoder, self).__init__()
        self.output_channels = output_channels
        self.de_blocks = Decoder(n_blocks, gate, use_se=use_se)
        self.tf = TimbreFilter([(32, 0), (64, 0), (128, 0), (256, 0), (384, 0), (384, 0)], use_se=use_se)
        self.after_conv1 = EncoderBlock(32, 32, n_blocks, None, False, use_se=use_se)
        self.after_conv2 = nn.Conv2d(32, self.output_channels, (1, 1))
        init_layer(self.after_conv2)
        if seq.lower() == 'gru':
            self.fc = nn.Sequential(
                BiGRU((seq_frames, N_MELS), (1, N_MELS), self.output_channels, seq_layers),
                nn.Linear(self.output_channels * N_MELS, N_CLASS),
                nn.Sigmoid()
            )
        elif seq.lower() == 'lstm':
            self.fc = nn.Sequential(
                BiLSTM((seq_frames, N_MELS), (1, N_MELS), self.output_channels, seq_layers),
                nn.Linear(self.output_channels * N_MELS, N_CLASS),
                nn.Sigmoid()
            )
        elif seq.lower() == "transformer":
            self.fc = nn.Sequential(
                TransformerBlock(
                    in_channels=self.output_channels, 
                    n_mels=N_MELS, 
                    n_heads=8, 
                    dim_feedforward=4,
                    n_layers=2
                ),
                nn.Linear(self.output_channels * N_MELS, N_CLASS),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                Rearrange('b c t f -> b t (c f)'),
                nn.Linear(self.output_channels * N_MELS, N_CLASS),
                nn.Sigmoid()
            )

    def forward(self, x, concat_tensors):
        ft = self.tf(concat_tensors)
        x = self.de_blocks(x, ft)
        x = self.after_conv2(self.after_conv1(x))
        x = self.fc(x)
        if x.dim() == 4 and x.size(1) == 1:
             x = x.squeeze(1)
        return x