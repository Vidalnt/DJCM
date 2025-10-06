import numpy as np
import torch


class Inference:
    """
    Inference class for VPE (Vocal Pitch Estimation) only - no SVS
    """
    def __init__(self, model, seg_len, seg_frames, hop_length, batch_size, device):
        super(Inference, self).__init__()
        self.model = model.eval()
        self.seg_len = seg_len
        self.seg_frames = seg_frames
        self.batch_size = batch_size
        self.hop_length = hop_length
        self.device = device

    def inference(self, audio):
        """
        Perform inference on audio - returns only pitch predictions
        
        Args:
            audio: Input audio tensor [channels, length]
            
        Returns:
            pitch_pred: Pitch predictions [frames, freq_bins]
        """
        with torch.no_grad():
            padded_audio = self.pad_audio(audio)
            segments = self.en_frame(padded_audio)
            pitch_segments = self.forward_in_mini_batch(self.model, segments)
            pitch_pred = self.de_frame(pitch_segments, type_seg='pitch')[:(audio.shape[-1] // self.hop_length + 1)]
            return pitch_pred

    def pad_audio(self, audio):
        """Pad audio to fit segment length"""
        c, audio_len = audio.shape
        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = seg_nums * self.seg_len - audio_len + self.seg_len // 2
        padded_audio = torch.cat([
            torch.zeros(c, self.seg_len // 4).to(self.device), 
            audio,
            torch.zeros(c, pad_len - self.seg_len // 4).to(self.device)
        ], dim=1)
        return padded_audio

    def en_frame(self, audio):
        """Split audio into overlapping segments"""
        c, audio_len = audio.shape
        assert audio_len % (self.seg_len // 2) == 0
        segments = []
        start = 0
        while start + self.seg_len <= audio_len:
            segments.append(audio[:, start:start + self.seg_len])
            start += self.seg_len // 2
        segments = torch.stack(segments, dim=0)
        return segments

    def forward_in_mini_batch(self, model, segments):
        """
        Process segments in mini-batches - VPE only
        
        Args:
            model: DJCM model
            segments: Audio segments [num_segments, channels, length]
            
        Returns:
            pitch_segments: Pitch predictions for all segments
        """
        pitch_segments = []
        segments_num = segments.shape[0]
        batch_start = 0
        
        while True:
            if batch_start + self.batch_size >= segments_num:
                # Last batch - may need padding
                batch_tmp = segments[batch_start:].shape[0]
                segment_in = torch.cat([
                    segments[batch_start:],
                    torch.zeros_like(segments)[:self.batch_size - batch_tmp].to(self.device)
                ], dim=0)
                
                # Forward pass - solo pitch
                pitch_pred = model(segment_in)
                pitch_segments.append(pitch_pred[:batch_tmp, :])
                break
            else:
                # Regular batch
                segment_in = segments[batch_start:batch_start + self.batch_size]
                pitch_pred = model(segment_in)
                pitch_segments.append(pitch_pred)
            
            batch_start += self.batch_size
        
        pitch_segments = torch.cat(pitch_segments, dim=0)
        return pitch_segments

    def de_frame(self, segments, type_seg='pitch'):
        """
        Reconstruct full sequence from overlapping segments
        
        Args:
            segments: Segmented predictions
            type_seg: 'pitch' only (audio removed)
            
        Returns:
            output: Reconstructed sequence
        """
        output = []
        # Only pitch reconstruction (audio removed)
        for segment in segments:
            output.append(segment[self.seg_frames // 4: int(self.seg_frames * 0.75)])
        output = torch.cat(output, dim=0)
        return output