# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple
import librosa
import numpy as np
import torch

from audioseal.libs.audiocraft.modules.seanet import SEANetEncoderKeepDimension

logger = logging.getLogger("Audioseal")

COMPATIBLE_WARNING = """
AudioSeal is designed to work at a sample rate 16khz.
Implicit sampling rate usage is deprecated and will be removed in future version.
To remove this warning please add this argument to the function call:
sample_rate = your_sample_rate
"""

class MsgProcessor(torch.nn.Module):
    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = torch.nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)
        indices = indices.repeat(msg.shape[0], 1)
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)
        msg_aux = msg_aux.sum(dim=-2)
        msg_aux = msg_aux.unsqueeze(-1).repeat(1, 1, hidden.shape[2])
        hidden = hidden + msg_aux
        return hidden

def compute_stft_energy(audio: torch.Tensor, sr: int, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    batch_size = audio.size(0)
    energy_values = []

    for i in range(batch_size):
        y = audio[i].cpu().numpy()
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        frame_energy = torch.tensor(np.sum(stft ** 2, axis=0), device=audio.device)
        energy_values.append(frame_energy)
    
    energy_values = torch.stack(energy_values, dim=0)
    return energy_values

def compute_adaptive_alpha_librosa(energy_values: torch.Tensor, min_alpha: float = 0.5, max_alpha: float = 1.5) -> torch.Tensor:
    normalized_energy = (energy_values - energy_values.min(dim=1, keepdim=True)[0]) / (
        energy_values.max(dim=1, keepdim=True)[0] - energy_values.min(dim=1, keepdim=True)[0] + 1e-6
    )
    alpha_values = min_alpha + normalized_energy * (max_alpha - min_alpha)
    return alpha_values

class AudioSealWM(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, msg_processor: Optional[torch.nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor
        self._message: Optional[torch.Tensor] = None

    @property
    def message(self) -> Optional[torch.Tensor]:
        return self._message

    @message.setter
    def message(self, message: torch.Tensor) -> None:
        self._message = message

    def get_watermark(self, x: torch.Tensor, sample_rate: Optional[int] = None, message: Optional[torch.Tensor] = None) -> torch.Tensor:
        length = x.size(-1)
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000
        if sample_rate != 16000:
            x_np = x.detach().cpu().numpy()  # Ensure detached tensor is converted to NumPy array
            resampled_x = librosa.resample(x_np, orig_sr=sample_rate, target_sr=16000)
            x = torch.tensor(resampled_x, device=x.device)
        hidden = self.encoder(x)

        if self.msg_processor is not None:
            if message is None:
                if self.message is None:
                    message = torch.randint(0, 2, (x.shape[0], self.msg_processor.nbits), device=x.device)
                else:
                    message = self.message.to(device=x.device)
            else:
                message = message.to(device=x.device)

            hidden = self.msg_processor(hidden, message)

        watermark = self.decoder(hidden)

        if sample_rate != 16000:
            watermark_np = watermark.detach().cpu().numpy()  # Ensure detached tensor is converted to NumPy array
            resampled_watermark = librosa.resample(watermark_np, orig_sr=16000, target_sr=sample_rate)
            watermark = torch.tensor(resampled_watermark, device=watermark.device)

        return watermark[..., :length]

    def forward(self, x: torch.Tensor, sample_rate: Optional[int] = None, message: Optional[torch.Tensor] = None,
                n_fft: int = 2048, hop_length: int = 512, min_alpha: float = 0.5, max_alpha: float = 1.5) -> torch.Tensor:
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000

        energy_values = compute_stft_energy(x, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        adaptive_alpha = compute_adaptive_alpha_librosa(energy_values, min_alpha=min_alpha, max_alpha=max_alpha)

        num_frames = adaptive_alpha.size(1)
        stretched_alpha = torch.repeat_interleave(adaptive_alpha, hop_length, dim=1)
        stretched_alpha = stretched_alpha[:, :x.size(1)]

        wm = self.get_watermark(x, sample_rate=sample_rate, message=message)
        watermarked_audio = x + stretched_alpha.unsqueeze(1) * wm

        return watermarked_audio

class AudioSealDetector(torch.nn.Module):
    def __init__(self, *args, nbits: int = 0, **kwargs):
        super().__init__()
        encoder = SEANetEncoderKeepDimension(*args, **kwargs)
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.detector = torch.nn.Sequential(encoder, last_layer)
        self.nbits = nbits

    def detect_watermark(self, x: torch.Tensor, sample_rate: Optional[int] = None, message_threshold: float = 0.5) -> Tuple[float, torch.Tensor]:
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000
        result, message = self.forward(x, sample_rate=sample_rate)
        detected = (torch.count_nonzero(torch.gt(result[:, 1, :], 0.5)) / result.shape[-1])
        detect_prob = detected.cpu().item()
        message = torch.gt(message, message_threshold).int()
        return detect_prob, message

    def decode_message(self, result: torch.Tensor) -> torch.Tensor:
        assert (result.dim() > 2 and result.shape[1] == self.nbits) or (
            result.dim() == 2 and result.shape[0] == self.nbits
        ), f"Expect message of size [,{self.nbits}, frames] (get {result.size()})"
        decoded_message = result.mean(dim=-1)
        return torch.sigmoid(decoded_message)

    def forward(self, x: torch.Tensor, sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000
        if sample_rate != 16000:
            x_np = x.detach().cpu().numpy()  # Ensure detached tensor is converted to NumPy array
            resampled_x = librosa.resample(x_np, orig_sr=sample_rate, target_sr=16000)
            x = torch.tensor(resampled_x, device=x.device)
        result = self.detector(x)
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        message = self.decode_message(result[:, 2:, :])
        return result[:, :2, :], message
