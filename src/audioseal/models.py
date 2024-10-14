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

def chunk_audio(audio_tensor, sample_rate):
    chunk_length = sample_rate  # Define chunk length as 1 second
    chunks = []
    num_samples = audio_tensor.size(1)  # Get the total number of samples in the audio
    start = 0

    while start < num_samples:
        end = start + chunk_length  # End of the current chunk
        chunks.append(audio_tensor[:, start:end])  # Append chunk of audio
        start = end

    return chunks

def recombine_audio(chunks_list):
    return torch.cat(chunks_list, dim=1)  # Recombine the chunks along the sample dimension

class AudioSealWM(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, msg_processor: Optional[torch.nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor
        self._message: Optional[torch.Tensor] = None
        self._original_payload: Optional[torch.Tensor] = None

    @property
    def message(self) -> Optional[torch.Tensor]:
        return self._message

    @message.setter
    def message(self, message: torch.Tensor) -> None:
        self._message = message

    def get_original_payload(self) -> Optional[torch.Tensor]:
        return self._original_payload

    def get_watermark(self, x: torch.Tensor, sample_rate: Optional[int] = None, message: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Call the forward method manually here
        print("get_watermark called")
        return self.forward(x, sample_rate, message)

    def forward(self, x: torch.Tensor, sample_rate: Optional[int] = None, message: Optional[torch.Tensor] = None,
                n_fft: int = 2048, hop_length: int = 512, alpha_val: float = 1.0) -> torch.Tensor:
        print("Forward method called!")  # This should always print if forward is being executed
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000

        if sample_rate != 16000:
            x_np = x.detach().cpu().numpy()  # Ensure detached tensor is converted to NumPy array
            resampled_x = librosa.resample(x_np, orig_sr=sample_rate, target_sr=16000)
            x = torch.tensor(resampled_x, device=x.device)

        # Split the audio into chunks
        print("checkpoint 0")
        audio_chunks = chunk_audio(x, sample_rate)
        print("checkpoint 0.1")
        wm_audio_list = []  # List to hold watermarked audio chunks
        print("checkpoint 0.2")
        # Iterate over chunks and apply watermarking to each chunk
        for chunk in audio_chunks:
            print("checkpoint 0.3")
            hidden = self.encoder(chunk.unsqueeze(0))  # Add batch dimension
            print("checkpoint 1")
            if self.msg_processor is not None:
                if message is None:
                    if self.message is None:
                        message = torch.randint(0, 2, (1, self.msg_processor.nbits), device=chunk.device)
                        print("checkpoint 2")
                    else:
                        message = self.message.to(device=chunk.device)
                else:
                    message = message.to(device=chunk.device)

                hidden = self.msg_processor(hidden, message)
                self._original_payload = message

            watermark = self.decoder(hidden)  # Decode hidden representation into a watermark

            # Combine the watermark with the audio chunk using the given alpha value
            wm_audio_chunk = chunk + alpha_val * watermark.squeeze(0)
            wm_audio_list.append(wm_audio_chunk)

        # Recombine the watermarked audio chunks
        watermarked_audio = recombine_audio(wm_audio_list)
        print("checkpoint 3")
        return watermarked_audio

class AudioSealDetector(torch.nn.Module):
    def __init__(self, *args, nbits: int = 0, **kwargs):
        super().__init__()
        encoder = SEANetEncoderKeepDimension(*args, **kwargs)
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.detector = torch.nn.Sequential(encoder, last_layer)
        self.nbits = nbits

    def detect_watermark(self, x: torch.Tensor, sample_rate: Optional[int] = None, message_threshold: float = 0.5) -> Tuple[float, torch.Tensor]:
        result, message = self.forward(x, sample_rate=sample_rate)
        print("Forward method in detector called!")
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
            x_np = x.detach().cpu().numpy()
            resampled_x = librosa.resample(x_np, orig_sr=sample_rate, target_sr=16000)
            x = torch.tensor(resampled_x, device=x.device)

        result = self.detector(x)
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        message = self.decode_message(result[:, 2:, :])
        return result[:, :2, :], message
