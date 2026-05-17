"""
Inference Pipeline for JUND-F0

Provides:
- Single-file F0 extraction
- Batch processing
- Integration with voice conversion pipelines (RVC, So-VITS-SVC)
- Numpy/PyTorch bridge for compatibility
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, Union

from .model import JUNDF0, JUNDF0Config


class JUNDF0Inference:
    """
    Inference wrapper for JUND-F0 model.

    Provides a simple API for extracting F0 and V/UV from audio files
    or raw waveform arrays. Designed to be compatible with existing
    voice conversion pipelines (RVC, So-VITS-SVC, etc.).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[JUNDF0Config] = None,
        device: str = "auto",
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            config: Model configuration (if None, uses default)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = config or JUNDF0Config()
        self.model = JUNDF0(self.config).to(self.device)

        if model_path:
            self.load_model(model_path)

        # Pre-compute mel transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
        ).to(self.device)
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(self.device)

        self.model.eval()

    def load_model(self, model_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Also restore config if available
            if "config" in checkpoint:
                saved_config = checkpoint["config"]
                for k, v in saved_config.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"JUND-F0 model loaded from {model_path}")

    @torch.no_grad()
    def extract_f0(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        return_vuv: bool = True,
        return_prob: bool = False,
    ) -> dict:
        """
        Extract F0 from audio.

        Args:
            audio: Audio input (file path, numpy array, or torch tensor)
            sample_rate: Sample rate of audio (if not provided, uses config)
            return_vuv: Whether to return V/UV flags
            return_prob: Whether to return V/UV probabilities

        Returns:
            Dictionary with:
                - f0: (time,) numpy array of F0 values in Hz
                - vuv: (time,) boolean V/UV flags (if return_vuv=True)
                - vuv_prob: (time,) float V/UV probabilities (if return_prob=True)
                - time: (time,) time stamps in seconds
        """
        # Load audio
        waveform = self._load_audio(audio, sample_rate)

        # Compute mel spectrogram
        mel = self.mel_transform(waveform.to(self.device))
        mel = self.amp_to_db(mel)
        mel = mel.squeeze(0)  # (n_mels, time)

        # Run model
        # Add batch dimension
        mel_batch = mel.unsqueeze(0)
        vuv, f0 = self.model.predict(mel_batch)

        # Remove batch dimension
        f0 = f0[0].cpu().numpy()
        vuv = vuv[0].cpu().numpy()

        # Time stamps
        n_frames = len(f0)
        time_stamps = np.arange(n_frames) * self.config.hop_length / self.config.sample_rate

        result = {
            "f0": f0,
            "time": time_stamps,
        }

        if return_vuv:
            result["vuv"] = vuv

        if return_prob:
            # Re-run to get probabilities
            outputs = self.model(mel_batch)
            result["vuv_prob"] = outputs["vuv_prob"][0, :, 0].cpu().numpy()

        return result

    def _load_audio(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Load and preprocess audio input."""
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(str(audio))
            target_sr = self.config.sample_rate

            # Resample if needed
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                waveform = resampler(waveform)

        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        elif isinstance(audio, torch.Tensor):
            waveform = audio.float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize
        waveform = waveform / (waveform.abs().max().clamp(min=1e-8))

        return waveform

    def extract_f0_from_file(self, audio_path: str) -> np.ndarray:
        """
        Simple API: extract F0 from audio file.

        Compatible with RVC / So-VITS-SVC F0 extraction interface.

        Returns:
            f0: (time,) numpy array of F0 values in Hz (0.0 for unvoiced)
        """
        result = self.extract_f0(audio_path, return_vuv=False, return_prob=False)
        return result["f0"]

    def extract_f0_batch(
        self,
        audio_paths: list,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Batch F0 extraction for multiple files.

        Args:
            audio_paths: List of audio file paths
            output_dir: If provided, save F0 arrays as .npy files

        Returns:
            Dictionary mapping file paths to F0 arrays
        """
        results = {}

        for i, path in enumerate(audio_paths):
            print(f"Processing {i+1}/{len(audio_paths)}: {Path(path).name}")
            result = self.extract_f0(path, return_vuv=True)
            results[path] = result

            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                name = Path(path).stem
                np.save(out_path / f"{name}_f0.npy", result["f0"])
                np.save(out_path / f"{name}_vuv.npy", result["vuv"])

        return results


def create_rvc_compatible_extractor(model_path: str, device: str = "auto"):
    """
    Create an RVC-compatible F0 extractor using JUND-F0.

    This function returns a callable that matches the interface used by
    RVC-Project/Retrieval-based-Voice-Conversion-WebUI for F0 extraction.

    Usage in RVC:
        from jund_f0.infer import create_rvc_compatible_extractor
        f0_extractor = create_rvc_compatible_extractor("best_model.pt")
        f0 = f0_extractor(audio_array, sample_rate)
    """
    inference = JUNDF0Inference(model_path=model_path, device=device)

    def extract_f0_rvc(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        """
        RVC-compatible F0 extraction function.

        Args:
            audio: (samples,) numpy float32 array
            sample_rate: Audio sample rate

        Returns:
            f0: (frames,) numpy float32 array of F0 in Hz
        """
        result = inference.extract_f0(
            audio,
            sample_rate=sample_rate,
            return_vuv=False,
        )
        return result["f0"].astype(np.float32)

    return extract_f0_rvc
