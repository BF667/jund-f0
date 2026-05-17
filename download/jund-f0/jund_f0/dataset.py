"""
VCTK Dataset Loader for JUND-F0 Training

Handles:
1. Download and extraction of VCTK corpus
2. Audio preprocessing (resampling, normalization)
3. Mel spectrogram computation
4. Pseudo F0 label generation using pyin/CREPE
5. V/UV label extraction from F0 labels
6. Data augmentation (optional noise, pitch shift)
"""

import os
import glob
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


class VCTKDataset(Dataset):
    """
    VCTK (CSTR-Edinburgh) Dataset for JUND-F0 training.

    The VCTK corpus contains speech data from 110 English speakers with
    various accents. Each speaker reads out about 400 sentences selected
    from a newspaper, the rainbow passage, and an elicitation paragraph.

    This dataset class:
    1. Downloads VCTK if not present
    2. Generates mel spectrograms on-the-fly
    3. Uses pre-computed F0 pseudo-labels (generated via pyin or CREPE)
    4. Provides V/UV labels derived from F0 labels
    5. Supports data augmentation
    """

    # VCTK download URL (Edinburgh DataShare)
    VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 50.0,
        f_max: float = 800.0,
        segment_length: int = 1024,
        split: str = "train",
        train_ratio: float = 0.9,
        speaker_list: Optional[List[str]] = None,
        use_augmentation: bool = True,
        label_method: str = "pyin",
        regenerate_labels: bool = False,
    ):
        """
        Args:
            root_dir: Root directory for VCTK dataset
            sample_rate: Target sample rate (will resample if needed)
            n_fft: FFT size for mel spectrogram
            hop_length: Hop length for mel spectrogram
            n_mels: Number of mel bins
            f_min: Minimum frequency for mel spectrogram
            f_max: Maximum frequency for mel spectrogram
            segment_length: Number of frames per training sample
            split: 'train' or 'val'
            train_ratio: Ratio of data for training (rest for validation)
            speaker_list: Optional list of speaker IDs to use
            use_augmentation: Whether to apply data augmentation
            label_method: Method for F0 label generation ('pyin' or 'crepe')
            regenerate_labels: Whether to regenerate F0 labels even if cached
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.segment_length = segment_length
        self.split = split
        self.train_ratio = train_ratio
        self.use_augmentation = use_augmentation and (split == "train")
        self.label_method = label_method
        self.regenerate_labels = regenerate_labels

        # Audio directory
        self.audio_dir = self.root_dir / "wav48_silence_trimmed"
        self.label_dir = self.root_dir / "f0_labels"
        self.label_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset if needed
        self._download_if_needed()

        # Find all audio files
        self.file_list = self._scan_files(speaker_list)

        # Split into train/val
        self.file_list = self._split_data(self.file_list, split, train_ratio)

        # Generate F0 labels if needed
        self._generate_labels_if_needed()

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        logger.info(
            f"VCTK {split} dataset: {len(self.file_list)} files, "
            f"{len(set(f['speaker'] for f in self.file_list))} speakers"
        )

    def _download_if_needed(self):
        """Download VCTK corpus if not found.

        Tries three methods in order:
        1. HuggingFace datasets library (most reliable, no zip needed)
        2. Direct zip download from Edinburgh DataShare
        3. Kaggle API (if KAGGLE_USERNAME/KAGGLE_KEY set)
        """
        if self.audio_dir.exists() and any(self.audio_dir.iterdir()):
            logger.info(f"VCTK found at {self.audio_dir}")
            return

        logger.info("VCTK not found. Downloading...")

        # Method 1: HuggingFace datasets library (recommended)
        try:
            logger.info("Trying HuggingFace datasets library...")
            self._download_via_hf_datasets()
            if self.audio_dir.exists() and any(self.audio_dir.iterdir()):
                logger.info("VCTK downloaded via HuggingFace datasets.")
                return
        except Exception as e:
            logger.warning(f"HF datasets download failed: {e}")

        # Method 2: Direct zip download from Edinburgh DataShare
        zip_path = self.root_dir / "VCTK-Corpus-0.92.zip"
        try:
            logger.info("Trying direct download from Edinburgh DataShare...")
            if not zip_path.exists():
                subprocess.run(
                    ["wget", "-q", "--show-progress", self.VCTK_URL, "-O", str(zip_path)],
                    check=True,
                    timeout=600,
                )

            # Validate zip file before extracting
            import zipfile as zf_module
            if zf_module.is_zipfile(zip_path):
                logger.info("Extracting VCTK...")
                with zf_module.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.root_dir)
                logger.info("VCTK extraction complete.")
                return
            else:
                logger.warning("Downloaded file is not a valid zip. Removing...")
                zip_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Direct download failed: {e}")
            zip_path.unlink(missing_ok=True)

        # Method 3: Kaggle
        try:
            logger.info("Trying Kaggle download...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", "kynthesis/vctk-corpus",
                 "-p", str(self.root_dir), "--unzip"],
                check=True,
                timeout=600,
            )
            if self.audio_dir.exists() and any(self.audio_dir.iterdir()):
                logger.info("VCTK downloaded via Kaggle.")
                return
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")

        raise RuntimeError(
            "All download methods failed. Please download VCTK manually:\n"
            "  1. pip install datasets && python -c \"from datasets import load_dataset; load_dataset('CSTR-Edinburgh/vctk')\"\n"
            "  2. Download from https://datashare.ed.ac.uk/handle/10283/3443\n"
            "  3. Download from https://www.kaggle.com/datasets/kynthesis/vctk-corpus\n"
            f"Then extract to {self.root_dir} so that {self.audio_dir}/ exists."
        )

    def _download_via_hf_datasets(self):
        """Download VCTK using the HuggingFace datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            subprocess.run(
                ["pip", "install", "-q", "datasets"],
                check=True,
            )
            from datasets import load_dataset

        import soundfile as sf
        from tqdm.auto import tqdm

        try:
            # datasets >= 3.0: use parquet revision (script-based datasets removed)
            ds = load_dataset(
                "CSTR-Edinburgh/vctk",
                split="train",
                streaming=False,
                revision="refs/convert/parquet",
            )
        except Exception:
            # datasets < 3.0: use trust_remote_code for script-based loading
            ds = load_dataset(
                "CSTR-Edinburgh/vctk",
                split="train",
                streaming=False,
                trust_remote_code=True,
            )
        logger.info(f"HF dataset loaded: {len(ds)} samples")

        self.audio_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for item in tqdm(ds, total=len(ds), desc="Saving VCTK audio"):
            # Extract speaker ID
            speaker_id = item.get("speaker_id", item.get("id", "unknown"))
            if isinstance(speaker_id, int):
                speaker_id = f"p{speaker_id}"

            speaker_dir = self.audio_dir / str(speaker_id)
            speaker_dir.mkdir(parents=True, exist_ok=True)

            # Extract file ID
            file_id = item.get("file", item.get("id", f"{saved:04d}"))
            if isinstance(file_id, str) and "/" in file_id:
                file_id = Path(file_id).stem

            out_path = speaker_dir / f"{file_id}.flac"

            if "audio" in item and item["audio"] is not None:
                audio_data = item["audio"]
                wav = audio_data.get("array")
                sr = audio_data.get("sampling_rate", 48000)
                if wav is not None:
                    sf.write(str(out_path), wav, sr)
                    saved += 1

        logger.info(f"Saved {saved} audio files to {self.audio_dir}")

    def _scan_files(self, speaker_list: Optional[List[str]] = None) -> List[Dict]:
        """Scan audio directory and build file list."""
        file_list = []
        if not self.audio_dir.exists():
            raise RuntimeError(f"Audio directory not found: {self.audio_dir}")

        for speaker_dir in sorted(self.audio_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name  # e.g., "p225"
            if speaker_list and speaker_id not in speaker_list:
                continue

            for audio_file in sorted(speaker_dir.glob("*.flac")):
                file_list.append({
                    "audio_path": str(audio_file),
                    "speaker": speaker_id,
                    "utt_id": audio_file.stem,
                })
            # Also check for .wav files
            for audio_file in sorted(speaker_dir.glob("*.wav")):
                if not any(f["utt_id"] == audio_file.stem for f in file_list):
                    file_list.append({
                        "audio_path": str(audio_file),
                        "speaker": speaker_id,
                        "utt_id": audio_file.stem,
                    })

        if not file_list:
            raise RuntimeError(f"No audio files found in {self.audio_dir}")

        return file_list

    def _split_data(
        self, file_list: List[Dict], split: str, train_ratio: float
    ) -> List[Dict]:
        """Split data into train/val by speaker to avoid data leakage."""
        speakers = sorted(set(f["speaker"] for f in file_list))
        n_train = int(len(speakers) * train_ratio)
        train_speakers = set(speakers[:n_train])
        val_speakers = set(speakers[n_train:])

        if split == "train":
            return [f for f in file_list if f["speaker"] in train_speakers]
        else:
            return [f for f in file_list if f["speaker"] in val_speakers]

    def _generate_labels_if_needed(self):
        """Generate F0 pseudo-labels using pyin or CREPE."""
        total = len(self.file_list)
        missing = 0
        for f in self.file_list:
            label_path = self._get_label_path(f["utt_id"])
            if not label_path.exists() or self.regenerate_labels:
                missing += 1

        if missing == 0:
            logger.info("All F0 labels already cached.")
            return

        logger.info(f"Generating F0 labels for {missing}/{total} files using {self.label_method}...")

        for i, f in enumerate(self.file_list):
            label_path = self._get_label_path(f["utt_id"])
            if label_path.exists() and not self.regenerate_labels:
                continue

            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{total}")

            try:
                self._generate_single_label(f["audio_path"], label_path)
            except Exception as e:
                logger.warning(f"Failed to generate label for {f['utt_id']}: {e}")

        logger.info("F0 label generation complete.")

    def _get_label_path(self, utt_id: str) -> Path:
        """Get path for cached F0 label file."""
        return self.label_dir / f"{utt_id}.npy"

    def _generate_single_label(self, audio_path: str, label_path: Path):
        """Generate F0 label for a single audio file."""
        import librosa

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        if self.label_method == "pyin":
            f0, voiced_flags, _ = librosa.pyin(
                y,
                fmin=self.f_min,
                fmax=self.f_max,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                frame_length=self.n_fft,
            )
        elif self.label_method == "crepe":
            import crepe
            _, f0, voiced_flags, _ = crepe.predict(
                y, self.sample_rate,
                step_size=int(self.hop_length / self.sample_rate * 1000),
                viterbi=True,
            )
        else:
            raise ValueError(f"Unknown label method: {self.label_method}")

        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)

        # Save
        np.save(label_path, f0.astype(np.float32))

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio."""
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Normalize
        waveform = waveform / (waveform.abs().max().clamp(min=1e-8))

        return waveform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - mel: (n_mels, time) mel spectrogram
                - vuv: (time,) voiced/unvoiced label
                - f0: (time,) F0 label in Hz
                - utt_id: str
        """
        info = self.file_list[idx]

        # Load audio
        waveform = self._load_audio(info["audio_path"])  # (1, samples)

        # Apply augmentation
        if self.use_augmentation:
            waveform = self._augment(waveform)

        # Compute mel spectrogram
        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        mel = self.amp_to_db(mel)  # (1, n_mels, time)
        mel = mel.squeeze(0)  # (n_mels, time)

        # Load F0 labels
        label_path = self._get_label_path(info["utt_id"])
        if label_path.exists():
            f0 = np.load(label_path)
        else:
            # Fallback: zeros
            n_frames = mel.shape[1]
            f0 = np.zeros(n_frames, dtype=np.float32)

        # Align F0 length with mel spectrogram
        n_frames = mel.shape[1]
        f0 = self._align_length(f0, n_frames)

        # Create V/UV label from F0
        vuv = (f0 > 0).astype(np.float32)

        # Convert to tensors
        mel = torch.from_numpy(mel).float()
        vuv = torch.from_numpy(vuv).float()
        f0 = torch.from_numpy(f0).float()

        # Pad or trim to segment_length
        mel, vuv, f0 = self._pad_or_trim(mel, vuv, f0)

        return {
            "mel": mel,
            "vuv": vuv,
            "f0": f0,
            "utt_id": info["utt_id"],
        }

    def _align_length(self, f0: np.ndarray, target_len: int) -> np.ndarray:
        """Align F0 array length to target length."""
        if len(f0) == target_len:
            return f0
        elif len(f0) > target_len:
            return f0[:target_len]
        else:
            # Pad with zeros
            return np.pad(f0, (0, target_len - len(f0)), mode="constant")

    def _pad_or_trim(
        self,
        mel: torch.Tensor,
        vuv: torch.Tensor,
        f0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad or trim sequences to segment_length frames."""
        n_frames = mel.shape[1]

        if n_frames >= self.segment_length:
            # Random crop for training, center crop for validation
            if self.split == "train":
                start = torch.randint(0, n_frames - self.segment_length + 1, (1,)).item()
            else:
                start = (n_frames - self.segment_length) // 2

            mel = mel[:, start:start + self.segment_length]
            vuv = vuv[start:start + self.segment_length]
            f0 = f0[start:start + self.segment_length]
        else:
            # Pad
            pad_len = self.segment_length - n_frames
            mel = F.pad(mel, (0, pad_len))
            vuv = F.pad(vuv, (0, pad_len))
            f0 = F.pad(f0, (0, pad_len))

        return mel, vuv, f0

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to waveform."""
        # 1. Random gain
        if torch.rand(1).item() < 0.5:
            gain = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            waveform = waveform * gain

        # 2. Add background noise
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        # 3. Time masking (simulate dropped frames)
        if torch.rand(1).item() < 0.2:
            n_samples = waveform.shape[1]
            mask_len = int(n_samples * 0.05)  # 5% of signal
            mask_start = torch.randint(0, max(1, n_samples - mask_len), (1,)).item()
            waveform[:, mask_start:mask_start + mask_len] = 0

        return waveform


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    mels = torch.stack([item["mel"] for item in batch])
    vuvs = torch.stack([item["vuv"] for item in batch])
    f0s = torch.stack([item["f0"] for item in batch])
    utt_ids = [item["utt_id"] for item in batch]

    return {
        "mel": mels,
        "vuv": vuvs.unsqueeze(-1),  # (batch, time, 1)
        "f0": f0s.unsqueeze(-1),    # (batch, time, 1)
        "utt_ids": utt_ids,
    }


def create_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    segment_length: int = 512,
    sample_rate: int = 16000,
    n_mels: int = 80,
    hop_length: int = 160,
    label_method: str = "pyin",
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for VCTK dataset."""

    train_dataset = VCTKDataset(
        root_dir=root_dir,
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        segment_length=segment_length,
        split="train",
        use_augmentation=use_augmentation,
        label_method=label_method,
    )

    val_dataset = VCTKDataset(
        root_dir=root_dir,
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        segment_length=segment_length,
        split="val",
        use_augmentation=False,
        label_method=label_method,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
