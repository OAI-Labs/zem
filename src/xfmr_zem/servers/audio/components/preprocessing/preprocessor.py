"""
Audio preprocessing module.

Converts audio to optimal format for DiariZen diarization.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import tempfile

import numpy as np
import librosa
import soundfile as sf

from ...core.interfaces import IPreprocessor
from ...core.models import PreprocessingConfig

logger = logging.getLogger(__name__)


class AudioPreprocessor(IPreprocessor):
    """Preprocess audio for optimal DiariZen performance.
    
    Features:
    - Resample to 16kHz (optimal for DiariZen)
    - Convert stereo to mono
    - Optional noise reduction
    - Optional loudness normalization
    - Format conversion (MP3/etc to WAV)
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        """Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
            output_dir: Directory for output files (uses temp if not provided)
        """
        self.config = config or PreprocessingConfig()
        self.output_dir = output_dir
        
        # Lazy import for optional dependencies
        self._noisereduce = None
        self._pyloudnorm = None
    
    def preprocess(self, audio_path: Path) -> Path:
        """Preprocess audio file for optimal diarization performance.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file (16kHz mono WAV)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Preprocessing audio: {audio_path}")
        
        # Load audio
        audio, sr = self._load_audio(audio_path)
        logger.debug(f"Loaded audio: {len(audio)/sr:.2f}s, sample_rate={sr}")
        
        # Convert to mono if needed
        if self.config.convert_to_mono and audio.ndim > 1:
            audio = self._to_mono(audio)
            logger.debug("Converted to mono")
        
        # Resample if needed
        if sr != self.config.target_sample_rate:
            audio = self._resample(audio, sr, self.config.target_sample_rate)
            sr = self.config.target_sample_rate
            logger.debug(f"Resampled to {sr}Hz")
        
        # Apply noise reduction if enabled
        if self.config.enable_noise_reduction:
            audio = self._reduce_noise(audio, sr)
            logger.debug("Applied noise reduction")
        
        # Normalize loudness if enabled
        if self.config.enable_normalization:
            audio = self._normalize_loudness(audio, sr)
            logger.debug("Normalized loudness")
        
        # Save preprocessed audio
        output_path = self._get_output_path(audio_path)
        self._save_audio(audio, sr, output_path)
        logger.info(f"Saved preprocessed audio: {output_path}")
        
        return output_path
    
    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Load with original sample rate
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        return audio, sr
    
    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo/multi-channel audio to mono.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            
        Returns:
            Mono audio data (samples,)
        """
        if audio.ndim == 1:
            return audio
        # Average across channels
        return np.mean(audio, axis=0)
    
    def _resample(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate.
        
        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio data
        """
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction using noisereduce library.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Noise-reduced audio
        """
        try:
            if self._noisereduce is None:
                import noisereduce as nr
                self._noisereduce = nr
            
            # Apply stationary noise reduction
            reduced = self._noisereduce.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=self.config.noise_reduction_strength,
                stationary=True,
            )
            return reduced
        except ImportError:
            logger.warning("noisereduce not installed, skipping noise reduction")
            return audio
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}, skipping")
            return audio
    
    def _normalize_loudness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio loudness using pyloudnorm.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Loudness-normalized audio
        """
        try:
            if self._pyloudnorm is None:
                import pyloudnorm as pyln
                self._pyloudnorm = pyln
            
            # Measure current loudness
            meter = self._pyloudnorm.Meter(sr)
            current_loudness = meter.integrated_loudness(audio)
            
            # Skip if audio is too quiet (likely silence)
            if current_loudness < -70:
                logger.debug("Audio too quiet for normalization")
                return audio
            
            # Normalize to target loudness
            normalized = self._pyloudnorm.normalize.loudness(
                audio, 
                current_loudness, 
                self.config.target_loudness_lufs
            )
            return normalized
        except ImportError:
            logger.warning("pyloudnorm not installed, skipping normalization")
            return audio
        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}, skipping")
            return audio
    
    def _get_output_path(self, input_path: Path) -> Path:
        """Generate output path for preprocessed audio.
        
        Args:
            input_path: Original input file path
            
        Returns:
            Path for preprocessed output file
        """
        if self.output_dir:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.gettempdir()) / "audio_transcribe"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        stem = input_path.stem
        output_path = output_dir / f"{stem}_preprocessed.wav"
        
        return output_path
    
    def _save_audio(
        self, 
        audio: np.ndarray, 
        sr: int, 
        output_path: Path
    ) -> None:
        """Save audio to WAV file.
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Output file path
        """
        # Ensure audio is in correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Clip to prevent clipping artifacts
        audio = np.clip(audio, -1.0, 1.0)
        
        sf.write(output_path, audio, sr, subtype='PCM_16')
