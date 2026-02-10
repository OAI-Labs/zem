"""
Audio slicing module with padding collar technique.

Cuts audio into segments based on diarization results.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import librosa

from ...core.interfaces import IAudioSlicer
from ...core.models import DiarizationSegment, AudioSegment
from ...config import SlicerConfig

logger = logging.getLogger(__name__)


class AudioSlicer(IAudioSlicer):
    """Slice audio with padding collar technique.
    
    Padding collar is crucial for Vietnamese ASR because:
    - Vietnamese consonants at the start of words may have low energy
    - Tonal endings are often quiet and can be cut off
    - Adding 0.2s padding ensures complete phoneme capture
    """
    
    def __init__(self, config: Optional[SlicerConfig] = None):
        """Initialize audio slicer.
        
        Args:
            config: Slicer configuration
        """
        self.config = config or SlicerConfig()
        self._audio_cache = {}  # Cache loaded audio
    
    def slice(
        self, 
        audio_path: Path, 
        segments: List[DiarizationSegment]
    ) -> List[AudioSegment]:
        """Slice audio into segments with padding collar.
        
        Args:
            audio_path: Path to audio file
            segments: Diarization segments with timestamps
            
        Returns:
            List of AudioSegment with padding applied
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        audio, sr = self._load_audio(audio_path)
        audio_duration = len(audio) / sr
        
        logger.info(
            f"Slicing {len(segments)} segments from {audio_duration:.2f}s audio "
            f"with {self.config.collar_padding}s padding"
        )
        
        audio_segments = []
        
        for seg in segments:
            # Apply padding collar
            start_with_padding = max(0, seg.start - self.config.collar_padding)
            end_with_padding = min(audio_duration, seg.end + self.config.collar_padding)
            
            # Skip segments that are too short
            duration = end_with_padding - start_with_padding
            if duration < self.config.min_segment_duration:
                logger.debug(
                    f"Skipping segment {seg.speaker_id} "
                    f"[{seg.start:.2f}-{seg.end:.2f}]: too short ({duration:.2f}s)"
                )
                continue
            
            # Handle segments that are too long by splitting
            if duration > self.config.max_segment_duration:
                sub_segments = self._split_long_segment(
                    audio, sr, seg, start_with_padding, end_with_padding
                )
                audio_segments.extend(sub_segments)
            else:
                # Extract audio slice
                audio_slice = self._extract_slice(
                    audio, sr, start_with_padding, end_with_padding
                )
                
                audio_segments.append(AudioSegment(
                    audio_data=audio_slice,
                    sample_rate=sr,
                    original_segment=seg,
                    start_with_padding=start_with_padding,
                    end_with_padding=end_with_padding,
                ))
        
        logger.info(f"Created {len(audio_segments)} audio segments")
        return audio_segments
    
    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa.
        
        Uses caching to avoid reloading the same file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        cache_key = str(audio_path)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        
        # Load audio (should already be preprocessed to 16kHz mono)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        self._audio_cache[cache_key] = (audio, sr)
        
        return audio, sr
    
    def _extract_slice(
        self,
        audio: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float,
    ) -> np.ndarray:
        """Extract audio slice by time.
        
        Args:
            audio: Full audio array
            sr: Sample rate
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio slice as numpy array
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        return audio[start_sample:end_sample].copy()
    
    def _split_long_segment(
        self,
        audio: np.ndarray,
        sr: int,
        original_segment: DiarizationSegment,
        start_with_padding: float,
        end_with_padding: float,
    ) -> List[AudioSegment]:
        """Split a long segment into smaller chunks.
        
        Uses overlap to ensure continuity.
        
        Args:
            audio: Full audio array
            sr: Sample rate
            original_segment: Original diarization segment
            start_with_padding: Padded start time
            end_with_padding: Padded end time
            
        Returns:
            List of AudioSegment chunks
        """
        segments = []
        current_start = start_with_padding
        chunk_duration = self.config.max_segment_duration
        overlap = self.config.collar_padding * 2  # Use collar as overlap
        
        while current_start < end_with_padding:
            current_end = min(current_start + chunk_duration, end_with_padding)
            
            # Extract audio slice
            audio_slice = self._extract_slice(audio, sr, current_start, current_end)
            
            # Create sub-segment with adjusted timestamps
            sub_original = DiarizationSegment(
                start=max(original_segment.start, current_start + self.config.collar_padding),
                end=min(original_segment.end, current_end - self.config.collar_padding),
                speaker_id=original_segment.speaker_id,
                is_overlapping=original_segment.is_overlapping,
            )
            
            segments.append(AudioSegment(
                audio_data=audio_slice,
                sample_rate=sr,
                original_segment=sub_original,
                start_with_padding=current_start,
                end_with_padding=current_end,
            ))
            
            # Move to next chunk with overlap
            current_start = current_end - overlap
            
            # Avoid infinite loop on very short remaining segments
            if current_end >= end_with_padding:
                break
        
        logger.debug(
            f"Split long segment [{start_with_padding:.2f}-{end_with_padding:.2f}] "
            f"into {len(segments)} chunks"
        )
        
        return segments
    
    def clear_cache(self) -> None:
        """Clear audio cache to free memory."""
        self._audio_cache.clear()
        logger.debug("Audio cache cleared")
