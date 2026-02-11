"""
Abstract interfaces for Audio Transcribe & Diarization system.

Defines contracts following Interface Segregation Principle (ISP).
All implementations must adhere to these interfaces.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import (
        DiarizationSegment,
        AudioSegment,
        TranscriptionResult,
        MergedTranscript,
    )


class IPreprocessor(ABC):
    """Interface for audio preprocessing.
    
    Responsible for converting audio to optimal format for diarization.
    """
    
    @abstractmethod
    def preprocess(self, audio_path: Path) -> Path:
        """Preprocess audio file for optimal diarization performance.
        
        Args:
            audio_path: Path to input audio file (.wav, .mp3, etc.)
            
        Returns:
            Path to preprocessed audio file (16kHz mono WAV)
        """
        pass


class IDiarizer(ABC):
    """Interface for speaker diarization.
    
    Identifies who spoke when in an audio file.
    """
    
    @abstractmethod
    def diarize(self, audio_path: Path) -> List["DiarizationSegment"]:
        """Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to preprocessed audio file
            
        Returns:
            List of segments with speaker IDs and timestamps.
            Overlapping speech creates multiple segments for same time range.
        """
        pass


class IAudioSlicer(ABC):
    """Interface for audio slicing with padding collar.
    
    Cuts audio into segments based on diarization results.
    """
    
    @abstractmethod
    def slice(
        self, 
        audio_path: Path, 
        segments: List["DiarizationSegment"]
    ) -> List["AudioSegment"]:
        """Slice audio into segments with padding collar.
        
        Args:
            audio_path: Path to audio file
            segments: Diarization segments with timestamps
            
        Returns:
            List of audio segments with padding applied.
            Padding is crucial for Vietnamese phonetics.
        """
        pass


class ITranscriber(ABC):
    """Interface for speech-to-text transcription.
    
    Converts audio segments to text using ASR.
    """
    
    @abstractmethod
    def transcribe(
        self, 
        segments: List["AudioSegment"]
    ) -> List["TranscriptionResult"]:
        """Transcribe audio segments to text.
        
        Args:
            segments: Audio segments to transcribe
            
        Returns:
            List of transcription results with text and metadata.
            Optimized for 3-10 second segments.
        """
        pass


class IMerger(ABC):
    """Interface for merging transcriptions with speaker labels.
    
    Combines transcription results with diarization information.
    """
    
    @abstractmethod
    def merge(
        self,
        diarization_segments: List["DiarizationSegment"],
        transcriptions: List["TranscriptionResult"],
    ) -> "MergedTranscript":
        """Merge transcriptions with speaker labels.
        
        Args:
            diarization_segments: Original diarization segments
            transcriptions: Transcription results for each segment
            
        Returns:
            Merged transcript with speaker labels in chronological order.
            Format: 'Speaker X: [content]'
        """
        pass
