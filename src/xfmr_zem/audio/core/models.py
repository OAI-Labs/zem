"""
Data models for Audio Transcribe & Diarization system.

Uses dataclasses for clean, immutable data structures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class DiarizationSegment:
    """Represents a speaker segment from diarization.
    
    Attributes:
        start: Start time in seconds
        end: End time in seconds
        speaker_id: Unique identifier for the speaker
        is_overlapping: True if this segment overlaps with another speaker
    """
    start: float
    end: float
    speaker_id: str
    is_overlapping: bool = False
    
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start
    
    def __post_init__(self):
        """Validate segment data."""
        if self.start < 0:
            raise ValueError(f"Start time must be non-negative, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"End time ({self.end}) must be greater than start time ({self.start})")


@dataclass
class AudioSegment:
    """Represents a sliced audio segment with metadata.
    
    Attributes:
        audio_data: Raw audio data as numpy array
        sample_rate: Sample rate of the audio
        original_segment: The diarization segment this was sliced from
        start_with_padding: Actual start time including padding
        end_with_padding: Actual end time including padding
    """
    audio_data: np.ndarray
    sample_rate: int
    original_segment: DiarizationSegment
    start_with_padding: float
    end_with_padding: float
    
    @property
    def duration(self) -> float:
        """Duration of the audio segment in seconds."""
        return len(self.audio_data) / self.sample_rate
    
    @property
    def speaker_id(self) -> str:
        """Speaker ID from the original segment."""
        return self.original_segment.speaker_id


@dataclass
class TranscriptionResult:
    """Represents the transcription result for an audio segment.
    
    Attributes:
        text: Transcribed text
        segment: The audio segment that was transcribed
        confidence: Optional confidence score (0-1)
    """
    text: str
    segment: AudioSegment
    confidence: Optional[float] = None
    
    @property
    def speaker_id(self) -> str:
        """Speaker ID from the original segment."""
        return self.segment.speaker_id
    
    @property
    def start_time(self) -> float:
        """Original start time (without padding)."""
        return self.segment.original_segment.start
    
    @property
    def end_time(self) -> float:
        """Original end time (without padding)."""
        return self.segment.original_segment.end


@dataclass
class SpeakerTurn:
    """A single speaker turn in the merged transcript.
    
    Attributes:
        speaker_id: ID of the speaker
        speaker_label: Human-readable label (e.g., "Speaker 1")
        text: Transcribed text for this turn
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    speaker_id: str
    speaker_label: str
    text: str
    start_time: float
    end_time: float
    
    def __str__(self) -> str:
        """Format as 'Speaker X: [text]'."""
        return f"{self.speaker_label}: {self.text}"


@dataclass
class MergedTranscript:
    """Final merged transcript with speaker labels.
    
    Attributes:
        turns: List of speaker turns in chronological order
        total_duration: Total duration of the audio in seconds
        speaker_count: Number of unique speakers
    """
    turns: List[SpeakerTurn] = field(default_factory=list)
    total_duration: float = 0.0
    speaker_count: int = 0
    
    def to_text(self, include_timestamps: bool = False) -> str:
        """Convert to plain text format.
        
        Args:
            include_timestamps: If True, include timestamps for each turn
            
        Returns:
            Formatted transcript string
        """
        lines = []
        for turn in self.turns:
            if include_timestamps:
                timestamp = f"[{turn.start_time:.1f}s - {turn.end_time:.1f}s] "
                lines.append(f"{timestamp}{turn}")
            else:
                lines.append(str(turn))
        return "\n".join(lines)
    
    def to_srt(self) -> str:
        """Convert to SRT subtitle format.
        
        Returns:
            SRT formatted string
        """
        lines = []
        for i, turn in enumerate(self.turns, 1):
            start = self._format_srt_time(turn.start_time)
            end = self._format_srt_time(turn.end_time)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(f"{turn.speaker_label}: {turn.text}")
            lines.append("")
        return "\n".join(lines)
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def to_json(self) -> dict:
        """Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the transcript
        """
        return {
            "total_duration": self.total_duration,
            "speaker_count": self.speaker_count,
            "turns": [
                {
                    "speaker_id": turn.speaker_id,
                    "speaker_label": turn.speaker_label,
                    "text": turn.text,
                    "start_time": turn.start_time,
                    "end_time": turn.end_time,
                }
                for turn in self.turns
            ],
        }
