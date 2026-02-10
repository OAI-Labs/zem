"""
Unit tests for core models and data structures.
"""

import pytest
import numpy as np

from xfmr_zem.audio.core.models import (
    DiarizationSegment,
    AudioSegment,
    TranscriptionResult,
    SpeakerTurn,
    MergedTranscript,
)


class TestDiarizationSegment:
    """Tests for DiarizationSegment dataclass."""
    
    def test_creation(self):
        """Test basic segment creation."""
        segment = DiarizationSegment(
            start=1.0,
            end=5.0,
            speaker_id="speaker_0",
        )
        assert segment.start == 1.0
        assert segment.end == 5.0
        assert segment.speaker_id == "speaker_0"
        assert segment.is_overlapping is False
    
    def test_duration(self):
        """Test duration property."""
        segment = DiarizationSegment(start=2.5, end=7.5, speaker_id="speaker_0")
        assert segment.duration == 5.0
    
    def test_overlapping_flag(self):
        """Test overlapping segment creation."""
        segment = DiarizationSegment(
            start=1.0,
            end=3.0,
            speaker_id="speaker_0",
            is_overlapping=True,
        )
        assert segment.is_overlapping is True
    
    def test_invalid_times_raises_error(self):
        """Test that invalid times raise ValueError."""
        with pytest.raises(ValueError):
            DiarizationSegment(start=5.0, end=3.0, speaker_id="speaker_0")
    
    def test_negative_start_raises_error(self):
        """Test that negative start raises ValueError."""
        with pytest.raises(ValueError):
            DiarizationSegment(start=-1.0, end=3.0, speaker_id="speaker_0")


class TestAudioSegment:
    """Tests for AudioSegment dataclass."""
    
    def test_creation(self):
        """Test basic audio segment creation."""
        audio_data = np.zeros(16000, dtype=np.float32)
        original = DiarizationSegment(start=0.0, end=1.0, speaker_id="speaker_0")
        
        segment = AudioSegment(
            audio_data=audio_data,
            sample_rate=16000,
            original_segment=original,
            start_with_padding=0.0,
            end_with_padding=1.2,
        )
        
        assert segment.sample_rate == 16000
        assert segment.speaker_id == "speaker_0"
    
    def test_duration(self):
        """Test duration calculation."""
        audio_data = np.zeros(32000, dtype=np.float32)  # 2 seconds at 16kHz
        original = DiarizationSegment(start=0.0, end=2.0, speaker_id="speaker_0")
        
        segment = AudioSegment(
            audio_data=audio_data,
            sample_rate=16000,
            original_segment=original,
            start_with_padding=0.0,
            end_with_padding=2.2,
        )
        
        assert segment.duration == 2.0


class TestSpeakerTurn:
    """Tests for SpeakerTurn dataclass."""
    
    def test_str_format(self):
        """Test string representation."""
        turn = SpeakerTurn(
            speaker_id="speaker_0",
            speaker_label="Speaker 1",
            text="Xin chào mọi người",
            start_time=0.0,
            end_time=2.5,
        )
        
        assert str(turn) == "Speaker 1: Xin chào mọi người"


class TestMergedTranscript:
    """Tests for MergedTranscript dataclass."""
    
    @pytest.fixture
    def sample_transcript(self):
        """Create a sample transcript for testing."""
        turns = [
            SpeakerTurn("s0", "Speaker 1", "Xin chào", 0.0, 1.0),
            SpeakerTurn("s1", "Speaker 2", "Chào bạn", 1.5, 2.5),
            SpeakerTurn("s0", "Speaker 1", "Bạn khỏe không?", 3.0, 4.5),
        ]
        return MergedTranscript(
            turns=turns,
            total_duration=5.0,
            speaker_count=2,
        )
    
    def test_to_text(self, sample_transcript):
        """Test text output."""
        text = sample_transcript.to_text()
        lines = text.split("\n")
        
        assert len(lines) == 3
        assert "Speaker 1: Xin chào" in lines[0]
        assert "Speaker 2: Chào bạn" in lines[1]
    
    def test_to_text_with_timestamps(self, sample_transcript):
        """Test text output with timestamps."""
        text = sample_transcript.to_text(include_timestamps=True)
        
        assert "[0.0s - 1.0s]" in text
        assert "[1.5s - 2.5s]" in text
    
    def test_to_srt(self, sample_transcript):
        """Test SRT output."""
        srt = sample_transcript.to_srt()
        
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:01,000" in srt
        assert "Speaker 1: Xin chào" in srt
    
    def test_to_json(self, sample_transcript):
        """Test JSON output."""
        data = sample_transcript.to_json()
        
        assert data["total_duration"] == 5.0
        assert data["speaker_count"] == 2
        assert len(data["turns"]) == 3
        assert data["turns"][0]["speaker_label"] == "Speaker 1"
