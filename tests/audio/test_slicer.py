"""
Unit tests for the audio slicer module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

from xfmr_zem.audio.core.models import DiarizationSegment
from xfmr_zem.audio.components.slicer.slicer import AudioSlicer
from xfmr_zem.audio.config import SlicerConfig


class TestAudioSlicer:
    """Tests for AudioSlicer."""
    
    @pytest.fixture
    def slicer(self):
        """Create a slicer instance."""
        config = SlicerConfig(
            collar_padding=0.2,
            min_segment_duration=0.5,
            max_segment_duration=10.0,
        )
        # Use simple slicer configuration without complex dependencies for testing
        return AudioSlicer(config=config)
    
    @pytest.fixture
    def sample_audio_path(self):
        """Create a sample audio file for testing."""
        # Create 5 seconds of audio at 16kHz
        sr = 16000
        duration = 5.0
        samples = int(sr * duration)
        
        # Generate a simple sine wave
        t = np.linspace(0, duration, samples, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield Path(f.name)
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_slice_single_segment(self, slicer, sample_audio_path):
        """Test slicing a single segment."""
        segments = [
            DiarizationSegment(start=1.0, end=3.0, speaker_id="speaker_0"),
        ]
        
        audio_segments = slicer.slice(sample_audio_path, segments)
        
        assert len(audio_segments) == 1
        seg = audio_segments[0]
        
        # Check padding was applied
        assert seg.start_with_padding == 0.8  # 1.0 - 0.2
        assert seg.end_with_padding == 3.2    # 3.0 + 0.2
        
        # Check audio duration matches
        expected_duration = 3.2 - 0.8  # 2.4 seconds
        assert abs(seg.duration - expected_duration) < 0.1
    
    def test_slice_multiple_segments(self, slicer, sample_audio_path):
        """Test slicing multiple segments."""
        segments = [
            DiarizationSegment(start=0.5, end=1.5, speaker_id="speaker_0"),
            DiarizationSegment(start=2.0, end=3.5, speaker_id="speaker_1"),
        ]
        
        audio_segments = slicer.slice(sample_audio_path, segments)
        
        assert len(audio_segments) == 2
        assert audio_segments[0].speaker_id == "speaker_0"
        assert audio_segments[1].speaker_id == "speaker_1"
    
    def test_padding_at_start(self, slicer, sample_audio_path):
        """Test padding doesn't go below 0."""
        segments = [
            DiarizationSegment(start=0.1, end=1.0, speaker_id="speaker_0"),
        ]
        
        audio_segments = slicer.slice(sample_audio_path, segments)
        
        # Padding should be clamped to 0
        assert audio_segments[0].start_with_padding == 0.0
    
    def test_padding_at_end(self, slicer, sample_audio_path):
        """Test padding doesn't exceed audio duration."""
        segments = [
            DiarizationSegment(start=4.0, end=4.9, speaker_id="speaker_0"),
        ]
        
        audio_segments = slicer.slice(sample_audio_path, segments)
        
        # Padding should be clamped to audio duration (5.0s)
        assert audio_segments[0].end_with_padding == 5.0
    
    def test_skip_short_segment(self, slicer, sample_audio_path):
        """Test that short segments are skipped."""
        segments = [
            DiarizationSegment(start=1.0, end=1.05, speaker_id="speaker_0"),  # 0.05s duration + 0.4s padding = 0.45s < 0.5s
        ]
        
        audio_segments = slicer.slice(sample_audio_path, segments)
        
        # Should be skipped (duration < min_segment_duration)
        assert len(audio_segments) == 0
    
    def test_split_long_segment(self, sample_audio_path):
        """Test that long segments are split."""
        config = SlicerConfig(
            collar_padding=0.2,
            min_segment_duration=0.5,
            max_segment_duration=2.0,  # Short max for testing
        )
        slicer = AudioSlicer(config=config)
        
        segments = [
            DiarizationSegment(start=0.0, end=4.5, speaker_id="speaker_0"),
        ]
        
        audio_segments = slicer.slice(sample_audio_path, segments)
        
        # Should be split into multiple segments
        assert len(audio_segments) > 1
    
    def test_cache_cleared(self, slicer, sample_audio_path):
        """Test cache clearing."""
        segments = [
            DiarizationSegment(start=1.0, end=2.0, speaker_id="speaker_0"),
        ]
        
        # First slice loads audio into cache
        slicer.slice(sample_audio_path, segments)
        assert len(slicer._audio_cache) > 0
        
        # Clear cache
        slicer.clear_cache()
        assert len(slicer._audio_cache) == 0
