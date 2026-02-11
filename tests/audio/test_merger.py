"""
Unit tests for the merger module.
"""

import pytest
import numpy as np

from xfmr_zem.audio.core.models import (
    DiarizationSegment,
    AudioSegment,
    TranscriptionResult,
)
from xfmr_zem.audio.components.merging.merger import TranscriptMerger


class TestTranscriptMerger:
    """Tests for TranscriptMerger."""
    
    @pytest.fixture
    def merger(self):
        """Create a merger instance."""
        return TranscriptMerger(merge_consecutive=True, gap_threshold=1.0)
    
    @pytest.fixture
    def sample_transcriptions(self):
        """Create sample transcription results."""
        results = []
        
        # Create 3 transcriptions from 2 speakers
        segments_data = [
            (0.0, 2.0, "speaker_0", "Xin chào mọi người"),
            (2.5, 4.0, "speaker_1", "Chào bạn nhé"),
            (4.5, 6.0, "speaker_0", "Hôm nay thế nào"),
        ]
        
        for start, end, speaker, text in segments_data:
            diar_seg = DiarizationSegment(start=start, end=end, speaker_id=speaker)
            audio_seg = AudioSegment(
                audio_data=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                original_segment=diar_seg,
                start_with_padding=max(0, start - 0.2),
                end_with_padding=end + 0.2,
            )
            results.append(TranscriptionResult(text=text, segment=audio_seg))
        
        return results
    
    def test_merge_basic(self, merger, sample_transcriptions):
        """Test basic merging."""
        diar_segments = [r.segment.original_segment for r in sample_transcriptions]
        merged = merger.merge(diar_segments, sample_transcriptions)
        
        assert merged.speaker_count == 2
        assert len(merged.turns) == 3
    
    def test_speaker_labels(self, merger, sample_transcriptions):
        """Test speaker label assignment."""
        diar_segments = [r.segment.original_segment for r in sample_transcriptions]
        merged = merger.merge(diar_segments, sample_transcriptions)
        
        # First speaker should be "Speaker 1"
        assert merged.turns[0].speaker_label == "Speaker 1"
        # Second speaker should be "Speaker 2"
        assert merged.turns[1].speaker_label == "Speaker 2"
    
    def test_chronological_order(self, merger, sample_transcriptions):
        """Test turns are ordered chronologically."""
        # Shuffle the input
        shuffled = sample_transcriptions[::-1]
        diar_segments = [r.segment.original_segment for r in shuffled]
        
        merged = merger.merge(diar_segments, shuffled)
        
        # Should still be sorted by start time
        for i in range(len(merged.turns) - 1):
            assert merged.turns[i].start_time <= merged.turns[i + 1].start_time
    
    def test_merge_consecutive_same_speaker(self, merger):
        """Test consecutive segments from same speaker are merged."""
        # Create consecutive segments from same speaker with small gap
        segments_data = [
            (0.0, 1.0, "speaker_0", "Xin chào"),
            (1.3, 2.0, "speaker_0", "mọi người"),  # Gap = 0.3s < 1.0s threshold
        ]
        
        transcriptions = []
        for start, end, speaker, text in segments_data:
            diar_seg = DiarizationSegment(start=start, end=end, speaker_id=speaker)
            audio_seg = AudioSegment(
                audio_data=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                original_segment=diar_seg,
                start_with_padding=start,
                end_with_padding=end,
            )
            transcriptions.append(TranscriptionResult(text=text, segment=audio_seg))
        
        diar_segments = [r.segment.original_segment for r in transcriptions]
        merged = merger.merge(diar_segments, transcriptions)
        
        # Should be merged into 1 turn
        assert len(merged.turns) == 1
        assert "Xin chào mọi người" in merged.turns[0].text
    
    def test_no_merge_different_speakers(self, merger):
        """Test segments from different speakers are not merged."""
        segments_data = [
            (0.0, 1.0, "speaker_0", "Xin chào"),
            (1.1, 2.0, "speaker_1", "Chào bạn"),  # Different speaker
        ]
        
        transcriptions = []
        for start, end, speaker, text in segments_data:
            diar_seg = DiarizationSegment(start=start, end=end, speaker_id=speaker)
            audio_seg = AudioSegment(
                audio_data=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                original_segment=diar_seg,
                start_with_padding=start,
                end_with_padding=end,
            )
            transcriptions.append(TranscriptionResult(text=text, segment=audio_seg))
        
        diar_segments = [r.segment.original_segment for r in transcriptions]
        merged = merger.merge(diar_segments, transcriptions)
        
        # Should remain 2 separate turns
        assert len(merged.turns) == 2
    
    def test_empty_input(self, merger):
        """Test handling empty input."""
        merged = merger.merge([], [])
        
        assert merged.speaker_count == 0
        assert len(merged.turns) == 0
        assert merged.total_duration == 0.0
