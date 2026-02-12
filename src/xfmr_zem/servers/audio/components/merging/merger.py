"""
Transcript merging module.

Combines transcriptions with speaker labels in chronological order.
"""

from loguru import logger
from typing import List, Dict

from ...core.interfaces import IMerger
from ...core.models import (
    DiarizationSegment,
    TranscriptionResult,
    MergedTranscript,
    SpeakerTurn,
)


class TranscriptMerger(IMerger):
    """Merge transcriptions with speaker labels.
    
    Features:
    - Chronological ordering by timestamp
    - Speaker label assignment (Speaker 1, Speaker 2, etc.)
    - Handles overlapping segments
    - Merges consecutive segments from same speaker
    """
    
    def __init__(self, merge_consecutive: bool = True, gap_threshold: float = 1.0):
        """Initialize merger.
        
        Args:
            merge_consecutive: If True, merge consecutive segments from same speaker
            gap_threshold: Maximum gap (seconds) between segments to merge
        """
        self.merge_consecutive = merge_consecutive
        self.gap_threshold = gap_threshold
    
    def merge(
        self,
        diarization_segments: List[DiarizationSegment],
        transcriptions: List[TranscriptionResult],
    ) -> MergedTranscript:
        """Merge transcriptions with speaker labels.
        
        Args:
            diarization_segments: Original diarization segments
            transcriptions: Transcription results for each segment
            
        Returns:
            MergedTranscript with speaker labels in chronological order
        """
        if not transcriptions:
            return MergedTranscript()
        
        logger.info(f"Merging {len(transcriptions)} transcriptions")
        
        # Create speaker label mapping
        speaker_map = self._create_speaker_map(transcriptions)
        
        # Create speaker turns
        turns = []
        for result in transcriptions:
            if not result.text.strip():
                continue
            
            speaker_id = result.speaker_id
            speaker_label = speaker_map.get(speaker_id, speaker_id)
            
            turn = SpeakerTurn(
                speaker_id=speaker_id,
                speaker_label=speaker_label,
                text=result.text.strip(),
                start_time=result.start_time,
                end_time=result.end_time,
            )
            turns.append(turn)
        
        # Sort by start time
        turns.sort(key=lambda t: t.start_time)
        
        # Merge consecutive segments if enabled
        if self.merge_consecutive:
            turns = self._merge_consecutive_turns(turns)
        
        # Calculate statistics
        total_duration = max(t.end_time for t in turns) if turns else 0.0
        speaker_count = len(speaker_map)
        
        merged = MergedTranscript(
            turns=turns,
            total_duration=total_duration,
            speaker_count=speaker_count,
        )
        
        logger.info(
            f"Merged into {len(turns)} turns, "
            f"{speaker_count} speakers, "
            f"{total_duration:.2f}s total"
        )
        
        return merged
    
    def _create_speaker_map(
        self, 
        transcriptions: List[TranscriptionResult]
    ) -> Dict[str, str]:
        """Create mapping from speaker IDs to human-readable labels.
        
        Args:
            transcriptions: List of transcription results
            
        Returns:
            Dict mapping speaker_id to label (e.g., "Speaker 1")
        """
        # Get unique speakers in order of appearance
        seen_speakers = []
        for result in transcriptions:
            speaker_id = result.speaker_id
            if speaker_id not in seen_speakers:
                seen_speakers.append(speaker_id)
        
        # Create labels
        speaker_map = {}
        for i, speaker_id in enumerate(seen_speakers, 1):
            speaker_map[speaker_id] = f"Speaker {i}"
        
        return speaker_map
    
    def _merge_consecutive_turns(
        self, 
        turns: List[SpeakerTurn]
    ) -> List[SpeakerTurn]:
        """Merge consecutive turns from the same speaker.
        
        Args:
            turns: List of speaker turns (sorted by start time)
            
        Returns:
            List of merged speaker turns
        """
        if not turns:
            return []
        
        merged = [turns[0]]
        
        for current in turns[1:]:
            previous = merged[-1]
            
            # Check if can merge
            same_speaker = current.speaker_id == previous.speaker_id
            small_gap = (current.start_time - previous.end_time) <= self.gap_threshold
            
            if same_speaker and small_gap:
                # Merge with previous
                merged_turn = SpeakerTurn(
                    speaker_id=previous.speaker_id,
                    speaker_label=previous.speaker_label,
                    text=f"{previous.text} {current.text}",
                    start_time=previous.start_time,
                    end_time=current.end_time,
                )
                merged[-1] = merged_turn
            else:
                # Add as new turn
                merged.append(current)
        
        if len(merged) < len(turns):
            logger.debug(f"Merged {len(turns)} turns into {len(merged)}")
        
        return merged
