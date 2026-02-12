"""
Speaker diarization module using DiariZen.

Wraps DiariZen pipeline for speaker identification and segmentation.
"""

import logging
from pathlib import Path
from typing import List, Optional

from ...core.interfaces import IDiarizer
from ...core.models import DiarizationSegment, DiarizationConfig

logger = logging.getLogger(__name__)


class DiariZenDiarizer(IDiarizer):
    """Speaker diarization using DiariZen pipeline.
    
    DiariZen is a speaker diarization toolkit using WavLM and Pyannote 3.1.
    Model: BUT-FIT/diarizen-wavlm-large-s80-md
    
    Features:
    - State-of-the-art diarization performance
    - Handles overlapping speech
    - Outputs RTTM format
    """
    
    def __init__(self, config: Optional[DiarizationConfig] = None):
        """Initialize DiariZen diarizer.
        
        Args:
            config: Diarization configuration
        """
        self.config = config or DiarizationConfig()
        self._pipeline = None
        self._device = None
    
    def _load_pipeline(self):
        """Lazy load DiariZen pipeline."""
        if self._pipeline is not None:
            return
        
        logger.info(f"Loading DiariZen model: {self.config.model_name}")
        
        try:
            # Import DiariZen pipeline
            from .diarizen.pipelines.inference import DiariZenPipeline
            
            # Load pre-trained model from HuggingFace
            pipeline_kwargs = {}
            if self.config.rttm_output_dir:
                pipeline_kwargs["rttm_out_dir"] = str(self.config.rttm_output_dir)
            
            self._pipeline = DiariZenPipeline.from_pretrained(
                self.config.model_name,
                **pipeline_kwargs
            )
            
            # Move to device if needed
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    logger.warning("CUDA not available, falling back to CPU")
                    self._device = "cpu"
            else:
                self._device = "cpu"
            
            logger.info(f"DiariZen loaded on {self._device}")
            
        except ImportError as e:
            logger.error(f"Failed to import DiariZen: {e}")
            raise ImportError(
                "DiariZen is not installed. Please install it from: "
                "https://github.com/BUTSpeechFIT/DiariZen"
            ) from e
    
    def diarize(self, audio_path: Path) -> List[DiarizationSegment]:
        """Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to preprocessed audio file (16kHz mono WAV)
            
        Returns:
            List of DiarizationSegment with speaker IDs and timestamps.
            Overlapping speech creates multiple segments for same time range.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load pipeline lazily
        self._load_pipeline()
        
        logger.info(f"Running diarization on: {audio_path}")
        
        # Run diarization
        diarization_result = self._pipeline(
            str(audio_path),
            sess_name=audio_path.stem
        )
        
        # Convert to DiarizationSegment list
        segments = self._parse_diarization_result(diarization_result)
        
        logger.info(f"Found {len(segments)} segments with {self._count_speakers(segments)} speakers")
        
        return segments
    
    def _parse_diarization_result(self, diarization) -> List[DiarizationSegment]:
        """Parse DiariZen/Pyannote diarization result.
        
        Args:
            diarization: Pyannote Annotation object
            
        Returns:
            List of DiarizationSegment
        """
        segments = []
        
        # Track time ranges for overlap detection
        time_ranges = []
        
        # First pass: collect all segments
        raw_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker_id": f"speaker_{speaker}",
            })
            time_ranges.append((turn.start, turn.end))
        
        # Second pass: detect overlapping segments
        for i, seg in enumerate(raw_segments):
            is_overlapping = self._check_overlap(
                seg["start"], 
                seg["end"], 
                time_ranges, 
                exclude_idx=i
            )
            
            segments.append(DiarizationSegment(
                start=seg["start"],
                end=seg["end"],
                speaker_id=seg["speaker_id"],
                is_overlapping=is_overlapping,
            ))
        
        # Sort by start time
        segments.sort(key=lambda s: (s.start, s.speaker_id))
        
        return segments
    
    def _check_overlap(
        self, 
        start: float, 
        end: float, 
        time_ranges: List[tuple], 
        exclude_idx: int
    ) -> bool:
        """Check if a segment overlaps with any other segment.
        
        Args:
            start: Segment start time
            end: Segment end time
            time_ranges: List of (start, end) tuples
            exclude_idx: Index to exclude from comparison
            
        Returns:
            True if overlapping with another segment
        """
        for i, (other_start, other_end) in enumerate(time_ranges):
            if i == exclude_idx:
                continue
            # Check for overlap
            if start < other_end and end > other_start:
                return True
        return False
    
    def _count_speakers(self, segments: List[DiarizationSegment]) -> int:
        """Count unique speakers in segments."""
        return len(set(seg.speaker_id for seg in segments))
    
    def save_rttm(
        self, 
        segments: List[DiarizationSegment], 
        output_path: Path,
        audio_name: str = "audio"
    ) -> None:
        """Save segments to RTTM format.
        
        RTTM format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        
        Args:
            segments: List of diarization segments
            output_path: Path to save RTTM file
            audio_name: Name identifier for the audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                duration = seg.end - seg.start
                line = f"SPEAKER {audio_name} 1 {seg.start:.3f} {duration:.3f} <NA> <NA> {seg.speaker_id} <NA> <NA>\n"
                f.write(line)
        
        logger.info(f"Saved RTTM to: {output_path}")
