"""
Main pipeline orchestrator for Audio Transcribe & Diarization.

Follows Dependency Inversion Principle - depends on abstractions.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional
import json

from .core.interfaces import (
    IPreprocessor, 
    IDiarizer, 
    IAudioSlicer, 
    ITranscriber, 
    IMerger,
)
from .core.models import MergedTranscript, PipelineConfig

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """Main orchestrator for audio transcription pipeline.
    
    Pipeline flow:
    1. Preprocess audio (16kHz mono, noise reduction)
    2. Diarize speakers (DiariZen)
    3. Slice audio into segments (with padding collar)
    4. Transcribe segments (VieASR)
    5. Merge results with speaker labels
    
    Follows Dependency Inversion - all components are injected.
    """
    
    def __init__(
        self,
        preprocessor: IPreprocessor,
        diarizer: IDiarizer,
        slicer: IAudioSlicer,
        transcriber: ITranscriber,
        merger: IMerger,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize pipeline with dependencies.
        
        Args:
            preprocessor: Audio preprocessing component
            diarizer: Speaker diarization component
            slicer: Audio slicing component
            transcriber: ASR transcription component
            merger: Transcript merging component
            config: Pipeline configuration
        """
        self.preprocessor = preprocessor
        self.diarizer = diarizer
        self.slicer = slicer
        self.transcriber = transcriber
        self.merger = merger
        self.config = config or PipelineConfig()
        
        # Setup temp directory
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def process(
        self, 
        audio_path: Path,
        output_path: Optional[Path] = None,
    ) -> MergedTranscript:
        """Run complete transcription pipeline.
        
        Args:
            audio_path: Path to input audio file
            output_path: Optional path for output file
            
        Returns:
            MergedTranscript with speaker-labeled transcription
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Starting pipeline for: {audio_path}")
        
        try:
            # Step 1: Preprocess audio
            logger.info("Step 1/5: Preprocessing audio...")
            preprocessed_path = self.preprocessor.preprocess(audio_path)
            
            # Step 2: Diarize speakers
            logger.info("Step 2/5: Diarizing speakers...")
            diarization_segments = self.diarizer.diarize(preprocessed_path)
            
            if not diarization_segments:
                logger.warning("No speech segments detected")
                return MergedTranscript()
            
            # Step 3: Slice audio
            logger.info("Step 3/5: Slicing audio segments...")
            audio_segments = self.slicer.slice(preprocessed_path, diarization_segments)
            
            if not audio_segments:
                logger.warning("No audio segments created")
                return MergedTranscript()
            
            # Step 4: Transcribe segments
            logger.info("Step 4/5: Transcribing segments...")
            transcriptions = self.transcriber.transcribe(audio_segments)
            
            # Step 5: Merge results
            logger.info("Step 5/5: Merging transcriptions...")
            merged = self.merger.merge(diarization_segments, transcriptions)
            
            # Save output if path provided
            if output_path:
                self._save_output(merged, output_path)
            
            logger.info("Pipeline complete!")
            return merged
            
        finally:
            # Cleanup
            if self.config.cleanup_temp:
                self._cleanup()
    
    def _save_output(self, transcript: MergedTranscript, output_path: Path) -> None:
        """Save transcript to file.
        
        Args:
            transcript: Merged transcript
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_format = self.config.output.format
        
        if output_format == "text" or output_format == "all":
            text_path = output_path.with_suffix(".txt")
            text_content = transcript.to_text(
                include_timestamps=self.config.output.include_timestamps
            )
            text_path.write_text(text_content, encoding="utf-8")
            logger.info(f"Saved text output: {text_path}")
        
        if output_format == "json" or output_format == "all":
            json_path = output_path.with_suffix(".json")
            json_content = json.dumps(
                transcript.to_json(), 
                ensure_ascii=False, 
                indent=2
            )
            json_path.write_text(json_content, encoding="utf-8")
            logger.info(f"Saved JSON output: {json_path}")
        
        if output_format == "srt" or output_format == "all":
            srt_path = output_path.with_suffix(".srt")
            srt_content = transcript.to_srt()
            srt_path.write_text(srt_content, encoding="utf-8")
            logger.info(f"Saved SRT output: {srt_path}")
    
    def _cleanup(self) -> None:
        """Cleanup temporary files."""
        try:
            # Clear slicer cache
            if hasattr(self.slicer, 'clear_cache'):
                self.slicer.clear_cache()
            
            # Clear transcriber temp files
            if hasattr(self.transcriber, 'cleanup'):
                self.transcriber.cleanup()
            
            # Remove temp directory contents
            if self.config.temp_dir.exists():
                for item in self.config.temp_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
            
            logger.debug("Cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


def create_pipeline(config: Optional[PipelineConfig] = None) -> TranscriptionPipeline:
    """Factory function to create pipeline with default implementations.
    
    Args:
        config: Optional pipeline configuration
        
    Returns:
        Configured TranscriptionPipeline instance
    """
    config = config or PipelineConfig()
    
    # Import implementations
    from .components.preprocessing import AudioPreprocessor
    from .components.diarization import DiariZenDiarizer
    from .components.slicer import AudioSlicer
    from .components.asr import VieASRTranscriber
    from .components.merging import TranscriptMerger
    
    # Create components with config
    preprocessor = AudioPreprocessor(
        config=config.preprocessing,
        output_dir=config.temp_dir / "preprocessed",
    )
    diarizer = DiariZenDiarizer(config=config.diarization)
    slicer = AudioSlicer(config=config.slicer)
    transcriber = VieASRTranscriber(config=config.asr)
    merger = TranscriptMerger()
    
    return TranscriptionPipeline(
        preprocessor=preprocessor,
        diarizer=diarizer,
        slicer=slicer,
        transcriber=transcriber,
        merger=merger,
        config=config,
    )
