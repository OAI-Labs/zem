"""
Configuration management for Audio Transcribe & Diarization system.

Uses Pydantic Settings for type-safe configuration.
"""

from pathlib import Path
from typing import Optional, Literal, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PreprocessingConfig(BaseSettings):
    """Configuration for audio preprocessing."""
    
    model_config = SettingsConfigDict(env_prefix="PREPROCESS_")
    
    target_sample_rate: int = Field(
        default=16000,
        description="Target sample rate for audio (DiariZen optimal: 16kHz)"
    )
    convert_to_mono: bool = Field(
        default=True,
        description="Convert stereo audio to mono"
    )
    enable_noise_reduction: bool = Field(
        default=True,
        description="Apply noise reduction to audio"
    )
    noise_reduction_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Noise reduction strength (0-1)"
    )
    enable_normalization: bool = Field(
        default=True,
        description="Normalize audio loudness"
    )
    target_loudness_lufs: float = Field(
        default=-23.0,
        description="Target loudness in LUFS for normalization"
    )


class DiarizationConfig(BaseSettings):
    """Configuration for speaker diarization."""
    
    model_config = SettingsConfigDict(env_prefix="DIARIZATION_")
    
    model_name: str = Field(
        default="BUT-FIT/diarizen-wavlm-large-s80-md",
        description="HuggingFace model name for DiariZen"
    )
    device: str = Field(
        default="cuda",
        description="Device to run model on (cuda/cpu)"
    )
    rttm_output_dir: Optional[Path] = Field(
        default=None,
        description="Directory to save RTTM files (optional)"
    )


class SlicerConfig(BaseSettings):
    """Configuration for audio slicing."""
    
    model_config = SettingsConfigDict(env_prefix="SLICER_")
    
    collar_padding: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Padding in seconds to add to segment boundaries (crucial for Vietnamese)"
    )
    min_segment_duration: float = Field(
        default=0.5,
        description="Minimum segment duration in seconds"
    )
    max_segment_duration: float = Field(
        default=30.0,
        description="Maximum segment duration in seconds"
    )


class ASRConfig(BaseSettings):
    """Configuration for ASR transcription."""
    
    model_config = SettingsConfigDict(env_prefix="ASR_")
    
    checkpoint_path: Optional[Path] = Field(
        default=None,
        description="Path to VieASR checkpoint (downloads from HF if not provided)"
    )
    model_name: str = Field(
        default="zzasdf/viet_iter3_pseudo_label",
        description="HuggingFace model name for VieASR"
    )
    device: str = Field(
        default="cuda",
        description="Device to run model on (cuda/cpu)"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for transcription"
    )
    decoding_method: Literal["greedy", "beam_search", "modified_beam_search"] = Field(
        default="modified_beam_search",
        description="Decoding method for ASR"
    )
    backend: Literal["icefall", "simplified"] = Field(
        default="simplified",
        description="ASR backend to use (icefall or simplified)"
    )
    hotwords: Optional[List[str]] = Field(
        default=None,
        description="List of hotwords/phrases to boost"
    )
    context_score: float = Field(
        default=2.0,
        description="Bonus score for hotwords (context biasing)"
    )


class OutputConfig(BaseSettings):
    """Configuration for output format."""
    
    model_config = SettingsConfigDict(env_prefix="OUTPUT_")
    
    format: Literal["text", "json", "srt", "all"] = Field(
        default="text",
        description="Output format for transcript"
    )
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in text output"
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directory for output files"
    )


class PipelineConfig(BaseSettings):
    """Main configuration combining all sub-configs."""
    
    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_nested_delimiter="__"
    )
    
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    slicer: SlicerConfig = Field(default_factory=SlicerConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Processing options
    temp_dir: Path = Field(
        default=Path("./temp"),
        description="Directory for temporary files"
    )
    cleanup_temp: bool = Field(
        default=True,
        description="Clean up temporary files after processing"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Load configuration from environment or file.
    
    Args:
        config_path: Optional path to config file (not implemented yet)
        
    Returns:
        PipelineConfig instance
    """
    # For now, just load from environment/defaults
    return PipelineConfig()
