import logging
import pandas as pd
from pathlib import Path
from xfmr_zem.server import ZemServer
from ...audio.core.models import AudioSegment
from ...audio.config import ASRConfig, DiarizationConfig

# Import components
# Note: These paths assume the refactored structure matches what we expect
try:
    from ...audio.components.asr.transcriber import VieASRTranscriber
    from ...audio.components.diarization.diarizer import DiariZenDiarizer
    from ...audio.core.models import AudioSegment as CoreAudioSegment
except ImportError:
    # Fallback/Mock for when optional deps are missing
    VieASRTranscriber = None
    DiariZenDiarizer = None
    CoreAudioSegment = None

# Initialize Logger
logger = logging.getLogger(__name__)

# Initialize ZemServer
mcp = ZemServer("audio")

@mcp.tool()
async def transcribe(
    audio_path: str,
    model: str = "program_vn_iter3",
    hotwords: str = "",
    batch_size: int = 1
) -> str:
    """
    Transcribe audio file to text using Vietnamese ASR (Zipformer).
    
    Args:
        audio_path: Absolute path to the audio file.
        model: Model name (default: program_vn_iter3).
        hotwords: Comma-separated list of hotwords for contextual biasing.
        batch_size: Batch size for inference.
    """
    if not VieASRTranscriber:
        return "Error: Audio dependencies not installed. Install with 'uv sync --extra audio'"
    
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Parse hotwords
    hw_list = [w.strip() for w in hotwords.split(",")] if hotwords else None
    
    config = ASRConfig(
        model_name=model,
        hotwords=hw_list,
        batch_size=batch_size,
    )
    
    transcriber = VieASRTranscriber(config)
    
    # Load audio
    # Assuming we have a helper to load audio or passed directly
    # For simplicity, we create a segment from file
    import soundfile as sf
    import numpy as np
    
    audio_data, sr = sf.read(str(path))
    # Convert to float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
        
    duration = len(audio_data) / sr
    
    segment = CoreAudioSegment(
        audio_data=audio_data,
        sample_rate=sr,
        duration=duration,
        original_segment=None # todo
    )
    
    results = transcriber.transcribe([segment])
    
    # Combine results
    full_text = " ".join([r.text for r in results])
    return full_text

@mcp.tool()
async def diarize(
    audio_path: str,
    num_speakers: int = 0,
    output_rttm: str = ""
) -> str:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_path: Absolute path to the audio file.
        num_speakers: Expected number of speakers (0 = auto).
        output_rttm: Optional path to save RTTM output.
    """
    if not DiariZenDiarizer:
        return "Error: Audio dependencies not installed. Install with 'uv sync --extra audio'"
        
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
        
    config = DiarizationConfig(
        rttm_output_dir=Path(output_rttm).parent if output_rttm else None
    )
    
    diarizer = DiariZenDiarizer(config)
    segments = diarizer.diarize(path)
    
    # Format output
    output = f"Found {len(set(s.speaker_id for s in segments))} speakers.\n"
    for seg in segments:
        output += f"[{seg.start:.2f} - {seg.end:.2f}] {seg.speaker_id}\n"
        
    if output_rttm:
        diarizer.save_rttm(segments, Path(output_rttm), audio_name=path.stem)
        output += f"\nRTTM saved to {output_rttm}"
        
    return output

if __name__ == "__main__":
    mcp.run()
