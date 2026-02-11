from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import json

from xfmr_zem.server import ZemServer
from ...audio.config import ASRConfig, DiarizationConfig

# Import components with lazy loading/error handling
try:
    from ...audio.components.asr.transcriber import VieASRTranscriber
    from ...audio.components.diarization.diarizer import DiariZenDiarizer
    from ...audio.core.models import AudioSegment, TranscriptionResult
    _AUDIO_DEPS_AVAILABLE = True
except ImportError:
    # Fallback for when optional deps are missing
    VieASRTranscriber = None
    DiariZenDiarizer = None
    AudioSegment = None
    TranscriptionResult = None
    _AUDIO_DEPS_AVAILABLE = False

# Initialize Logger
logger = logging.getLogger(__name__)

# Initialize ZemServer
mcp = ZemServer("audio")

@mcp.tool()
async def transcribe(
    audio_path: str,
    model: str = "zzasdf/viet_iter3_pseudo_label",
    hotwords: str = "",
    batch_size: int = 1,
    device: str = "cuda",
    output_json: bool = False
) -> str:
    """
    Transcribe audio file to text using Vietnamese ASR (Zipformer).
    
    Args:
        audio_path: Absolute path to the audio file.
        model: Model name/path (default: zzasdf/viet_iter3_pseudo_label).
        hotwords: Comma-separated list of hotwords for contextual biasing.
        batch_size: Batch size for inference.
        device: Device to run model on (cuda/cpu).
        output_json: If True, returns JSON string with timestamps and segments.
    """
    if not _AUDIO_DEPS_AVAILABLE:
        raise RuntimeError("Audio dependencies (icefall/k2) not installed. Install with 'uv sync --extra voice' and external deps.")
    
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Parse hotwords
    hw_list = [w.strip() for w in hotwords.split(",")] if hotwords else None
    
    config = ASRConfig(
        model_name=model,
        hotwords=hw_list,
        batch_size=batch_size,
        device=device
    )
    
    try:
        transcriber = VieASRTranscriber(config)
        
        # Load audio using soundfile for high performance
        import soundfile as sf
        import numpy as np
        
        audio_data, sr = sf.read(str(path))
        # Convert to float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        duration = len(audio_data) / sr
        
        # Create single segment for the whole file
        # In a real pipeline, we might use the Slicer component here first
        segment = AudioSegment(
            audio_data=audio_data,
            sample_rate=sr,
            duration=duration,
            original_segment=None 
        )
        
        results = transcriber.transcribe([segment])
        
        if output_json:
            json_results = []
            for r in results:
                json_results.append({
                    "text": r.text,
                    "start": r.segment.start_time if hasattr(r.segment, 'start_time') else 0.0,
                    "end": r.segment.end_time if hasattr(r.segment, 'end_time') else duration,
                    "confidence": r.confidence
                })
            return json.dumps(json_results, ensure_ascii=False)
        else:
            # Combine results into plain text
            full_text = " ".join([r.text for r in results])
            return full_text

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

@mcp.tool()
async def diarize(
    audio_path: str,
    num_speakers: int = 0,
    output_rttm: str = "",
    device: str = "cuda"
) -> str:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_path: Absolute path to the audio file.
        num_speakers: Expected number of speakers (0 = auto).
        output_rttm: Optional path to save RTTM output.
        device: Device to run model on (cuda/cpu).
    """
    if not _AUDIO_DEPS_AVAILABLE:
        raise RuntimeError("Audio dependencies not installed. Install with 'uv sync --extra voice'")
        
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
        
    config = DiarizationConfig(
        rttm_output_dir=Path(output_rttm).parent if output_rttm else None,
        device=device
    )
    
    try:
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
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise RuntimeError(f"Diarization failed: {str(e)}")

if __name__ == "__main__":
    mcp.run()
