import os
import sys
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from .core.models import ASRConfig, DiarizationConfig

# Initialize ZemServer
server = ZemServer("audio", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

# Import components (lazy load or try-except inside tools if needed, but top-level is fine if deps are expected)
try:
    from .components.asr.transcriber import VieASRTranscriber
    from .components.diarization.diarizer import DiariZenDiarizer
    from .core.models import AudioSegment as CoreAudioSegment
except ImportError as e:
    logger.warning(f"Audio dependencies missing: {e}. Vui lòng chạy 'python scripts/setup_audio.py' để thiết lập môi trường.")
    VieASRTranscriber = None
    DiariZenDiarizer = None
    CoreAudioSegment = None


@server.tool()
def transcribe(
    data: Any,
    audio_column: str = "audio_path",
    text_column: str = "text",
    model: str = "program_vn_iter3",
    hotwords: str = "",
    batch_size: int = 1
) -> Any:
    """
    Transcribe audio files to text using Vietnamese ASR.
    
    Args:
        data: List of records or file path (handled by ZemServer).
        audio_column: Key in record containing audio path.
        text_column: Key to store transcription result.
        model: Model name.
        hotwords: Comma-separated list of hotwords.
        batch_size: Batch size for inference.
    """
    if not VieASRTranscriber:
        return "Error: Audio dependencies not installed. Install with 'uv sync --extra audio'" # Or raise error

    items = server.get_data(data)
    if not items:
        return []

    logger.info(f"Audio: Transcribing {len(items)} items")

    # Parse hotwords
    hw_list = [w.strip() for w in hotwords.split(",")] if hotwords else None
    
    config = ASRConfig(
        model_name=model,
        hotwords=hw_list,
        batch_size=batch_size,
    )
    
    transcriber = VieASRTranscriber(config)
    
    import soundfile as sf
    import numpy as np

    # Process items
    # Note: VieASRTranscriber supports batching via `transcribe(segments)`.
    # We should collect all valid segments and transcribe them in batches for efficiency.
    
    segments_map = [] # List of (item_index, segment)
    
    for idx, item in enumerate(items):
        audio_path_str = item.get(audio_column)
        if not audio_path_str:
            continue
            
        path = Path(audio_path_str)
        if not path.exists():
            logger.warning(f"Audio file not found: {path}")
            item[text_column] = "" # Or error
            continue
            
        try:
            audio_data, sr = sf.read(str(path))
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            duration = len(audio_data) / sr
            
            # Create a dummy diarization segment for the whole file
            from .core.models import DiarizationSegment
            dummy_seg = DiarizationSegment(
                start=0.0,
                end=duration,
                speaker_id="speaker_0"
            )
            
            segment = CoreAudioSegment(
                audio_data=audio_data,
                sample_rate=sr,
                original_segment=dummy_seg,
                start_with_padding=0.0,
                end_with_padding=duration
            )
            segments_map.append((idx, segment))
        except Exception as e:
            logger.error(f"Error loading audio {path}: {e}")
            item[text_column] = ""

    if segments_map:
        segments_only = [s for _, s in segments_map]
        results = transcriber.transcribe(segments_only)
        
        # Map results back to items
        for i, result in enumerate(results):
            original_idx = segments_map[i][0]
            # Accumulate text if needed, or just set it
            # VieASRTranscriber returns TranscriptionResult with .text
            items[original_idx][text_column] = result.text

    return server.save_output(items)


@server.tool()
def diarize(
    data: Any,
    audio_column: str = "audio_path",
    rttm_column: str = "rttm_path",
    speakers_column: str = "num_speakers",
    num_speakers: int = 0,
    output_dir: str = ""
) -> Any:
    """
    Perform speaker diarization on audio files.
    
    Args:
        data: input data.
        audio_column: input audio path column.
        rttm_column: output column to store path to RTTM file.
        speakers_column: output column to store detected speaker count.
        num_speakers: expected number of speakers (0=auto).
        output_dir: directory to save RTTM files (optional).
    """
    if not DiariZenDiarizer:
        return "Error: Audio dependencies not installed."

    items = server.get_data(data)
    if not items:
        return []

    config = DiarizationConfig(
        rttm_output_dir=Path(output_dir) if output_dir else None
    )
    diarizer = DiariZenDiarizer(config)

    for item in items:
        path = Path(item.get(audio_column, ""))
        if path.exists():
            try:
                segments = diarizer.diarize(path)
                item[speakers_column] = len(set(s.speaker_id for s in segments))
                if output_dir:
                    rttm_path = Path(output_dir) / f"{path.stem}.rttm"
                    diarizer.save_rttm(segments, rttm_path, audio_name=path.stem)
                    item[rttm_column] = str(rttm_path)
            except Exception as e:
                logger.error(f"Error diarizing {path}: {e}")

    return server.save_output(items)


@server.tool()
def preprocess(
    data: Any,
    audio_column: str = "audio_path",
    output_column: str = "preprocessed_path",
) -> Any:
    """
    Tiền xử lý âm thanh (chuẩn hóa 16kHz, mono, khử nhiễu).
    """
    from .components.preprocessing import AudioPreprocessor
    from .core.models import PreprocessingConfig

    items = server.get_data(data)
    preprocessor = AudioPreprocessor(PreprocessingConfig())

    for item in items:
        path = Path(item.get(audio_column, ""))
        if path.exists():
            item[output_column] = str(preprocessor.preprocess(path))
    
    return server.save_output(items)


@server.tool()
def full_pipeline(
    data: Any,
    audio_column: str = "audio_path",
    output_column: str = "transcript_path",
) -> Any:
    """
    Chạy toàn bộ quy trình: Preprocess -> Diarize -> Slice -> Transcribe -> Merge.
    """
    from .pipeline import create_pipeline
    
    items = server.get_data(data)
    pipeline = create_pipeline()

    for item in items:
        path = Path(item.get(audio_column, ""))
        if path.exists():
            # Lưu kết quả ra file JSON/Text tùy cấu hình
            output_path = path.parent / f"{path.stem}_result.json"
            transcript = pipeline.process(path, output_path=output_path)
            item[output_column] = str(output_path)
            item["text_preview"] = transcript.to_text()[:200]
            
    return server.save_output(items)


if __name__ == "__main__":
    server.run()
