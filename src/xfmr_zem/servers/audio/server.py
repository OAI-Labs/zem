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
def slice(
    data: Any,
    audio_column: str = "audio_path",
    rttm_column: str = "rttm_path",
    output_dir: str = "output/chunks"
) -> Any:
    """
    Cắt audio thành các đoạn nhỏ dựa trên kết quả diarization.
    Trả về danh sách các segment kèm đường dẫn file chunk.
    """
    from .components.slicer import AudioSlicer
    from .components.diarization.diarizer import DiariZenDiarizer
    from .core.models import SlicerConfig
    import os

    items = server.get_data(data)
    slicer = AudioSlicer(SlicerConfig())
    diarizer = DiariZenDiarizer() # Dùng để parse RTTM nếu cần
    
    os.makedirs(output_dir, exist_ok=True)
    all_segments = []

    for item in items:
        audio_path = Path(item.get(audio_column, ""))
        rttm_path = Path(item.get(rttm_column, ""))
        
        if audio_path.exists() and rttm_path.exists():
            # Parse RTTM thành các vùng cần cắt
            with open(rttm_path, "r") as f:
                rttm_content = f.read()
            
            # Logic đơn giản hóa: Slicer nhận audio và danh sách regions
            # Ở đây chúng ta giả định slicer có thể parse hoặc chúng ta parse ở đây
            # Để đơn giản, tôi sẽ gọi pipeline nội bộ để xử lý phần này
            try:
                # Cắt và lưu thành file thực tế
                from .core.models import DiarizationSegment
                segments = []
                for line in rttm_content.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 8:
                        segments.append(DiarizationSegment(
                            start=float(parts[3]),
                            end=float(parts[3]) + float(parts[4]),
                            speaker_id=parts[7]
                        ))
                
                sliced_chunks = slicer.slice(audio_path, segments)
                
                for i, chunk in enumerate(sliced_chunks):
                    chunk_name = f"{audio_path.stem}_seg_{i:04d}_{chunk.speaker_id}.wav"
                    chunk_path = os.path.join(output_dir, chunk_name)
                    # Lưu file chunk
                    import soundfile as sf
                    sf.write(chunk_path, chunk.audio_data, chunk.sample_rate)
                    
                    all_segments.append({
                        "original_audio": str(audio_path),
                        "chunk_path": chunk_path,
                        "speaker_id": chunk.speaker_id,
                        "start": chunk.start_with_padding,
                        "end": chunk.end_with_padding
                    })
            except Exception as e:
                logger.error(f"Error slicing {audio_path}: {e}")

    return server.save_output(all_segments)


@server.tool()
def merge(
    data: Any,
    text_column: str = "text",
    group_by: str = "original_audio"
) -> Any:
    """
    Hợp nhất các đoạn text đã dịch thành hội thoại hoàn chỉnh.
    """
    from .components.merging.merger import TranscriptMerger
    from .core.models import TranscriptionResult, AudioSegment, DiarizationSegment
    import numpy as np

    items = server.get_data(data)
    merger = TranscriptMerger()
    
    # Gom nhóm theo file gốc
    groups = {}
    for item in items:
        original = item.get(group_by, "default")
        if original not in groups:
            groups[original] = []
        groups[original].append(item)
    
    final_results = []
    for original_file, segments in groups.items():
        # Chuyển đổi dữ liệu input thành TranscriptionResult objects
        results = []
        for s in segments:
            # Tạo dummy objects để merger hoạt động
            dummy_dia = DiarizationSegment(start=s['start'], end=s['end'], speaker_id=s['speaker_id'])
            dummy_audio = AudioSegment(
                audio_data=np.array([]), 
                sample_rate=16000, 
                original_segment=dummy_dia,
                start_with_padding=s['start'],
                end_with_padding=s['end']
            )
            results.append(TranscriptionResult(text=s.get(text_column, ""), segment=dummy_audio))
        
        merged = merger.merge([], results) # DiarizationSegment list không dùng trong logic merge hiện tại
        final_results.append({
            "audio_path": original_file,
            "transcript": merged.to_text(),
            "turns_count": len(merged.turns)
        })

    return server.save_output(final_results)


if __name__ == "__main__":
    server.run()
