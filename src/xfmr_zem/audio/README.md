# Audio Transcribe & Diarization Module

This module provides an industrial-grade pipeline for Vietnamese Speech-to-Text with Speaker Diarization, integrated into `xfmr-zem`.

## Architecture

The pipeline consists of the following stages:

1.  **Preprocessing**: Standardizes audio format (16kHz, Mono), performs noise reduction and normalization.
2.  **Slicer**: Splits audio into manageable chunks based on silence (VAD) to improve processing efficiency.
3.  **Diarization (DiariZen)**: State-of-the-art speaker diarization using a vendored version of `DiariZen` (built on `pyannote-audio`).
    - Uses local `diarizen_lib` to ensure compatibility.
    - Optimized for GPU execution.
4.  **ASR (VieASR)**: High-accuracy Vietnamese Automatic Speech Recognition using `icefall`/`k2` framework.
    - Model: `zzasdf/viet_iter3_pseudo_label` (Zipformer).
    - Supports hotword boosting (contextual biasing).
5.  **Merging**: Aligns transcripts with speaker segments and merges them into a final output.

## Dependencies

### Standard Dependencies
Install with `uv` or `pip`:
```bash
uv sync --extra voice
# OR
pip install ".[voice]"
```
This installs `pyannote-audio`, `torchaudio`, and other standard libraries.

### Advanced Dependencies (VieASR)
The ASR component (`VieASR`) requires `k2`, `kaldifeat`, and `icefall`. These are NOT included in the standard `voice` extra due to platform-specific wheel requirements.

**Installation Instructions:**

1.  **k2**: Follow instructions at [k2-fsa.github.io](https://k2-fsa.github.io/k2/installation/index.html).
    ```bash
    # Example for CUDA 11.8
    pip install k2==1.24.4.dev20240223+cuda11.8.torch2.1.0 -f https://k2-fsa.github.io/k2/cuda.html
    ```

2.  **kaldifeat**:
    ```bash
    pip install kaldifeat
    ```

3.  **icefall**: Must be installed as an external dependency (via `pip install git+...` or local install). The wrapper logic in `transcriber.py` handles the integration.

**Note:** If these dependencies are missing, the pipeline will fallback to a "Simplified" backend which generates placeholder transcripts (useful for testing pipeline flow).

## Scaling & Performance

- **Batch Processing**: Both Diarization and ASR support batch processing.
    - ASR Batch Size can be configured via `ASR_BATCH_SIZE` or CLI.
- **GPU Usage**: Highly recommended. The pipeline automatically detects CUDA.
- **Concurrency**: The current implementation is sequential per file. For high throughput, run multiple `zem audio transcribe` processes or use a task queue wrapper.

## Troubleshooting

### `pyannote.audio` Version Conflict
This module vendors a specific version of `pyannote-audio` inside `components/diarization/diarizen_lib`.
We use a custom `sys.path` injection in `src/xfmr_zem/__init__.py` to ensure this vendored version takes precedence over any system-installed `pyannote` packages.

If you encounter `TypeError: SpeakerDiarization.__init__() got an unexpected keyword argument`, ensure you are triggering the code via `python -m xfmr_zem.cli` or `zem` command, which initializes the path patching.
