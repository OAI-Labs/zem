"""
Vietnamese ASR module using VieASR (Zipformer).

Transcribes audio segments to Vietnamese text.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Import k2 dependencies
try:
    import k2
    import kaldifeat
    import sentencepiece as spm
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    import math

    # Try to import icefall modules from installed package
    try:
        from icefall.utils import AttributeDict
        from icefall.decode import one_best_decoding
        # Note: modified_beam_search and greedy_search_batch were in scripts.
        # We will attempt to use icefall.decode or fallback to simplified.
        # If specific beam search is needed, it should be re-implemented or supplied as a library function.
        _ICEFALL_AVAILABLE = True
    except ImportError:
         logger.warning("icefall package not found. Install with 'pip install git+https://github.com/k2-fsa/icefall.git'")
         _ICEFALL_AVAILABLE = False

except ImportError as e:
    logger.warning(f"Failed to import VieASR dependencies: {e}. Fallback to simplified backend.")
    _ICEFALL_AVAILABLE = False

import numpy as np
import soundfile as sf

from ...core.interfaces import ITranscriber
from ...core.models import AudioSegment, TranscriptionResult
from ...config import ASRConfig

logger = logging.getLogger(__name__)


class VieASRTranscriber(ITranscriber):
    """Vietnamese ASR using VieASR Zipformer model.
    
    VieASR is trained on 70,000 hours of Vietnamese data.
    Model: zzasdf/viet_iter3_pseudo_label
    
    Features:
    - Industry-level Vietnamese ASR
    - Optimized for 3-10 second segments
    - Supports batch inference
    """
    
    def __init__(self, config: Optional[ASRConfig] = None):
        """Initialize VieASR transcriber.
        
        Args:
            config: ASR configuration
        """
        self.config = config or ASRConfig()
        self._model = None
        self._device = None
        self._token_table = None
        self._fbank = None
        self._params = None
        
        # Determine model paths
        # Priority: Config > Cache Dir > Error
        
        base_dir = Path.home() / ".cache" / "xfmr_zem" / "audio" / "models" / "viet_iter3_pseudo_label"
        
        if config.checkpoint_path:
             self.model_dir = config.checkpoint_path.parent.parent # Assuming structure
             # Actually, if checkpoint is provided, rely on it.
             # But VieASR needs specific structure (data/tokens.txt, etc.)
             # We assume if checkpoint_path is provided, it points to `exp/epoch-xx.pt` inside a valid structure.
             self.model_dir = config.checkpoint_path.parent.parent
        elif base_dir.exists():
             self.model_dir = base_dir
        else:
             # Fallback to relative path for backward compatibility if it exists
             local_path = Path(__file__).parent / "models" / "viet_iter3_pseudo_label"
             if local_path.exists():
                 self.model_dir = local_path
             else:
                 # We set a default but it will fail validation later if not found
                 self.model_dir = base_dir

        self.checkpoint_path = self.config.checkpoint_path or (self.model_dir / "exp" / "epoch-12.pt")
        self.tokens_path = self.model_dir / "data" / "Vietnam_bpe_2000_new" / "tokens.txt"
        self.bpe_model_path = self.model_dir / "data" / "Vietnam_bpe_2000_new" / "bpe.model"
        self._temp_dir = None
        self._sp = None
    
    def _load_model(self):
        """Lazy load VieASR model and resources."""
        if self._model is not None:
            return

        if not _ICEFALL_AVAILABLE:
            logger.error("VieASR dependencies not met. Cannot load model.")
            raise RuntimeError("VieASR dependencies (k2, kaldifeat, etc.) are not installed.")

        try:
            logger.info(f"Loading VieASR model from {self.checkpoint_path}")
            
            # 1. Setup device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self._device}")

            # 2. Load params
            # Get default args from parser to ensure all model config params are present
            parser = get_parser()
            args = parser.parse_args([])
            
            self._params = get_params()
            self._params.update(vars(args))
            
            # Override with specific paths and inference settings
            self._params.update({
                "exp_dir": str(self.model_dir / "exp"),
                "tokens": str(self.tokens_path),
                "use_fp16": False, # Inference usually FP32 or we can enable if supported
                "sample_rate": 16000,
                "feature_dim": 80,
                "subsampling_factor": 4,
                "avg": 1, 
                "causal": False, # Non-streaming model by default
            })

            # 3. Load Token Table
            if not self.tokens_path.exists():
                raise FileNotFoundError(f"Tokens file not found at {self.tokens_path}")
            
            self._token_table = k2.SymbolTable.from_file(str(self.tokens_path))
            self._params.blank_id = self._token_table["<blk>"]
            self._params.unk_id = self._token_table["<unk>"]
            self._params.vocab_size = num_tokens(self._token_table) + 1

            # Load BPE model
            if not self.bpe_model_path.exists():
                raise FileNotFoundError(f"BPE model not found at {self.bpe_model_path}")
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(str(self.bpe_model_path))

            # 4. Create Model
            self._model = get_model(self._params)
            
            # 5. Load Checkpoint
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
                
            checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
            self._model.load_state_dict(checkpoint["model"], strict=False)
            self._model.to(self._device)
            self._model.eval()

            # 6. Setup Fbank computer
            opts = kaldifeat.FbankOptions()
            opts.device = self._device
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = self._params.sample_rate
            opts.mel_opts.num_bins = self._params.feature_dim
            opts.mel_opts.high_freq = -400
            
            self._fbank = kaldifeat.Fbank(opts)
            
            logger.info("VieASR model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VieASR model: {e}")
            raise RuntimeError(f"VieASR load failed: {e}")
    
    def transcribe(self, segments: List[AudioSegment]) -> List[TranscriptionResult]:
        """Transcribe batch of audio segments.
        
        Args:
            segments: List of audio segments
            
        Returns:
            List of transcription results
        """
        if not segments:
            return []
            
        if _ICEFALL_AVAILABLE:
            self._load_model()
        results = []
        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_results = self._transcribe_batch(batch)
            results.extend(batch_results)
            
            logger.debug(
                f"Processed batch {i // batch_size + 1}/"
                f"{(len(segments) + batch_size - 1) // batch_size}"
            )
        
        logger.info(f"Transcription complete: {len(results)} results")
        return results
    
    def _transcribe_batch(
        self, 
        segments: List[AudioSegment]
    ) -> List[TranscriptionResult]:
        """Transcribe a batch of segments.
        
        Args:
            segments: Batch of audio segments
            
        Returns:
            List of TranscriptionResult
        """
        if _ICEFALL_AVAILABLE and self._model is not None:
            return self._transcribe_icefall(segments)
        else:
            return self._transcribe_simplified(segments)
    
    def _transcribe_icefall(
        self, 
        segments: List[AudioSegment]
    ) -> List[TranscriptionResult]:
        """Transcribe using icefall framework."""
        
        # 1. Prepare audio data
        waves = []
        for seg in segments:
            # seg.audio_data in numpy float32
            # Icefall expects 1-D float32 torch tensor
            audio_tensor = torch.from_numpy(seg.audio_data).float()
            
            # Resample if needed
            if seg.sample_rate != 16000:
                import torchaudio.transforms as T
                # Note: Ideally cache resampler. For now creating new one is acceptable.
                resampler = T.Resample(seg.sample_rate, 16000)
                audio_tensor = resampler(audio_tensor)
            
            waves.append(audio_tensor.contiguous().to(self._device))

        # 2. Compute Features
        features = self._fbank(waves)
        feature_lengths = [f.size(0) for f in features]

        features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
        feature_lengths = torch.tensor(feature_lengths, device=self._device)

        # 3. Model Forward
        with torch.no_grad():
            encoder_out, encoder_out_lens = self._model.forward_encoder(features, feature_lengths)
            
            # 4. Decode
            if self.config.hotwords:
                from icefall.context_graph import ContextGraph
                from beam_search import modified_beam_search
                
                # Create context graph
                # ContextGraph expects list of list of token IDs
                context_token_ids = []
                for word in self.config.hotwords:
                    # Use BPE model to encode hotwords
                    # out_type=int returns list of token IDs
                    token_ids = self._sp.encode(word, out_type=int)
                    context_token_ids.append(token_ids)
                
                context_graph = ContextGraph(context_score=self.config.context_score)
                context_graph.build(context_token_ids)
                
                # Perform modified beam search with context
                hyp_tokens = modified_beam_search(
                    model=self._model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    beam=4, # Default beam size
                    context_graph=context_graph,
                )
            else:
                # Default to greedy for speed
                # Use greedy_search_batch from beam_search module
                hyp_tokens = greedy_search_batch(
                        model=self._model,
                        encoder_out=encoder_out,
                        encoder_out_lens=encoder_out_lens,
                    )

            # 5. Convert tokens to text
            hyps = []
            for hyp in hyp_tokens:
                text = ""
                for i in hyp:
                    text += self._token_table[i]
                # Format text
                # Replace SentencePiece style space (U+2581)
                clean_text = text.replace(" ", " ").replace("▁", " ").strip()
                # Normalize multiple spaces
                clean_text = " ".join(clean_text.split())
                hyps.append(clean_text)

        # 6. Result Construction
        results = []
        for seg, text in zip(segments, hyps):
            results.append(TranscriptionResult(
                text=text,
                segment=seg,
                confidence=1.0
            ))
            
        return results
    
    def _transcribe_simplified(
        self,
        segments: List[AudioSegment]
    ) -> List[TranscriptionResult]:
        """Simplified transcription (placeholder for demo).
        
        In production, this would use the actual VieASR model.
        For now, returns placeholder text to demonstrate the pipeline.
        
        Args:
            segments: Batch of audio segments
            
        Returns:
            List of TranscriptionResult with placeholder text
        """
        results = []
        
        for segment in segments:
            # Placeholder - in production, this would call the actual model
            duration = segment.duration
            speaker = segment.speaker_id
            
            # Generate placeholder text based on segment info
            placeholder_text = f"[Segment {speaker}: {duration:.1f}s audio]"
            
            results.append(TranscriptionResult(
                text=placeholder_text,
                segment=segment,
                confidence=0.0,  # Indicate placeholder
            ))
            
            logger.debug(
                f"Placeholder transcription for {speaker} "
                f"[{segment.original_segment.start:.2f}-{segment.original_segment.end:.2f}]"
            )
        
        return results
    
    def _save_audio_segment(
        self, 
        segment: AudioSegment, 
        output_path: Path
    ) -> None:
        """Save audio segment to WAV file.
        
        Args:
            segment: Audio segment to save
            output_path: Output file path
        """
        audio = segment.audio_data
        sr = segment.sample_rate
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        
        sf.write(output_path, audio, sr, subtype='PCM_16')
    
    def cleanup(self) -> None:
        """Cleanup temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
            logger.debug("Temp directory cleaned up")
