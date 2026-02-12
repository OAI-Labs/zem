import os
import abc
from typing import Dict, Any, List
from PIL import Image
from loguru import logger
import tempfile
import subprocess
import glob
from pathlib import Path
from dotenv import load_dotenv

class OCREngineBase(abc.ABC):
    """
    Abstract Base Class for OCR Engines (Dependency Inversion & Open/Closed).
    """
    @abc.abstractmethod
    def process(self, image_path: str) -> Dict[str, Any]:
        """Process an image and return extracted text and metadata."""
        pass

class TesseractEngine(OCREngineBase):
    """
    Lightweight OCR using Tesseract (Fast & Simple).
    """
    def __init__(self):
        logger.debug("TesseractEngine: Initializing...")
        try:
            import pytesseract
            import shutil
            
            logger.debug("TesseractEngine: Checking for tesseract binary...")
            # Check if tesseract binary exists
            if not shutil.which("tesseract"):
                raise RuntimeError(
                    "Tesseract binary not found. To use the 'tesseract' engine, "
                    "please install it using: sudo apt install tesseract-ocr"
                )
            
            self.pytesseract = pytesseract
            logger.debug("TesseractEngine: Initialization complete")
        except ImportError:
            logger.error("pytesseract not installed. Please install with 'pip install pytesseract'")
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using Tesseract to process: {image_path}")
        image = Image.open(image_path)
        text = self.pytesseract.image_to_string(image)
        return {
            "text": text,
            "engine": "tesseract",
            "metadata": {"format": image.format, "size": image.size}
        }

class PaddleEngine(OCREngineBase):
    """
    Medium-weight OCR using PaddleOCR (High accuracy for multi-language).
    """
    def __init__(self):
        logger.debug("PaddleEngine: Initializing...")
        try:
            logger.debug("PaddleEngine: Importing PaddleOCR...")
            from paddleocr import PaddleOCR
            logger.debug("PaddleEngine: Creating PaddleOCR instance (use_angle_cls=True, lang='en')...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en') # Default to English
            logger.debug("PaddleEngine: Initialization complete")
        except ImportError:
            logger.error("paddleocr not installed. Please install with 'pip install paddleocr paddlepaddle'")
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using PaddleOCR to process: {image_path}")
        result = self.ocr.ocr(image_path, cls=True)
        
        full_text = []
        scores = []
        for line in result:
            if line:
                for res in line:
                    full_text.append(res[1][0])
                    scores.append(float(res[1][1]))
        
        return {
            "text": "\n".join(full_text),
            "engine": "paddleocr",
            "metadata": {"avg_confidence": sum(scores)/len(scores) if scores else 0}
        }

class PaddleVLEngine(OCREngineBase):
    """
    OCR using PaddleOCR VL (Vision-Language) pipeline for Markdown extraction.
    Auto-installs dependencies inside __init__ if missing.
    """
    def __init__(self):
        logger.debug("PaddleVLEngine: Initializing...")
        _ = load_dotenv(override=True)
        try:
            # Thử import lần đầu
            from paddleocr import PaddleOCRVL

            self.pipeline = PaddleOCRVL()
            logger.debug("PaddleVLEngine: Initialization complete")
            
        except ImportError:
            logger.warning("PaddleOCR dependencies not found. Auto-installing via 'uv'...")
            try:
                # Chạy lệnh cài đặt 1: PaddlePaddle GPU
                subprocess.check_call([
                    "uv", "pip", "install", 
                    "paddlepaddle-gpu==3.2.1", 
                    "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
                ])
                
                # Chạy lệnh cài đặt 2: PaddleOCR doc-parser
                subprocess.check_call([
                    "uv", "pip", "install", "-U", "paddleocr[doc-parser]"
                ])

                # Import lại sau khi cài đặt thành công
                logger.info("Installation complete. Retrying initialization...")
                from paddleocr import PaddleOCRVL
                self.pipeline = PaddleOCRVL()
                logger.success("PaddleVLEngine initialized successfully after installation.")
                
            except Exception as e:
                logger.error(f"Failed to auto-install or initialize PaddleOCR: {e}")
                raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using PaddleOCR VL to process: {image_path}")
        
        # Gọi pipeline
        output = self.pipeline.predict(str(image_path))
        file_md = ""
        
        # Xử lý kết quả qua file temp
        with tempfile.TemporaryDirectory() as temp_dir:
            for res in output:
                res.save_to_markdown(save_path=temp_dir)
            
            md_files = sorted(glob.glob(os.path.join(temp_dir, "*.md")))
            for md_file in md_files:
                with open(md_file, 'r', encoding='utf-8') as f:
                    file_md += f.read() + "\n\n"
        
        return {
            "text": file_md.strip(),
            "engine": "paddle_vl",
            "metadata": {"format": "markdown"}
        }

class LandingAIEngine(OCREngineBase):
    """
    OCR using LandingAI for Markdown extraction.
    """
    def __init__(self, model: str = "dpt-2-latest"):
        logger.debug("LandingAIEngine: Initializing...")
        try:
            from landingai_ade import LandingAIADE
            from dotenv import load_dotenv
            load_dotenv(override=True)
            self.client = LandingAIADE()
            self.model = model
            logger.debug("LandingAIEngine: Initialization complete")
        except ImportError:
            logger.error("landingai_ade or python-dotenv not installed.")
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using LandingAI ({self.model}) to process: {image_path}")
        parse_result = self.client.parse(document=Path(image_path), model=self.model)
        
        return {
            "text": parse_result.markdown,
            "engine": "landingai",
            "metadata": {"model": self.model, "format": "markdown"}
        }

class HuggingFaceVLEngine(OCREngineBase):
    """
    Advanced OCR using Hugging Face Vision Language Models (e.g. Qwen2-VL, Molmo).
    """
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_id = model_id or "Qwen/Qwen2-VL-2B-Instruct"
        self.model = None
        self.processor = None

    def _lazy_load(self):
        if self.model is None:
            try:
                logger.debug(f"HuggingFaceVLEngine: Starting lazy load for model: {self.model_id}")
                import torch
                logger.debug(f"HuggingFaceVLEngine: PyTorch version={torch.__version__}, CUDA available={torch.cuda.is_available()}")
                from transformers import AutoModelForVision2Seq, AutoProcessor
                
                logger.info(f"Loading Hugging Face VL model: {self.model_id} (this may take a while)...")
                logger.debug(f"HuggingFaceVLEngine: Loading processor from {self.model_id}...")
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                logger.debug(f"HuggingFaceVLEngine: Processor loaded successfully")
                
                # Use GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Default to float32
                dtype = torch.float32
                if device == "cuda":
                    # Check compute capability
                    cc_major = torch.cuda.get_device_properties(0).major
                    # Pascal (6.1) has poor FP16 performance, so use FP32.
                    # Volta (7.0) and newer have good FP16 performance.
                    if cc_major >= 7:
                        dtype = torch.float16
                        
                logger.debug(f"HuggingFaceVLEngine: Loading model with dtype={dtype}, device='{device}'...")
                # Using AutoModelForVision2Seq for generality
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id, 
                    torch_dtype=dtype,
                    device_map=None,
                    trust_remote_code=True
                ).to(device)
                self._device = device
                logger.debug(f"HuggingFaceVLEngine: Model loaded successfully on {device}")
            except ImportError:
                logger.error("transformers/torch not installed. Required for HuggingFace-VL.")
                raise
            except Exception as e:
                logger.error(f"Error loading model {self.model_id}: {e}")
                raise

    def process(self, image_path: str) -> Dict[str, Any]:
        self._lazy_load()
        logger.info(f"Using {self.model_id} via HuggingFaceVLEngine to process: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Use proper chat template format for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this image exactly as it appears."}
                ]
            }
        ]
        
        # Apply chat template for proper formatting
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(self._device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        # Only decode new tokens
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
        return {
            "text": text,
            "engine": "huggingface",
            "metadata": {"model_id": self.model_id}
        }

class VietOCREngine(OCREngineBase):
    """
    Specialized Vietnamese OCR using built-in Deep-ocr DocumentPipeline (Layout + OCR + MD).
    """
    def __init__(self):
        logger.debug("VietOCREngine: Initializing...")
        try:
            logger.debug("VietOCREngine: Importing DocumentPipeline and components...")
            from xfmr_zem.servers.ocr.deepdoc_vietocr.pipeline import DocumentPipeline
            from xfmr_zem.servers.ocr.deepdoc_vietocr.implementations import (
                PaddleStructureV3Analyzer,
                PaddleOCRTextDetector,
                VietOCRRecognizer,
                VietnameseTextPostProcessor,
                SmartMarkdownReconstruction
            )
            
            logger.info("Initializing Internal Deep-ocr DocumentPipeline for Vietnamese...")
            logger.debug("VietOCREngine: Creating PaddleStructureV3Analyzer...")
            layout_analyzer = PaddleStructureV3Analyzer()
            logger.debug("VietOCREngine: Creating PaddleOCRTextDetector...")
            text_detector = PaddleOCRTextDetector()
            logger.debug("VietOCREngine: Creating VietOCRRecognizer...")
            text_recognizer = VietOCRRecognizer()
            logger.debug("VietOCREngine: Creating VietnameseTextPostProcessor...")
            post_processor = VietnameseTextPostProcessor()
            logger.debug("VietOCREngine: Creating SmartMarkdownReconstruction...")
            reconstructor = SmartMarkdownReconstruction()
            
            logger.debug("VietOCREngine: Assembling DocumentPipeline...")
            self.pipeline = DocumentPipeline(
                layout_analyzer=layout_analyzer,
                text_detector=text_detector,
                text_recognizer=text_recognizer,
                post_processor=post_processor,
                reconstructor=reconstructor
            )
            logger.debug("VietOCREngine: Initialization complete")
        except Exception as e:
            logger.error(f"Error loading internal Deep-ocr components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using Internal Deep-ocr (DocumentPipeline) to process: {image_path}")
        from PIL import Image
        
        img = Image.open(image_path)
        
        # document.process returns reconstructed markdown text
        markdown_text = self.pipeline.process(img)
            
        return {
            "text": markdown_text,
            "engine": "deepdoc_vietocr",
            "metadata": {"format": "markdown"}
        }

class OCREngineFactory:
    """
    Factory to create OCR engines (Switching strategy).
    """
    @staticmethod
    def get_engine(engine_type: str, **kwargs) -> OCREngineBase:
        if engine_type == "tesseract":
            return TesseractEngine()
        elif engine_type == "paddle":
            return PaddleEngine()
        elif engine_type == "paddle_vl":
            return PaddleVLEngine()
        elif engine_type == "huggingface" or engine_type == "qwen":
            return HuggingFaceVLEngine(model_id=kwargs.get("model_id"))
        elif engine_type == "viet":
            return VietOCREngine()
        elif engine_type == "landingai":
            return LandingAIEngine(model=kwargs.get("model", "dpt-2-latest"))
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
