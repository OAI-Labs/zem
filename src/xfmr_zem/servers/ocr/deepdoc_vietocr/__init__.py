#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import io
import sys
import threading
import pdfplumber

from .ocr import OCR
from .recognizer import Recognizer
from .layout_recognizer import LayoutRecognizerDocLayoutYOLO as LayoutRecognizer
from .table_structure_recognizer import TableStructureRecognizer
# from .engine import VietDocEngine

# New Phase-Based Architecture
from .pipeline import DocumentPipeline
from .phases import (
    LayoutAnalysisPhase,
    TextDetectionPhase,
    TextRecognitionPhase,
    PostProcessingPhase,
    DocumentReconstructionPhase,
)
from .implementations import (
    DocLayoutYOLOAnalyzer,
    PaddleOCRTextDetector,
    VietOCRRecognizer,
    SVTRv2Recognizer,
    LandingAIRecognizer,
    VietnameseTextPostProcessor,
    SmartMarkdownReconstruction,
    create_default_pipeline,
    create_svtrv2_pipeline,
    create_experimental_pipeline,
)


LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


# Removed init_in_out


__all__ = [
    # Legacy API (backward compatibility)
    "OCR",
    "Recognizer",
    "LayoutRecognizer",
    "TableStructureRecognizer",
    # "VietDocEngine",
    # "init_in_out",

    # New Phase-Based Architecture
    "DocumentPipeline",

    # Abstract Phase Interfaces
    "LayoutAnalysisPhase",
    "TextDetectionPhase",
    "TextRecognitionPhase",
    "PostProcessingPhase",
    "DocumentReconstructionPhase",

    # Concrete Implementations
    "DocLayoutYOLOAnalyzer",
    "PaddleOCRTextDetector",
    "VietOCRRecognizer",
    "SVTRv2Recognizer",
    "LandingAIRecognizer",
    "VietnameseTextPostProcessor",
    "SmartMarkdownReconstruction",

    # Factory Functions
    "create_default_pipeline",
    "create_svtrv2_pipeline",
    "create_experimental_pipeline",
]
