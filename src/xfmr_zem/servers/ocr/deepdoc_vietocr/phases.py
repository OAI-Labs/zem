"""
Abstract Phase Architecture for Document Processing Pipeline

Kiến trúc này tách biệt các giai đoạn xử lý thành các abstract base classes,
cho phép dễ dàng thử nghiệm và thay thế các implementations khác nhau mà
không cần thay đổi code ở các phase khác.

Pipeline Flow:
1. Layout Analysis Phase: Phát hiện vùng layout (text, table, figure, etc.)
2. Text Detection Phase: Phát hiện text boxes trong mỗi region
3. Text Recognition Phase: Nhận dạng text từ các text boxes
4. Post-Processing Phase: Làm sạch và cải thiện text output
5. Document Reconstruction Phase: Ghép nối các regions thành markdown

Mỗi phase có thể được thay thế độc lập bằng implementations khác.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import numpy as np


# ============================================================================
# Phase 1: Layout Analysis
# ============================================================================

class LayoutAnalysisPhase(ABC):
    """
    Abstract base class cho Layout Analysis.
    Phát hiện và phân loại các vùng trong document (text, table, figure, etc.)
    """

    @abstractmethod
    def analyze(self, image: Image.Image, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Phân tích layout của document image.

        Args:
            image: PIL Image cần phân tích
            threshold: Ngưỡng confidence để giữ lại detection

        Returns:
            List of regions, mỗi region là dict:
            {
                "bbox": [x0, y0, x1, y1],  # Bounding box
                "type": str,                # "text", "table", "figure", "title", etc.
                "score": float,             # Confidence score (0-1)
                "label": str                # Nhãn chi tiết hơn (optional)
            }
        """
        pass


# ============================================================================
# Phase 2: Text Detection
# ============================================================================

class TextDetectionPhase(ABC):
    """
    Abstract base class cho Text Detection.
    Phát hiện vị trí các text boxes trong image region.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Any]:
        """
        Phát hiện text boxes trong image.

        Args:
            image: Numpy array của image (BGR hoặc RGB format)

        Returns:
            Tuple of (dt_boxes, elapsed_time):
            - dt_boxes: numpy array shape (N, 4, 2) hoặc None nếu không detect được
                       Mỗi box là 4 điểm góc [top-left, top-right, bottom-right, bottom-left]
            - elapsed_time: Thời gian xử lý (có thể bỏ qua, return 0)
        """
        pass


# ============================================================================
# Phase 3: Text Recognition
# ============================================================================

class TextRecognitionPhase(ABC):
    """
    Abstract base class cho Text Recognition.
    Nhận dạng text từ các image crops đã được phát hiện.
    """

    @abstractmethod
    def recognize(self, image_crops: List[Any]) -> Tuple[List[Tuple[str, float]], float]:
        """
        Nhận dạng text từ list các image crops.

        Args:
            image_crops: List các image crops (numpy array hoặc PIL Image)

        Returns:
            Tuple of (results, elapsed_time):
            - results: List of (text, confidence) tuples
                      text: str - Recognized text
                      confidence: float - Confidence score (0-1)
            - elapsed_time: Thời gian xử lý (có thể bỏ qua, return 0)
        """
        pass


# ============================================================================
# Phase 4: Post-Processing
# ============================================================================

class PostProcessingPhase(ABC):
    """
    Abstract base class cho Post-Processing.
    Làm sạch và cải thiện text đã nhận dạng (remove noise, spell check, etc.)
    """

    @abstractmethod
    def process(self, text: str, confidence: float, metadata: Optional[Dict] = None) -> str:
        """
        Xử lý hậu kỳ cho recognized text.

        Args:
            text: Text đã được nhận dạng
            confidence: Confidence score của recognition
            metadata: Thông tin bổ sung (region type, bbox, etc.)

        Returns:
            Cleaned/improved text
        """
        pass


# ============================================================================
# Phase 5: Document Reconstruction
# ============================================================================

class DocumentReconstructionPhase(ABC):
    """
    Abstract base class cho Document Reconstruction.
    Sắp xếp và ghép nối các regions thành document cuối cùng (markdown, html, etc.)
    """

    @abstractmethod
    def reconstruct(
        self,
        regions: List[Tuple[int, str, Any]],
        output_format: str = "markdown"
    ) -> str:
        """
        Ghép nối các regions đã xử lý thành document.

        Args:
            regions: List of (y_position, content, bbox) tuples
            output_format: "markdown", "html", "plain", etc.

        Returns:
            Document cuối cùng dạng string
        """
        pass


# ============================================================================
# Default Implementations (No-op)
# ============================================================================

class NoOpPostProcessing(PostProcessingPhase):
    """
    Default implementation: không xử lý gì, trả về text nguyên bản.
    Sử dụng khi không cần post-processing.
    """
    def process(self, text: str, confidence: float, metadata: Optional[Dict] = None) -> str:
        return text


class SimpleMarkdownReconstruction(DocumentReconstructionPhase):
    """
    Default implementation: Ghép nối các regions bằng double newline.
    """
    def reconstruct(
        self,
        regions: List[Tuple[int, str, Any]],
        output_format: str = "markdown"
    ) -> str:
        """Simple concatenation with double newline"""
        if output_format != "markdown":
            raise NotImplementedError(f"Format {output_format} not supported")

        return "\n\n".join([item[1] for item in regions])
