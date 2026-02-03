"""
Document Processing Pipeline

Pipeline class orchestrates tất cả các phases để xử lý document.
Cho phép dễ dàng thử nghiệm bằng cách swap các phase implementations.

Usage Example:
    from deepdoc_vietocr.pipeline import DocumentPipeline
    from deepdoc_vietocr.implementations import (
        DocLayoutYOLOAnalyzer,
        PaddleOCRTextDetector,
        VietOCRRecognizer,
        VietnameseTextPostProcessor,
        SmartMarkdownReconstruction
    )

    # Create pipeline với custom phases
    pipeline = DocumentPipeline(
        layout_analyzer=DocLayoutYOLOAnalyzer(),
        text_detector=PaddleOCRTextDetector(),
        text_recognizer=VietOCRRecognizer(),
        post_processor=VietnameseTextPostProcessor(),
        reconstructor=SmartMarkdownReconstruction()
    )

    # Process image
    from PIL import Image
    img = Image.open("document.jpg")
    markdown = pipeline.process(img)
"""

import os
import time
import logging
from typing import Optional, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw

from .phases import (
    LayoutAnalysisPhase,
    TextDetectionPhase,
    TextRecognitionPhase,
    PostProcessingPhase,
    DocumentReconstructionPhase,
    NoOpPostProcessing,
    SimpleMarkdownReconstruction,
)
from . import LayoutRecognizer, TableStructureRecognizer


class DocumentPipeline:
    """
    Main pipeline orchestrator cho document processing.

    Coordinates các phases để xử lý document từ image sang markdown.
    Mỗi phase có thể được swap độc lập để thử nghiệm implementations khác.
    """

    def __init__(
        self,
        layout_analyzer: LayoutAnalysisPhase,
        text_detector: TextDetectionPhase,
        text_recognizer: TextRecognitionPhase,
        post_processor: Optional[PostProcessingPhase] = None,
        reconstructor: Optional[DocumentReconstructionPhase] = None,
        threshold: float = 0.5,
        max_workers: int = 4,
    ):
        """
        Initialize pipeline với các phase implementations.

        Args:
            layout_analyzer: Phase phân tích layout
            text_detector: Phase detect text boxes
            text_recognizer: Phase nhận dạng text
            post_processor: Phase xử lý hậu kỳ (optional, default: NoOp)
            reconstructor: Phase ghép nối document (optional, default: Simple)
            threshold: Detection threshold
            max_workers: Number of parallel workers for table processing
        """
        self.layout_analyzer = layout_analyzer
        self.text_detector = text_detector
        self.text_recognizer = text_recognizer
        self.post_processor = post_processor or NoOpPostProcessing()
        self.reconstructor = reconstructor or SimpleMarkdownReconstruction()

        self.threshold = threshold
        self.max_workers = max_workers

        # Initialize table recognizer (specialized component)
        self.table_recognizer = TableStructureRecognizer()

        logging.info("✓ DocumentPipeline initialized with custom phases")

    def process(
        self,
        image: Image.Image,
        img_name: str = "page",
        figure_save_dir: str = "figures",
        image_link_prefix: str = "figures",
        debug_path: Optional[str] = None,
    ) -> str:
        """
        Process một document image thành markdown.

        Args:
            image: PIL Image cần xử lý
            img_name: Base name cho figures
            figure_save_dir: Directory lưu figures
            image_link_prefix: Prefix cho image links trong markdown
            debug_path: Path để lưu debug visualization (optional)

        Returns:
            Markdown string
        """
        start_time = time.time()

        # Ensure figure directory exists
        os.makedirs(figure_save_dir, exist_ok=True)

        # Prepare visualization if debug requested
        vis_draw = None
        vis_img = None
        if debug_path:
            vis_img = image.copy()
            vis_draw = ImageDraw.Draw(vis_img)

        # ====================================================================
        # PHASE 1: Layout Analysis
        # ====================================================================
        logging.info("Phase 1: Running Layout Analysis...")
        layouts = self.layout_analyzer.analyze(image, threshold=self.threshold)
        logging.info(f"✓ Detected {len(layouts)} layout regions")

        # Collection for final output
        region_and_pos = []

        # Create mask for leftover detection
        mask = Image.new("1", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Collect regions by type for batch processing
        text_regions_batch = []  # [(bbox, label, y_pos, region_index), ...]
        table_regions_batch = []  # [(region, y_pos, region_index), ...]

        # ====================================================================
        # Process Each Layout Region
        # ====================================================================
        for i, region in enumerate(layouts):
            bbox = region["bbox"]
            label = region["type"]
            score = region["score"]
            y_pos = bbox[1]

            # Draw debug visualization
            if vis_draw:
                color = "red" if label in ["table", "figure", "equation"] else "blue"
                vis_draw.rectangle(bbox, outline=color, width=3)
                vis_draw.text((bbox[0], bbox[1]), f"{label} ({score:.2f})", fill=color)

            # Mark region as processed
            draw.rectangle(bbox, fill=1)

            # Route to appropriate handler
            if label == "table":
                table_regions_batch.append((region, y_pos, i))

            elif label == "figure":
                # Save figure image
                fig_filename = f"{img_name}_fig_{i}.jpg"
                fig_path = os.path.join(figure_save_dir, fig_filename)
                md_link_path = f"{image_link_prefix}/{fig_filename}" if image_link_prefix else fig_filename

                try:
                    crop_img = image.crop(bbox)
                    crop_img.save(fig_path)
                    region_and_pos.append((y_pos, f"![Figure]({md_link_path})", bbox))
                except Exception as e:
                    logging.error(f"Failed to save figure {fig_path}: {e}")

                # Also run OCR on figure region (might contain misclassified text)
                text_regions_batch.append((bbox, "figure_text", y_pos + 1, i))

            else:
                # Text-based regions: title, text, header, footer, captions, equation
                text_regions_batch.append((bbox, label, y_pos, i))

        # ====================================================================
        # PHASE 2+3: Batch Text Processing (Detection + Recognition)
        # ====================================================================
        if text_regions_batch:
            logging.info(f"Phase 2+3: Processing {len(text_regions_batch)} text regions...")
            try:
                text_results = self._process_text_regions_batch(
                    image, text_regions_batch
                )
                region_and_pos.extend(text_results)
                logging.info(f"✓ Processed {len(text_results)} text regions")
            except Exception as e:
                logging.error(f"Error in batch text processing: {e}", exc_info=True)

        # ====================================================================
        # Table Processing (Parallel)
        # ====================================================================
        if table_regions_batch:
            logging.info(f"Processing {len(table_regions_batch)} table regions...")
            try:
                table_results = self._process_tables_parallel(image, table_regions_batch)
                for y_pos, markdown in table_results:
                    if markdown and markdown.strip():
                        region_and_pos.append((y_pos, markdown, None))
                logging.info(f"✓ Processed {len(table_results)} tables")
            except Exception as e:
                logging.error(f"Error in table processing: {e}", exc_info=True)

        # ====================================================================
        # Leftover OCR (Undetected Areas)
        # ====================================================================
        inv_mask = mask.point(lambda p: 1 - p)
        if inv_mask.getbbox():
            leftover_text = self._process_leftover_regions(
                image, inv_mask, layouts, vis_draw
            )
            if leftover_text:
                lx0, ly0, lx1, ly1 = inv_mask.getbbox()
                region_and_pos.append((ly0, leftover_text, None))

        # ====================================================================
        # Save Debug Visualization
        # ====================================================================
        if debug_path and vis_img:
            try:
                vis_img.save(debug_path)
                logging.info(f"✓ Saved debug visualization to: {debug_path}")
            except Exception as e:
                logging.error(f"Failed to save debug image: {e}")

        # ====================================================================
        # PHASE 5: Document Reconstruction
        # ====================================================================
        logging.info("Phase 5: Reconstructing document...")
        markdown = self.reconstructor.reconstruct(region_and_pos, output_format="markdown")

        elapsed = time.time() - start_time
        logging.info(f"✓ Pipeline completed in {elapsed:.2f} seconds")

        return markdown

    def _process_text_regions_batch(
        self,
        image: Image.Image,
        text_regions_batch: List[Tuple[List[int], str, int, int]]
    ) -> List[Tuple[int, str, List[int]]]:
        """
        Process batch of text regions với text detection + recognition.

        Args:
            image: PIL Image
            text_regions_batch: List of (bbox, label, y_pos, index) tuples

        Returns:
            List of (y_pos, formatted_text, bbox) tuples
        """
        results = []

        # Crop all regions
        cropped_images = [np.array(image.crop(bbox)) for bbox, _, _, _ in text_regions_batch]

        # Batch OCR: Detect + Recognize
        try:
            batch_ocr_results = self._batch_ocr(cropped_images)

            # PHASE 4: Post-process each result
            for (bbox, label, y_pos, idx), ocr_results in zip(text_regions_batch, batch_ocr_results):
                # Extract text from OCR results
                texts = []
                for _, (text, confidence) in ocr_results:
                    if text:
                        # Apply post-processing
                        processed_text = self.post_processor.process(
                            text, confidence, metadata={"label": label, "bbox": bbox}
                        )
                        texts.append(processed_text)

                text_content = "\n".join(texts)
                if not text_content.strip():
                    continue

                # Apply formatting based on label
                formatted_text = self._apply_formatting(text_content, label)
                results.append((y_pos, formatted_text, bbox))

        except Exception as e:
            logging.error(f"Batch OCR processing failed: {e}", exc_info=True)

        return results

    def _batch_ocr(self, image_list: List[np.ndarray]) -> List[List[Tuple[Any, Tuple[str, float]]]]:
        """
        Batch OCR processing: Detect + Recognize tất cả images.

        Strategy:
        1. Detect text boxes in all images
        2. Collect ALL text boxes from all images
        3. Batch recognize ALL boxes at once
        4. Map results back to images

        Returns:
            List of OCR results, one per image
        """
        if not image_list:
            return []

        all_boxes_info = []  # [(image_idx, box, img_crop), ...]
        results = [[] for _ in range(len(image_list))]

        # Phase 2: Detect boxes in all images
        for img_idx, img_array in enumerate(image_list):
            dt_boxes, _ = self.text_detector.detect(img_array)

            if dt_boxes is None:
                continue

            # Sort boxes for reading order
            dt_boxes = self._sorted_boxes(dt_boxes)

            # Crop all detected boxes
            for box in dt_boxes:
                img_crop = self._get_rotate_crop_image(img_array, box)
                all_boxes_info.append((img_idx, box, img_crop))

        # Phase 3: Batch recognize ALL boxes at once
        if all_boxes_info:
            all_crops = [crop for _, _, crop in all_boxes_info]
            rec_results, _ = self.text_recognizer.recognize(all_crops)

            # Map results back to original images
            for (img_idx, box, _), (text, score) in zip(all_boxes_info, rec_results):
                # Filter low confidence results
                if score >= 0.5:  # TODO: Make this configurable
                    results[img_idx].append((box.tolist() if hasattr(box, 'tolist') else box, (text, score)))

        return results

    def _process_tables_parallel(
        self,
        image: Image.Image,
        table_regions_batch: List[Tuple[dict, int, int]]
    ) -> List[Tuple[int, str]]:
        """Process multiple tables in parallel."""
        def process_single_table(args):
            region, y_pos, idx = args
            try:
                markdown = self._extract_table_markdown(image, region)
                return (y_pos, markdown)
            except Exception as e:
                logging.error(f"Error processing table {idx}: {e}")
                return (y_pos, "")

        # Single table: no parallelization needed
        if len(table_regions_batch) == 1:
            return [process_single_table(table_regions_batch[0])]

        # Parallel processing
        results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(table_regions_batch))) as executor:
            future_to_table = {
                executor.submit(process_single_table, table_data): table_data
                for table_data in table_regions_batch
            }

            for future in as_completed(future_to_table):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    table_data = future_to_table[future]
                    logging.error(f"Table processing failed for region {table_data[2]}: {e}")

        return results

    def _extract_table_markdown(self, image: Image.Image, table_region: dict) -> str:
        """Extract table as markdown (uses specialized table recognizer)."""
        import re

        bbox = table_region["bbox"]
        table_img = image.crop(bbox)

        tb_cpns = self.table_recognizer([table_img])[0]

        # Run OCR on table image
        table_ocr_results = self._batch_ocr([np.array(table_img)])
        boxes = table_ocr_results[0] if table_ocr_results else []

        # Sort and clean up boxes
        boxes = LayoutRecognizer.sort_Y_firstly(
            [{
                "x0": b[0][0], "x1": b[1][0],
                "top": b[0][1], "text": t[0],
                "bottom": b[-1][1],
                "layout_type": "table",
                "page_number": 0
            } for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
            np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3 if boxes else 10
        )

        if not boxes:
            return ""

        def gather(kwd, fzy=10, ption=0.6):
            eles = LayoutRecognizer.sort_Y_firstly(
                [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
            eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
            return LayoutRecognizer.sort_Y_firstly(eles, 0)

        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in tb_cpns if re.match(r"table column$", r["label"])], key=lambda x: x["x0"])
        clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

        for b in boxes:
            self._map_cell_to_structure(b, rows, headers, clmns, spans)

        return TableStructureRecognizer.construct_table(boxes, markdown=True)

    def _process_leftover_regions(
        self,
        image: Image.Image,
        inv_mask: Image.Image,
        layouts: List[dict],
        vis_draw: Optional[ImageDraw.ImageDraw]
    ) -> Optional[str]:
        """Process leftover areas not detected by layout analysis."""
        lx0, ly0, lx1, ly1 = inv_mask.getbbox()
        leftover_area = (lx1 - lx0) * (ly1 - ly0)
        total_area = image.size[0] * image.size[1]
        leftover_ratio = leftover_area / total_area if total_area > 0 else 0

        # Skip if too small
        if leftover_ratio < 0.03:
            logging.info(f"Skipping leftover OCR: only {leftover_ratio:.1%} of image area")
            return None

        # Create leftover image
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        leftover_img = Image.composite(image, white_bg, inv_mask)
        leftover_crop = leftover_img.crop((lx0, ly0, lx1, ly1))

        # Check if mostly white
        leftover_array = np.array(leftover_crop)
        white_pixels = np.sum(np.all(leftover_array > 240, axis=-1))
        total_pixels = leftover_array.shape[0] * leftover_array.shape[1]
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 1.0

        force_ocr = len(layouts) == 0
        if white_ratio > 0.99 and not force_ocr:
            logging.info(f"Skipping leftover OCR: {white_ratio:.1%} white pixels")
            return None

        # Draw debug
        if vis_draw:
            vis_draw.rectangle((lx0, ly0, lx1, ly1), outline="green", width=2)
            vis_draw.text((lx0, ly0), "leftover", fill="green")

        # Run OCR on leftover
        try:
            ocr_results = self._batch_ocr([np.array(leftover_crop)])[0]
            texts = [t[0] for _, t in ocr_results if t and t[0]]
            leftover_text = "\n".join(texts)
            return leftover_text if leftover_text.strip() else None
        except Exception as e:
            logging.error(f"Error processing leftovers: {e}")
            return None

    # Helper methods
    def _apply_formatting(self, text: str, label: str) -> str:
        """Apply markdown formatting based on region label."""
        if label == "title":
            return f"# {text}"
        elif label in ["header", "footer"]:
            return f"_{text}_"
        elif label in ["figure caption", "table caption"]:
            return f"*{text}*"
        elif label == "equation":
            return f"$$ {text} $$"
        return text

    def _map_cell_to_structure(self, b, rows, headers, clmns, spans):
        """Map cell to table structure (helper for table extraction)."""
        ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    def _sorted_boxes(self, dt_boxes):
        """Sort text boxes by position (từ OCR class)."""
        num_boxes = len(dt_boxes)
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                   (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def _get_rotate_crop_image(self, img, points):
        """Rotate and crop image based on points (từ OCR class)."""
        import cv2
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
