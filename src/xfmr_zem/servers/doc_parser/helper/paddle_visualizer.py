import json
import logging
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

class PaddleVisualizer:
    def __init__(self):
        pass

    def draw_bounding_boxes(self, image_path, json_path, save_path=None):
        """
        Vẽ bounding box và label từ file JSON kết quả của PaddleOCR lên ảnh gốc.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            return None

        draw = ImageDraw.Draw(image)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            return image

        # Parse JSON content
        res_list = data.get('parsing_res_list', [])
        
        for block in res_list:
            points = block.get('block_polygon_points', [])
            label = block.get('block_label', '')
            
            if points:
                poly_points = [tuple(p) for p in points]
                draw.polygon(poly_points, outline="red", width=3)
                if label:
                    draw.text(poly_points[0], label, fill="blue")
        
        if save_path:
            image.save(save_path)
            
        return image
