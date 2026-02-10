"""
File: /app/xfmr-zem/src/xfmr_zem/servers/doc_parser/corrector.py
Description: Vietnamese Legal Text Corrector.
"""

import sys
import os
import re
import torch
from typing import Any, Dict, Optional, List, Union
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Import Zem
from xfmr_zem.server import ZemServer

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("corrector", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

# Global cache for model to avoid reloading
_model_cache = {"model": None, "tokenizer": None, "device": None}

@server.tool()
def protonx_legal_markdown_corrector(
    data: Any,
    field: str = "markdown",  # Trường input cần lấy dữ liệu để xử lý
    model_path: Optional[str] = None,
    tokenizer_config: Dict[str, Any] = None,
    seq2seq_config: Dict[str, Any] = None,
) -> Any:
    """
    Nhận vào items, lấy text từ `field`, xử lý và trả về list các dict {"markdown": "kết quả"}.
    """

    items = server.get_data(data)
    items = items if isinstance(items, list) else [items]

    # =========================================================================
    # PHẦN 1: LOAD MODEL
    # =========================================================================
    
    tokenizer_params = server.parameters['protonx_legal_text_corrector']['tokenizer_config']
    seq2seq_params = server.parameters['protonx_legal_text_corrector']['seq2seq_config']
    model_path_param = server.parameters['protonx_legal_text_corrector']['model_path']

    if _model_cache["model"] is None:
        if model_path_param:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🚀 Loading VietnameseCorrector model from: {model_path_param} ({device})")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path_param)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path_param).to(device)
                
                _model_cache["tokenizer"] = tokenizer
                _model_cache["model"] = model
                _model_cache["device"] = device
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # Nếu lỗi load model, trả về danh sách rỗng hoặc lỗi tương ứng
                return server.save_output([{"markdown": ""} for _ in items])
        else:
            logger.warning("No model_path found in parameters.")

    # =========================================================================
    # PHẦN 2: Inner Functions
    # =========================================================================

    def correct_text(text: str) -> str:
        """Hàm nội bộ: Chạy model AI để sửa text thuần"""
        if _model_cache["model"] is None or not text or not isinstance(text, str) or not text.strip():
            return text

        tokenizer = _model_cache["tokenizer"]
        model = _model_cache["model"]
        device = _model_cache["device"]

        inputs = tokenizer(
            text,
            truncation=tokenizer_params.get('truncation', True),
            max_length=tokenizer_params.get('max_length', 512),
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                num_beams=seq2seq_params.get('num_beams', 5),
                max_new_tokens=seq2seq_params.get('max_new_tokens', 512),
                early_stopping=seq2seq_params.get('early_stopping', True)
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process_markdown_text(markdown_content: str) -> str:
        """Hàm nội bộ: Parse và tái tạo Markdown"""
        if not markdown_content or not isinstance(markdown_content, str):
            return ""

        # Regex patterns definition
        inline_code_pattern = re.compile(r'(`[^`]+`)')
        link_pattern = re.compile(r'(\[[^\]]+\]\([^)]+\))')
        image_pattern = re.compile(r'(!\[[^\]]*\]\([^)]+\))')
        html_tag_pattern = re.compile(r'(<[^>]+>)')
        url_pattern = re.compile(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)')

        def _process_segment(text):
            if not text or not text.strip(): return text
            protected_segments = []
            def protect_match(match):
                protected_segments.append(match.group(0))
                return f"__PROTECTED_{len(protected_segments)-1}__"

            temp_text = inline_code_pattern.sub(protect_match, text)
            temp_text = image_pattern.sub(protect_match, temp_text)
            temp_text = link_pattern.sub(protect_match, temp_text)
            temp_text = url_pattern.sub(protect_match, temp_text)
            temp_text = html_tag_pattern.sub(protect_match, temp_text)

            if temp_text.strip() and not re.match(r'^__PROTECTED_\d+__$', temp_text.strip()):
                 corrected_text = correct_text(temp_text)
            else:
                 corrected_text = temp_text

            for i, segment in enumerate(protected_segments):
                corrected_text = corrected_text.replace(f"__PROTECTED_{i}__", segment)
            return corrected_text

        lines = markdown_content.split('\n')
        result_lines = []
        in_code_block = False
        
        line_patterns = [
            re.compile(r'^(#{1,6}\s+)(.*)'), 
            re.compile(r'^(\s*>\s+)(.*)'), 
            re.compile(r'^(\s*(?:[-*+]|\d+\.)\s+)(.*)')
        ]

        for line in tqdm(lines, desc = "Correcting lines Progress"):
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            if in_code_block or not line.strip() or re.match(r'^[-*_]{3,}$', line.strip()):
                result_lines.append(line)
                continue

            matched = False
            for pattern in line_patterns:
                match = pattern.match(line)
                if match:
                    prefix, content = match.groups()
                    result_lines.append(f"{prefix}{_process_segment(content)}")
                    matched = True
                    break
            if matched: continue

            if '|' in line and re.match(r'^\s*\|.*\|\s*$', line):
                parts = line.split('|')
                res_p = [p if re.match(r'^\s*:?-+:?\s*$', p) else _process_segment(p) for p in parts]
                result_lines.append('|'.join(res_p))
                continue

            result_lines.append(_process_segment(line))
        return '\n'.join(result_lines)

    # =========================================================================
    # PHẦN 3: Main Executor (Updated logic)
    # =========================================================================
    
    all_results = []
    
    for item in tqdm(items, desc="Processing Items"):
        try:
            # Logic: Cố gắng lấy dữ liệu từ field. 
            # Nếu item không phải dict hoặc không có key -> nhảy vào except
            raw_text = item[field]
            
            # Xử lý văn bản
            corrected_text = process_markdown_text(raw_text)
            
            # Output ra list mới với key cố định là "markdown"
            all_results.append({"markdown": corrected_text})
            
        except Exception as e:
            # Log lỗi (ví dụ: KeyError nếu không tìm thấy field)
            logger.warning(f"Error extracting/processing field '{field}': {e}")
            
            # Nếu lỗi, trả về chuỗi rỗng cho key markdown
            all_results.append({"markdown": ""})

    return server.save_output(all_results)

if __name__ == "__main__":
    server.run()