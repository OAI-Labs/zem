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


@server.tool()
def protonx_legal_markdown_corrector(
    data: Any,
    model_path: Optional[str] = None,
    tokenizer_config: Dict[str, Any] = None,
    seq2seq_config: Dict[str, Any] = None,
) -> Any:
    """
    Nhận vào văn bản, tự động load model, xử lý markdown và trả về văn bản đã sửa lỗi.
    """

    items = server.get_data(data)
    
    # Extract texts from dictionaries if present
    texts = []
    items = items if isinstance(items, list) else [items]
    for item in items:
        if isinstance(item, dict):
            found = False
            for key in item:
                if "markdown" in key:
                    texts.append(item[key])
                    found = True
                    break
            if not found:
                texts.append("")
        else:
            texts.append(item)

    _model_cache = {"model": None, "tokenizer": None, "device": None}
    logger.info(f"Text to process: {texts}")

    # =========================================================================
    # PHẦN 1: LOAD MODEL
    # =========================================================================
    
    # Load configuration
    tokenizer_config = server.parameters['protonx_legal_text_corrector']['tokenizer_config']
    seq2seq_config = server.parameters['protonx_legal_text_corrector']['seq2seq_config']
    model_path = server.parameters['protonx_legal_text_corrector']['model_path']

    # Kiểm tra cache, nếu chưa có model thì load ngay lập tức
    if _model_cache["model"] is None:
        
        if model_path:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🚀 Loading VietnameseCorrector model from: {model_path} ({device})")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
                
                _model_cache["tokenizer"] = tokenizer
                _model_cache["model"] = model
                _model_cache["device"] = device
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # Nếu lỗi load model, trả về input gốc
                return server.save_output(data if isinstance(data, list) else [data])
        else:
            logger.warning("No model_path found in parameters.")

    # =========================================================================
    # PHẦN 2: Inner Functions
    # =========================================================================

    def correct_text(text: str) -> str:
        """Hàm nội bộ: Chạy model AI để sửa text thuần"""
        if _model_cache["model"] is None or not text or not text.strip():
            return text

        tokenizer = _model_cache["tokenizer"]
        model = _model_cache["model"]
        device = _model_cache["device"]

        # Tokenize
        tokenizer_kwargs = tokenizer_config.copy()
        inputs = tokenizer(
            text,
            truncation=tokenizer_kwargs.get('truncation', True),
            max_length=tokenizer_kwargs.get('max_length', 512),
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        gen_kwargs = seq2seq_config.copy()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                num_beams=gen_kwargs.get('num_beams', 5),
                max_new_tokens=gen_kwargs.get('max_new_tokens', 512),
                early_stopping=gen_kwargs.get('early_stopping', True),
                return_dict_in_generate=True
            )

        # Decode
        sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
        corrected_batch = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return corrected_batch[0]

    def process_markdown_text(markdown_content: str) -> str:
        """Hàm nội bộ: Parse và tái tạo Markdown"""
        if not markdown_content:
            return ""

        # Regex patterns definition
        inline_code_pattern = re.compile(r'(`[^`]+`)')
        link_pattern = re.compile(r'(\[[^\]]+\]\([^)]+\))')
        image_pattern = re.compile(r'(!\[[^\]]*\]\([^)]+\))')
        html_tag_pattern = re.compile(r'(<[^>]+>)')
        url_pattern = re.compile(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)')

        def _process_segment(text):
            if not text or not text.strip():
                return text

            protected_segments = []
            def protect_match(match):
                protected_segments.append(match.group(0))
                return f"__PROTECTED_{len(protected_segments)-1}__"

            # 1. Protect Syntax
            temp_text = inline_code_pattern.sub(protect_match, text)
            temp_text = image_pattern.sub(protect_match, temp_text)
            temp_text = link_pattern.sub(protect_match, temp_text)
            temp_text = url_pattern.sub(protect_match, temp_text)
            temp_text = html_tag_pattern.sub(protect_match, temp_text)

            # 2. Inference
            if temp_text.strip() and not re.match(r'^__PROTECTED_\d+__$', temp_text.strip()):
                 corrected_text = correct_text(temp_text)
            else:
                 corrected_text = temp_text

            # 3. Restore Syntax
            for i, segment in enumerate(protected_segments):
                corrected_text = corrected_text.replace(f"__PROTECTED_{i}__", segment)
            return corrected_text

        # Line processing loop
        lines = markdown_content.split('\n')
        result_lines = []
        in_code_block = False

        # Patterns for lines that have a prefix to preserve (Header, Blockquote, List)
        line_patterns = [
            re.compile(r'^(#{1,6}\s+)(.*)'),           # Headers
            re.compile(r'^(\s*>\s+)(.*)'),             # Blockquotes
            re.compile(r'^(\s*(?:[-*+]|\d+\.)\s+)(.*)') # Lists
        ]

        for line in tqdm(lines, desc="Correcting Markdown"):
            # Detect Code Block
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            if in_code_block:
                result_lines.append(line)
                continue

            # Skip empty/separators
            if not line.strip() or re.match(r'^[-*_]{3,}$', line.strip()):
                result_lines.append(line)
                continue

            # Check for standard prefix patterns
            matched = False
            for pattern in line_patterns:
                match = pattern.match(line)
                if match:
                    prefix, content = match.groups()
                    corrected = _process_segment(content)
                    result_lines.append(f"{prefix}{corrected}")
                    matched = True
                    break
            
            if matched:
                continue

            # Tables
            if '|' in line and re.match(r'^\s*\|.*\|\s*$', line):
                parts = line.split('|')
                corrected_parts = []
                for part in parts:
                    if re.match(r'^\s*:?-+:?\s*$', part):
                        corrected_parts.append(part)
                    else:
                        corrected_parts.append(_process_segment(part))
                result_lines.append('|'.join(corrected_parts))
                continue

            # Normal Text
            result_lines.append(_process_segment(line))

        return '\n'.join(result_lines)

    # # =========================================================================
    # # PHẦN 3: Main Executor
    # # =========================================================================
    
    all_results = []
    for text in texts:
        try:
            if not text:
                all_results.append({"markdown": ""})
            else:
                
                # Gọi hàm xử lý markdown nội bộ
                corrected_text = process_markdown_text(text)
                all_results.append({"markdown": corrected_text})
        except Exception as e:
            logger.error(f"Error correcting text: {e}")
            all_results.append({"markdown": f"Error correcting text: {e}"})
    return server.save_output(all_results)


if __name__ == "__main__":
    server.run()