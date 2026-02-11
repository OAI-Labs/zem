"""
File: /app/xfmr-zem/src/xfmr_zem/servers/doc_parser/corrector.py
Description: Vietnamese Legal Text Corrector.
"""

import sys
import os
import re
from typing import Any, Dict, Optional, List, Union
from loguru import logger
from tqdm import tqdm

# Import Zem
from xfmr_zem.server import ZemServer
from xfmr_zem.servers.corrector.engines import CorrectorEngineFactory

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("corrector", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

# Global engine instance
_engine = None

def process_markdown_text(markdown_content: str, engine) -> str:
    """Parse and reconstruct Markdown, correcting text segments using the engine."""
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
                corrected_text = engine.correct(temp_text)
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

    for line in lines:
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

@server.tool()
def correct_text(
    data: Any,
    field: str = "text",
    engine: str = "huggingface",
    model_id: str = "protonx-models/protonx-legal-tc",
    verbose: bool = True # <--- [ADD 1] Thêm tham số verbose, mặc định là True
) -> Any:
    """
    Corrects markdown text using the configured LM engine.
    """
    items = server.get_data(data)
    items = items if isinstance(items, list) else [items]
    
    engine_name = engine
    # Lưu ý: Sửa lại cách gọi Factory cho đúng static method (nếu Factory là static)
    corrector_engine = CorrectorEngineFactory.get_engine(engine, model_id=model_id)
    
    if not corrector_engine:
        logger.error(f"Failed to initialize engine: {engine_name}")
        # Trả về items gốc nếu không load được engine
        return server.save_output(items)

    all_results = []
    
    # <--- [ADD 2] Sử dụng disable=not verbose để điều khiển tqdm
    for item in tqdm(items, desc="Processing Items", disable=not verbose):
        result_entry = {
            "text": "",
            "engine": engine_name,
            "metadata": {"model_id": model_id, "field": field}
        }
        try:
            if not isinstance(item, dict) or field not in item:
                # Nếu item lỗi, log warning nhưng không crash luồng
                logger.warning(f"Skipping item: Field '{field}' missing")
                result_entry["text"] = str(item)
                all_results.append(result_entry)
                continue

            raw_text = item.get(field, "")
            
            # Kiểm tra raw_text
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)

            if not raw_text.strip():
                result_entry["text"] = raw_text
            else:
                # Gọi hàm xử lý với engine đã khởi tạo
                corrected_text = process_markdown_text(raw_text, corrector_engine)
                result_entry["text"] = corrected_text
                
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            result_entry["text"] = item.get(field, "") if isinstance(item, dict) else ""
            
        all_results.append(result_entry)

    if verbose:
        logger.info(f"Processed {len(all_results)} items.")
        
    return server.save_output(all_results)
    
if __name__ == "__main__":
    server.run()
