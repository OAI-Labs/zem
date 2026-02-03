import os
import sys
import json
import httpx
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("instruction", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

@server.tool()
def generate_qa_pairs(
    data: Any = None,
    base_url: str = "http://localhost:8000/v1",
    model: str = "default",
    num_pairs: int = 3,
    text_column: str = "text",
    prompt_template: str = "Dựa trên văn bản pháp luật sau, hãy tạo {num_pairs} cặp Câu hỏi và Trả lời chi tiết. Trả về định dạng JSON list: [{{'q': '...', 'a': '...'}}]. Văn bản: {text}"
) -> Any:
    raw_data = server.get_data(data)
    if not raw_data: return []
    
    # Handle wrapped data from previous steps
    if isinstance(raw_data, dict) and 'data' in raw_data:
        actual_items = raw_data['data']
    else:
        actual_items = raw_data
    
    logger.info(f"InstructionGen: Connecting to vLLM at {base_url} for {len(actual_items)} items")
    
    processed_items = []
    with httpx.Client(timeout=60.0) as client:
        for item in actual_items:
            text = str(item.get(text_column, ""))
            if not text: continue
            
            prompt = prompt_template.format(num_pairs=num_pairs, text=text)
            try:
                response = client.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3
                    }
                )
                if response.status_code == 200:
                    raw_content = response.json()["choices"][0]["message"]["content"]
                    start = raw_content.find('[')
                    end = raw_content.rfind(']') + 1
                    if start != -1 and end != -1:
                        item["instructions"] = json.loads(raw_content[start:end])
                else:
                    item["instructions"] = []
            except Exception:
                item["instructions"] = []
            processed_items.append(item)
            
    return server.save_output(processed_items)

@server.tool()
def complexity_scorer(
    data: Any = None, 
    text_column: str = "text"
) -> Any:
    raw_data = server.get_data(data)
    if not raw_data: return []
    
    if isinstance(raw_data, dict) and 'data' in raw_data:
        actual_items = raw_data['data']
    else:
        actual_items = raw_data
    
    logger.info(f"InstructionGen: Scoring complexity for {len(actual_items)} items")
    
    for item in actual_items:
        text = str(item.get(text_column, ""))
        words = text.split()
        unique_words = set(words)
        score = min(1.0, (len(unique_words) / 100) * (len(words) / 500))
        item["complexity_score"] = round(float(score), 2)
        
    return server.save_output(actual_items)

if __name__ == "__main__":
    server.run()
