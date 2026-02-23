# 📚 Zem SDK — Tài liệu Nghiên cứu Chuyên sâu

> **Version phân tích**: v0.3.0  
> **Nguồn**: [github.com/OAI-Labs/xfmr-zem](https://github.com/OAI-Labs/xfmr-zem)  
> **Ngày viết**: 2026-02-23

---

## 1. Tổng quan Kiến trúc

Zem (xfmr-zem) là một **unified data pipeline framework** kết hợp hai lớp công nghệ:

| Lớp | Công nghệ | Vai trò |
|-----|-----------|---------|
| **Orchestration** | ZenML | Quản lý pipeline, artifact tracking, caching, DAG visualization |
| **Processing Units** | MCP (Model Context Protocol) via FastMCP | Các server xử lý độc lập, giao tiếp qua JSON-RPC over stdio |

```
pipeline.yaml
     │
     ▼
PipelineClient  ──── đọc config, resolve servers, inject params
     │
     ▼
ZenML Pipeline  ──── xây dựng DAG động từ các bước YAML
     │
     ▼
mcp_generic_step  ── gửi JSON-RPC tới MCP subprocess
     │
     ▼
ZemServer (FastMCP)  ── tool thực thi, trả về file reference (parquet/jsonl)
```

**Cơ chế truyền data**: Để tránh gửi dữ liệu lớn qua JSON-RPC, các server **lưu kết quả ra file tạm** (`/tmp/zem_artifacts/`) và trả về `{"path": "...", "type": "parquet"}` — gọi là **Pass-by-Reference**.

---

## 2. Cài đặt từ PyPI

### 2.1. Cài đặt cơ bản

```bash
pip install xfmr-zem
```

Cài bản cơ bản chỉ gồm core engine + CLI `zem`, đủ để chạy pipeline với các built-in server.

### 2.2. Cài đặt có optional extras

Package cung cấp nhiều "extras" tùy theo nhu cầu:

```bash
# Chỉ ZenML
pip install "xfmr-zem[zenml]"

# NeMo Curator (GPU data curation)
pip install "xfmr-zem[nemo]"

# OCR (Tesseract, PaddleOCR, Qwen-VL)
pip install "xfmr-zem[ocr]"

# Voice / ASR (Whisper)
pip install "xfmr-zem[voice]"

# Web UI (FastAPI + uvicorn)
pip install "xfmr-zem[ui]"

# LLM evaluation (Opik)
pip install "xfmr-zem[evaluator]"

# Cài tất cả
pip install "xfmr-zem[all]"
```

> **Lưu ý**: Python `>=3.10, <3.13`. Khuyến nghị dùng `uv` hoặc `conda` để quản lý môi trường.

### 2.3. Yêu cầu bổ sung sau cài đặt

Sau khi cài package, phải khởi động ZenML store lần đầu:

```bash
# Khởi tạo ZenML local store (chỉ lần đầu)
zenml init

# (Tùy chọn) Khởi động ZenML Dashboard tại port 8871
zenml up --port 8871
```

Nếu dùng **Parallel Orchestrator**, phải đăng ký stack:

```bash
zenml orchestrator register parallel_orchestrator \
    --flavor=local \
    --type=parallel_local

zenml stack register parallel_stack \
    -o parallel_orchestrator \
    -a default

zenml stack set parallel_stack
```

---

## 3. Sử dụng CLI `zem`

Sau khi cài, lệnh `zem` sẵn có trên terminal:

```bash
# Xem thông tin framework
zem info

# Tạo project mới (bootstrap)
zem init my_project
cd my_project

# Liệt kê tools từ config
zem list-tools -c pipeline.yaml

# Chạy pipeline
zem run pipeline.yaml

# Chạy với parameters file riêng
zem run pipeline.yaml --params custom_params.yml

# Chạy với verbose/debug logging
zem run pipeline.yaml --verbose

# Mở ZenML dashboard
zem dashboard

# Preview artifact
zem preview <artifact_id> --limit 10
zem preview <artifact_id> --sample --limit 5

# So sánh 2 artifacts (diff mode)
zem preview <artifact_id_1> --id2 <artifact_id_2>

# Visual configurator (web UI)
zem explore --port 8878

# Cài model OCR
zem ocr install
```

---

## 4. Sử dụng SDK bằng Python API

### 4.1. PipelineClient — API chính

```python
from xfmr_zem import PipelineClient

# Chạy pipeline từ YAML config
client = PipelineClient("pipeline.yaml")
run = client.run()

print(f"Run: {run.name}, Status: {run.status}")
```

```python
# Với custom parameters file
client = PipelineClient(
    "pipeline.yaml",
    params_path="my_params.yml"
)
run = client.run()
```

```python
# Discover tools của tất cả servers trong config
client = PipelineClient("pipeline.yaml")
tools = client.discover_tools()
# tools: {"nemo": [...], "dj": [...]}
for server_name, tool_list in tools.items():
    print(f"\n{server_name}:")
    for tool in tool_list:
        print(f"  - {tool['name']}: {tool['description']}")
```

### 4.2. Pipeline YAML — Format đầy đủ

```yaml
# pipeline.yaml
name: my_pipeline_name

# (Tùy chọn) Định nghĩa parameters có thể dùng template {{ key }}
parameters:
  my_lang: "vi"
  my_min_len: 50

# Khai báo servers (tên_alias: đường_dẫn)
servers:
  # Built-in server (giải quyết từ package)
  nemo: servers/nemo_curator
  dj: servers/data_juicer
  llm: servers/llm
  profiler: servers/profiler
  sinks: servers/sinks
  
  # Custom server (relative tới file yaml)
  my_agent: servers/my_custom_server.py

# Định nghĩa các bước pipeline (thứ tự = thứ tự thực thi mặc định)
pipeline:
  - name: step1_ingest
    dj.clean_content:
      input:
        data:
          - text: "Hello world <b>HTML</b> content"
          - text: "Another document here"
        remove_html: true

  - name: step2_normalize
    nemo.normalize:
      input:
        data: "$step1_ingest"   # Tham chiếu output của step trước
        normalization: "NFC"

  - name: step3_filter
    dj.refining_filter:
      input:
        data: "$step2_normalize"
        min_len: "{{ my_min_len }}"   # Template substitution
      cache: false                   # Tắt cache cho step này
```

### 4.3. Parameters file (parameters.yml)

File `parameters.yml` đặt cùng thư mục với `pipeline.yaml` sẽ được tự động load:

```yaml
# parameters.yml
# Global parameters
my_min_len: 50
my_lang: "vi"

# Server-specific parameters (tên = tên alias trong YAML servers:)
nemo:
  normalization: "NFC"
  text_column: "content"

dj:
  remove_html: true
  min_len: 30
  
# Tool-specific parameters
ocr:
  extract_text:
    engine: "tesseract"
    temp_dir: /tmp/my_ocr
```

---

## 5. Built-in Servers — Danh sách đầy đủ

| Server alias | Path trong YAML | Tools |
|-------------|-----------------|-------|
| `nemo` | `servers/nemo_curator` | `normalize`, `quality_filter`, `exact_deduplication`, `fuzzy_deduplication`, `language_filter` |
| `dj` | `servers/data_juicer` | `clean_content`, `refining_filter`, `language_id` |
| `llm` | `servers/llm` | `mask_pii`, `classify_domain` |
| `profiler` | `servers/profiler` | `profile_data` |
| `sinks` | `servers/sinks` | `to_huggingface`, `to_vector_db` |
| `unstructured` | `servers/unstructured` | `parse_document`, `extract_tables` |
| `ocr` | `servers/ocr` | `extract_text` (engines: tesseract/paddle/qwen/viet) |
| `voice` | `servers/voice` | `transcribe` |
| `deduplication` | `servers/deduplication` | MinHash LSH + NER-based tools |
| `io` | `servers/io` | load/save jsonl, csv, parquet |
| `corrector` | `servers/corrector` | Text correction tools |
| `instruction_gen` | `servers/instruction_gen` | Instruction generation |

---

## 6. Cách thêm Custom Server vào Project khác

Đây là tính năng cốt lõi — cho phép mở rộng Zem bằng bất kỳ logic xử lý nào.

### 6.1. Cấu trúc project được khuyến nghị

```
my_project/
├── pipeline.yaml          # Config pipeline chính
├── parameters.yml         # Parameters (tự động load)
├── servers/
│   ├── my_classifier/
│   │   ├── server.py      # Server code
│   │   └── parameters.yml # Parameters riêng của server (optional)
│   └── my_enricher.py     # Hoặc single-file server
└── data/
    └── input.jsonl
```

### 6.2. Tạo custom server (chuẩn SOLID)

```python
# servers/my_classifier/server.py
import sys
from typing import Any, List, Dict, Optional
from xfmr_zem.server import ZemServer
from loguru import logger

# Bắt buộc: setup logger chỉ ra stderr để không can thiệp JSON-RPC
logger.remove()
logger.add(sys.stderr, level="INFO")

# Khởi tạo server — tên phải unique
server = ZemServer(
    name="my_classifier",
    # Tự động load parameters.yml trong cùng thư mục nếu không chỉ định
    # parameter_file="path/to/custom_params.yml"  # Tùy chọn
)

@server.tool()
def classify_text(
    data: Any,
    categories: List[str] = ["positive", "negative", "neutral"],
    text_column: str = "text",
    model: Optional[str] = None
) -> Any:
    """
    Phân loại văn bản vào các danh mục.
    
    Args:
        data: Input data (list, dict, file path, hoặc JSON string)
        categories: Danh sách danh mục
        text_column: Tên cột chứa văn bản
        model: Tên model (override từ parameters)
    """
    # server.get_data() tự xử lý mọi loại input:
    # - list of dicts, single dict, file path (.jsonl/.csv/.parquet), JSON string
    items = server.get_data(data)
    if not items:
        return []
    
    # Dùng parameters từ YAML nếu không truyền trực tiếp
    actual_model = model or server.parameters.get("model", "default_model")
    
    logger.info(f"Classifying {len(items)} items with model={actual_model}")
    
    for item in items:
        text = str(item.get(text_column, ""))
        # --- Logic xử lý của bạn ---
        # Ví dụ đơn giản:
        item["category"] = categories[len(text) % len(categories)]
        item["confidence"] = 0.95
    
    # QUAN TRỌNG: Dùng save_output() để trả về file reference
    # thay vì trả list trực tiếp (tránh quá tải JSON-RPC)
    return server.save_output(items, format="parquet")
    # format có thể là "parquet" hoặc "jsonl"

@server.tool()
def batch_enrich(
    data: Any,
    extra_field: str = "enriched",
    text_column: str = "text"
) -> Any:
    """Tool thứ 2 trong cùng server."""
    items = server.get_data(data)
    if not items:
        return []
    
    for item in items:
        item[extra_field] = f"Processed: {item.get(text_column, '')[:50]}"
    
    return server.save_output(items)

# BẮTBUỘC: Entry point để ZemClient spawn subprocess
if __name__ == "__main__":
    server.run()   # Mặc định transport="stdio"
```

### 6.3. Khai báo trong pipeline.yaml

```yaml
# pipeline.yaml
name: my_custom_pipeline

servers:
  # Built-in servers từ package
  dj: servers/data_juicer
  nemo: servers/nemo_curator
  
  # Custom servers của project
  classifier: servers/my_classifier        # Thư mục → tự append /server.py
  enricher: servers/my_enricher.py        # Single file

pipeline:
  - name: clean
    dj.clean_content:
      input:
        data:
          - text: "Raw document text here"
          - text: "Another document"

  - name: classify
    classifier.classify_text:
      input:
        data: "$clean"
        categories: ["Finance", "Legal", "Medical"]
        text_column: "text"

  - name: enrich
    enricher.batch_enrich:
      input:
        data: "$classify"
        extra_field: "enrichment_label"
```

### 6.4. Server Resolution Logic (thứ tự ưu tiên)

`PipelineClient` tìm server theo thứ tự sau:

1. **Relative to config file** — `<config_dir>/<path_str>` (ưu tiên nhất, dành cho project-local server)
2. **Internal package** — Nếu `path_str` bắt đầu bằng `servers/`, tìm trong `site-packages/xfmr_zem/servers/<name>/server.py`
3. **Project root** — `<project_root>/<path_str>`
4. **Directory auto-append** — Nếu path là thư mục, tự append `/server.py`

### 6.5. Cấu hình parameters cho custom server

**Cách 1**: File `parameters.yml` riêng trong thư mục server (tự động load):

```yaml
# servers/my_classifier/parameters.yml
model: "bert-base-multilingual"
text_column: "content"
batch_size: 32
```

**Cách 2**: Khai báo trong `parameters` của `pipeline.yaml`:

```yaml
# pipeline.yaml
parameters:
  classifier:           # Khớp với tên alias trong servers:
    model: "my-model"
    text_column: "body"
    classify_text:      # Tool-specific override
      categories: ["A", "B", "C"]
```

**Cách 3**: File `parameters.yml` riêng bên ngoài:

```bash
zem run pipeline.yaml --params production_params.yml
```

**Thứ tự merge** (sau ghi đè trước):
```
parameters.yml của server (base)
    ← parameters.yml bên cạnh pipeline.yaml
        ← parameters: trong pipeline.yaml
            ← --params custom_file.yml
                ← step-level input args (ưu tiên cao nhất)
```

### 6.6. Truy cập parameters bên trong server

```python
# server.py
server = ZemServer("my_classifier")

@server.tool()
def my_tool(data: Any, model: str = None) -> Any:
    # server.parameters chứa toàn bộ parameters đã merge
    effective_model = model or server.parameters.get("model", "default")
    batch_size = server.parameters.get("batch_size", 16)
    
    # Hoặc dot-notation:
    # server.parameters["classify_text"]["categories"]
```

---

## 7. Tính năng Nâng cao

### 7.1. DAG Dependencies — Cross-step References

```yaml
pipeline:
  # Hai branch độc lập (chạy song song nếu dùng parallel stack)
  - name: branch_a
    dj.clean_content:
      input:
        data: [{text: "Document A"}]

  - name: branch_b
    nemo.normalize:
      input:
        data: [{text: "Document B"}]

  # Merge từ branch_a
  - name: filter_a
    dj.refining_filter:
      input:
        data: "$branch_a"    # Tham chiếu bằng $step_name
        min_len: 5

  # Merge từ branch_b
  - name: filter_b
    nemo.quality_filter:
      input:
        data: "$branch_b"
        min_words: 1
```

### 7.2. Adaptive Caching

```yaml
pipeline:
  - name: expensive_step
    nemo.normalize:
      input:
        data: [...]
      cache: true    # Bật cache (default)

  - name: always_fresh
    profiler.profile_data:
      input:
        data: "$expensive_step"
      cache: false   # Luôn chạy lại
```

### 7.3. Template Variables

```yaml
parameters:
  input_file: "/data/corpus.jsonl"
  lang: "vi"
  output_repo: "my-org/my-dataset"

pipeline:
  - name: load
    io.load_jsonl:
      input:
        file_path: "{{ input_file }}"   # Substitution tự động

  - name: filter_lang
    nemo.language_filter:
      input:
        data: "$load"
        target_lang: "{{ lang }}"

  - name: push
    sinks.to_huggingface:
      input:
        data: "$filter_lang"
        repo_id: "{{ output_repo }}"
```

### 7.4. Verbose Logging

```bash
# Kích hoạt DEBUG logging cho tất cả servers
zem run pipeline.yaml --verbose

# Hoặc dùng biến môi trường
ZEM_VERBOSE=1 zem run pipeline.yaml
```

---

## 8. Ví dụ End-to-End: Project dùng Custom Server

### Bước 1: Cài thư viện

```bash
pip install xfmr-zem
zenml init
```

### Bước 2: Tạo cấu trúc project

```bash
zem init my_nlp_project
cd my_nlp_project
```

Hoặc tạo thủ công:

```
my_nlp_project/
├── pipeline.yaml
├── parameters.yml
└── servers/
    └── sentiment/
        ├── server.py
        └── parameters.yml
```

### Bước 3: Viết custom server

```python
# servers/sentiment/server.py
import sys
from typing import Any
from xfmr_zem.server import ZemServer
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("sentiment")

@server.tool()
def analyze(data: Any, text_column: str = "text") -> Any:
    """Phân tích cảm xúc văn bản."""
    items = server.get_data(data)
    for item in items:
        text = item.get(text_column, "").lower()
        # Logic đơn giản (thay bằng model thực)
        if any(w in text for w in ["tốt", "hay", "tuyệt"]):
            item["sentiment"] = "positive"
        elif any(w in text for w in ["tệ", "kém", "xấu"]):
            item["sentiment"] = "negative"
        else:
            item["sentiment"] = "neutral"
    return server.save_output(items)

if __name__ == "__main__":
    server.run()
```

### Bước 4: Viết pipeline.yaml

```yaml
name: sentiment_pipeline

servers:
  dj: servers/data_juicer
  sentiment: servers/sentiment

pipeline:
  - name: preprocess
    dj.clean_content:
      input:
        data:
          - text: "Sản phẩm này rất tốt!"
          - text: "Dịch vụ kém quá!"
        remove_html: true

  - name: sentiment_analysis
    sentiment.analyze:
      input:
        data: "$preprocess"
        text_column: "text"
```

### Bước 5: Chạy

```bash
zem list-tools -c pipeline.yaml
zem run pipeline.yaml
zem preview <artifact_id> --limit 10
```

---

## 9. Các Vấn đề Thường gặp

| Vấn đề | Nguyên nhân | Giải pháp |
|---------|------------|-----------|
| Server không tìm thấy | Path sai | Đảm bảo path relative với file YAML, kiểm tra thứ tự resolution |
| JSON-RPC lỗi | Print ra stdout | Không dùng `print()` trong server; dùng `logger` (stderr) |
| Data quá lớn | Return list trực tiếp | Dùng `server.save_output()` thay vì `return items` |
| Import error trong server | PYTHONPATH không có `src/` | `export PYTHONPATH=$PYTHONPATH:$(pwd)/src` hoặc dùng `uv run` |
| ZenML cache không hoạt động | Cấu hình sai | Set `cache: true` trong YAML hoặc `enable_cache=True` |
| Port bị chiếm | Conflict | ZenML Dashboard dùng port **8871-8879** |

---

## 10. Tham chiếu Nhanh

```python
# Import chính
from xfmr_zem import PipelineClient
from xfmr_zem.server import ZemServer

# Tạo và chạy pipeline
client = PipelineClient("pipeline.yaml")
run = client.run()

# Tạo custom server
server = ZemServer("my_server")

@server.tool()
def my_tool(data: Any, param: str = "default") -> Any:
    items = server.get_data(data)      # Chuẩn hóa input
    # ... xử lý ...
    return server.save_output(items)   # Trả về file reference

if __name__ == "__main__":
    server.run()                       # Chạy như MCP subprocess
```

**CLI tóm tắt**:
```bash
zem init <project>          # Bootstrap project mới
zem run <yaml>              # Chạy pipeline
zem run <yaml> -p <params>  # Chạy với custom params
zem run <yaml> -v           # Verbose mode
zem list-tools -c <yaml>    # Xem tools
zem preview <id>            # Preview artifact
zem dashboard               # Mở ZenML UI
zem explore                 # Visual configurator
```
