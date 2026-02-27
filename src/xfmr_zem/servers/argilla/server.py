"""
Argilla MCP Server cho Zem.

Tools dataset (8):
  1. create_dataset       – Tạo dataset với fields & questions
  2. push_records         – Đẩy records vào Argilla
  3. get_records          – Lấy records đã annotate
  4. query_records        – Query theo filter
  5. export_dataset       – Export ra file (jsonl/parquet)
  6. delete_records       – Xóa records theo filter
  7. annotation_progress  – Thống kê tiến độ annotation
  8. agreement_score      – Tính Inter-Annotator Agreement (IAA)

Tools user/annotator management (5):
  9.  create_user         – Tạo user mới (admin | annotator)
  10. list_users          – Liệt kê users (có thể lọc theo role)
  11. delete_user         – Xóa user theo username
  12. manage_workspace    – Thêm/xóa user khỏi workspace
  13. annotator_stats     – Thống kê annotations theo từng annotator
"""
import os
import sys
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from dotenv import load_dotenv

from xfmr_zem.server import ZemServer
from xfmr_zem.servers.argilla.factory.connection import ArgillaConnectionFactory
from xfmr_zem.servers.argilla.factory.dataset import DatasetFactory
from xfmr_zem.servers.argilla.factory.record import RecordFactory
from xfmr_zem.servers.argilla.factory.export import ExportFactory
from xfmr_zem.servers.argilla.factory.agreement import AgreementFactory
from xfmr_zem.servers.argilla.factory.user import UserFactory

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO")

# ── Server ───────────────────────────────────────────────────────────────────
server = ZemServer("argilla")


def _get_client(api_url: Optional[str] = None, api_key: Optional[str] = None):
    """Helper lấy client với fallback từ parameters.yml và env."""
    _ = load_dotenv(override=True)
    resolved_url = (
        api_url
        or os.getenv("ARGILLA_API_URL")
        or server.parameters.get("argilla_url", "http://localhost:6900")
    )
    resolved_key = (
        api_key
        or os.getenv("ARGILLA_API_KEY")
        or server.parameters.get("argilla_api_key", "argilla.apikey")
    )
    return ArgillaConnectionFactory.get_client(resolved_url, resolved_key)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: create_dataset
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def create_dataset(
    data: Any = None,
    name: str = "zem_dataset",
    workspace: str = "admin",
    fields: Optional[List[Dict[str, Any]]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    guidelines: str = "",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tạo hoặc lấy dataset trên Argilla server.
    Fix lỗi logic indexing trong built-in server 0.3.7.
    """
    import argilla as rg
    client = _get_client(api_url, api_key)

    # 1. Kiểm tra tồn tại với logic đúng cho Argilla 2.x
    try:
        existing = client.datasets(name=name, workspace=workspace)
        if existing:
            logger.info(f"Dataset '{name}' đã tồn tại.")
            return {"status": "exists", "dataset_name": name, "workspace": workspace}
    except Exception as e:
        logger.debug(f"Lỗi khi check dataset tồn tại: {e}")

    # 2. Xây dựng config nếu chưa có
    if fields is None:
        fields = [{"name": "text", "type": "text"}]
    if questions is None:
        questions = [
            {"name": "label", "type": "label", "labels": ["positive", "negative", "neutral"]}
        ]

    # 3. Tạo mới (sử dụng logic fixed tương tự DatasetFactory)
    built_fields = []
    for f in fields:
        ftype = f.get("type", "text").lower()
        fname = f["name"]
        ftitle = f.get("title", fname)
        built_fields.append(rg.TextField(name=fname, title=ftitle))

    built_questions = []
    for q in questions:
        qtype = q.get("type", "label").lower()
        qname = q["name"]
        qtitle = q.get("title", qname)
        if qtype == "label":
            built_questions.append(rg.LabelQuestion(
                name=qname, title=qtitle, labels=q.get("labels", []),
                required=q.get("required", True)
            ))
        # Có thể thêm các loại question khác nếu cần

    settings = rg.Settings(
        fields=built_fields,
        questions=built_questions,
        guidelines=guidelines,
    )

    dataset = rg.Dataset(
        name=name,
        workspace=workspace,
        settings=settings,
        client=client,
    )
    dataset.create()
    logger.info(f"Đã tạo dataset '{name}'")
    
    return {
        "status": "created",
        "dataset_name": name,
        "workspace": workspace
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: push_records
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def push_records(
    data: Any,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    field_map: Optional[Dict[str, str]] = None,
    metadata_fields: Optional[List[str]] = None,
    suggestion_map: Optional[Dict[str, Any]] = None,
    batch_size: int = 200,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Đẩy records từ pipeline vào Argilla dataset.

    Args:
        data: Dữ liệu đầu vào (list dicts, path file, hoặc file reference)
        dataset_name: Tên dataset đích
        workspace: Workspace chứa dataset
        field_map: Ánh xạ key pipeline → field Argilla
                   Ví dụ: {"input": "text", "output": "response"}
        metadata_fields: Keys đưa vào metadata thay vì fields
        suggestion_map: Gợi ý nhãn tự động
                        {"label": {"value": "positive", "score": 0.9, "agent": "model-v1"}}
        batch_size: Số records mỗi batch khi upload
        api_url: URL Argilla server
        api_key: API key

    Returns:
        {"status": "ok", "pushed": int, "dataset_name": str}
    """
    client = _get_client(api_url, api_key)

    # Lấy data qua ZemServer.get_data (hỗ trợ file reference, JSONL, Parquet...)
    rows = server.get_data(data)
    logger.info(f"Chuẩn bị push {len(rows)} records → dataset '{dataset_name}'")

    records = RecordFactory.from_list(
        data=rows,
        field_map=field_map,
        metadata_fields=metadata_fields,
        suggestion_map=suggestion_map,
    )

    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        raise ValueError(
            f"Dataset '{dataset_name}' không tồn tại. "
            "Hãy chạy create_dataset trước."
        )
    dataset = dataset[0]

    # Push theo batch
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        dataset.records.log(batch)
        logger.info(f"Đã push batch {i // batch_size + 1}: {len(batch)} records")

    return {
        "status": "ok",
        "pushed": len(records),
        "dataset_name": dataset_name,
        "workspace": workspace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: get_records
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def get_records(
    data: Any = None,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    status: Optional[str] = None,
    limit: int = 100,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lấy records đã annotate từ Argilla.

    Args:
        data: Không dùng (giữ cho tương thích pipeline)
        dataset_name: Tên dataset
        workspace: Workspace
        status: Lọc theo trạng thái response: "submitted" | "discarded" | "pending" | None (lấy tất cả)
        limit: Số records tối đa
        api_url: URL Argilla server
        api_key: API key

    Returns:
        File reference (parquet) chứa records
    """
    client = _get_client(api_url, api_key)

    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại.")
    dataset = dataset[0]

    # Lấy records
    query = {"response_status": [status]} if status else {}
    fetched = list(dataset.records(query=query or None, limit=limit, with_responses=True))
    logger.info(f"Lấy được {len(fetched)} records từ '{dataset_name}'")

    # Serialize
    rows = []
    for rec in fetched:
        row = {
            "id": str(rec.id) if rec.id else None,
            **(rec.fields or {}),
            **(rec.metadata or {}),
        }
        if rec.responses:
            for resp in rec.responses:
                row[f"status"] = resp.status.value if resp.status else None
                for q, v in (resp.values or {}).items():
                    row[f"answer_{q}"] = v.value
        rows.append(row)

    return server.save_output(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: query_records
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def query_records(
    data: Any = None,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    status: Optional[str] = None,
    annotator_id: Optional[str] = None,
    label_filter: Optional[str] = None,
    question_name: str = "label",
    limit: int = 100,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query records theo nhiều điều kiện lọc.

    Args:
        data: Không dùng (tương thích pipeline)
        dataset_name: Tên dataset
        workspace: Workspace
        status: Lọc status responses: "submitted" | "discarded" | "pending"
        annotator_id: Lọc theo annotator ID cụ thể
        label_filter: Lọc records có nhãn này (khớp giá trị question)
        question_name: Tên question để áp label_filter
        limit: Số records tối đa
        api_url: URL Argilla server
        api_key: API key

    Returns:
        File reference (parquet) chứa records đã lọc
    """
    client = _get_client(api_url, api_key)

    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại.")
    dataset = dataset[0]

    query = {}
    if status:
        query["response_status"] = [status]

    fetched = list(dataset.records(query=query or None, with_responses=True))

    # Filter phía client (annotator & label)
    filtered = []
    for rec in fetched:
        if not rec.responses:
            if not annotator_id and not label_filter:
                filtered.append(rec)
            continue
        for resp in rec.responses:
            if annotator_id and str(resp.user_id) != annotator_id:
                continue
            if label_filter and resp.values:
                val = resp.values.get(question_name)
                if not val or str(val.value) != str(label_filter):
                    continue
            filtered.append(rec)
            break

    filtered = filtered[:limit]
    logger.info(f"Query: {len(filtered)} records (filter: status={status}, annotator={annotator_id}, label={label_filter})")

    rows = []
    for rec in filtered:
        row = {
            "id": str(rec.id) if rec.id else None,
            **(rec.fields or {}),
            **(rec.metadata or {}),
        }
        if rec.responses:
            for resp in rec.responses:
                row["status"] = resp.status.value if resp.status else None
                row["annotator"] = str(resp.user_id) if resp.user_id else None
                for q, v in (resp.values or {}).items():
                    row[f"answer_{q}"] = v.value
        rows.append(row)

    return server.save_output(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: export_dataset
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def export_dataset(
    data: Any = None,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    output_path: str = "/tmp/zem_argilla_export",
    format: str = "jsonl",
    status: Optional[str] = "submitted",
    limit: int = 10000,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export annotated dataset ra file.

    Args:
        data: Không dùng (tương thích pipeline)
        dataset_name: Tên dataset
        workspace: Workspace
        output_path: Đường dẫn thư mục output
        format: "jsonl" | "parquet"
        status: Lọc status: "submitted" | "discarded" | None (xuất tất cả)
        limit: Số records tối đa
        api_url: URL Argilla server
        api_key: API key

    Returns:
        {"path": str, "format": str, "n_records": int}
    """
    import os
    client = _get_client(api_url, api_key)

    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại.")
    dataset = dataset[0]

    query = {"response_status": [status]} if status else {}
    records = list(dataset.records(query=query or None, limit=limit, with_responses=True))
    logger.info(f"Xuất {len(records)} records từ '{dataset_name}'")

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"{dataset_name}.{format}")

    if format == "jsonl":
        ExportFactory.to_jsonl(records, out_file)
    elif format == "parquet":
        ExportFactory.to_parquet(records, out_file)
    else:
        raise ValueError(f"Format không hỗ trợ: '{format}'. Dùng: jsonl | parquet")

    return {
        "path": out_file,
        "format": format,
        "n_records": len(records),
        "dataset_name": dataset_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6: delete_records
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def delete_records(
    data: Any = None,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    status: Optional[str] = "discarded",
    limit: int = 1000,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Xóa records theo filter.

    Args:
        data: Không dùng
        dataset_name: Tên dataset
        workspace: Workspace
        status: Lọc records cần xóa theo status: "submitted" | "discarded" | "pending" | None (xóa tất cả)
        limit: Số records tối đa cần xóa (để tránh xóa quá nhiều)
        api_url: URL Argilla server
        api_key: API key

    Returns:
        {"deleted": int, "dataset_name": str}
    """
    client = _get_client(api_url, api_key)

    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại.")
    dataset = dataset[0]

    query = {"response_status": [status]} if status else {}
    records = list(dataset.records(query=query or None, limit=limit))
    logger.info(f"Xóa {len(records)} records (filter status='{status}') từ '{dataset_name}'")

    dataset.records.delete(records)

    return {
        "deleted": len(records),
        "dataset_name": dataset_name,
        "workspace": workspace,
        "filter_status": status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 7: annotation_progress
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def annotation_progress(
    data: Any = None,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Thống kê tiến độ annotation của dataset.

    Args:
        data: Không dùng
        dataset_name: Tên dataset
        workspace: Workspace
        api_url: URL Argilla server
        api_key: API key

    Returns:
        {
            "total_records": int,
            "submitted": int, "discarded": int, "pending": int,
            "completion_pct": float,
            "per_annotator": {user_id: {"submitted": int, "discarded": int}},
        }
    """
    from collections import defaultdict

    client = _get_client(api_url, api_key)

    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại.")
    dataset = dataset[0]

    all_records = list(dataset.records(with_responses=True))
    total = len(all_records)

    submitted = discarded = pending = 0
    per_annotator: Dict[str, Dict[str, int]] = defaultdict(lambda: {"submitted": 0, "discarded": 0, "pending": 0})

    for rec in all_records:
        has_response = False
        if rec.responses:
            for resp in rec.responses:
                has_response = True
                uid = str(resp.user_id) if resp.user_id else "unknown"
                st = resp.status.value if resp.status else "pending"
                per_annotator[uid][st] = per_annotator[uid].get(st, 0) + 1
                if st == "submitted":
                    submitted += 1
                elif st == "discarded":
                    discarded += 1
                else:
                    pending += 1
        if not has_response:
            pending += 1

    completion_pct = round((submitted + discarded) / max(total, 1) * 100, 2)

    result = {
        "dataset_name": dataset_name,
        "workspace": workspace,
        "total_records": total,
        "submitted": submitted,
        "discarded": discarded,
        "pending": pending,
        "completion_pct": completion_pct,
        "per_annotator": {k: dict(v) for k, v in per_annotator.items()},
    }

    logger.info(
        f"Tiến độ '{dataset_name}': {submitted}/{total} submitted "
        f"({completion_pct}% hoàn thành)"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 8: agreement_score
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def agreement_score(
    data: Any = None,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    question_name: str = "label",
    method: str = "cohen_kappa",
    data_type: str = "nominal",
    min_annotators: int = 2,
    limit: int = 5000,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tính độ đồng thuận giữa các annotators (Inter-Annotator Agreement).

    Args:
        data: Không dùng
        dataset_name: Tên dataset
        workspace: Workspace
        question_name: Tên question cần tính IAA
        method: Thuật toán tính IAA:
                "cohen_kappa"    – 2 annotators, categorical (Cohen's κ)
                "fleiss_kappa"   – N annotators, categorical (Fleiss' κ)
                "krippendorff"   – N annotators, mọi thang đo (Krippendorff α)
                "overlap"        – % records được ≥ min_annotators label
                "distribution"   – Phân phối nhãn theo annotator
        data_type: Thang đo cho krippendorff: "nominal"|"ordinal"|"interval"|"ratio"
        min_annotators: Ngưỡng tối thiểu annotators/record (cho overlap)
        limit: Số records tối đa
        api_url: URL Argilla server
        api_key: API key

    Returns:
        Dict chứa IAA score và diễn giải
    """
    client = _get_client(api_url, api_key)

    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' không tồn tại.")
    dataset = dataset[0]

    # Chỉ lấy records đã submitted
    records = list(
        dataset.records(
            query={"response_status": ["submitted"]},
            limit=limit,
            with_responses=True,
        )
    )
    logger.info(f"Tính IAA ({method}) cho '{question_name}' trên {len(records)} records")

    if not records:
        return {
            "error": "Không có records submitted.",
            "method": method,
            "question_name": question_name,
        }

    result = AgreementFactory.compute(
        method=method,
        records=records,
        question_name=question_name,
        data_type=data_type,
        min_annotators=min_annotators,
    )

    result["dataset_name"] = dataset_name
    result["question_name"] = question_name
    return result


# =============================================================================
# Tools 9-13: User / Annotator Management
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Tool 9: create_user
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def create_user(
    data: Any = None,
    username: str = "",
    password: str = "",
    role: str = "annotator",
    first_name: str = "",
    last_name: str = "",
    workspace: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tạo user mới trên Argilla server.

    Args:
        data: Không dùng, cho phép Zem auto-chain từ step trước
        username: Tên đăng nhập (unique)
        password: Mật khẩu (min 8 ký tự)
        role: "annotator" (mặc định) | "admin"
        first_name: Tên user
        last_name: Họ user
        workspace: Tự động thêm vào workspace này sau khi tạo (optional)
        api_url: URL Argilla server
        api_key: API key

    Returns:
        {"status": "created", "username": str, "role": str, "workspace": str|None}
    """
    client = _get_client(api_url, api_key)

    UserFactory.create_user(
        client=client,
        username=username,
        password=password,
        role=role,
        first_name=first_name,
        last_name=last_name,
    )

    ws_result = None
    if workspace:
        ws_result = UserFactory.add_to_workspace(client, username, workspace)

    return {
        "status": "created",
        "username": username,
        "role": role,
        "workspace": ws_result["workspace"] if ws_result else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 10: list_users
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def list_users(
    data: Any = None,
    role: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Liệt kê tất cả users trên Argilla server.

    Args:
        data: Không dùng
        role: Lọc theo role: "admin" | "annotator" | None (tất cả)
        api_url: URL Argilla server
        api_key: API key

    Returns:
        File reference chứa list users (id, username, role, first_name, last_name)
    """
    client = _get_client(api_url, api_key)
    users = UserFactory.list_users(client, role_filter=role)
    return server.save_output(users)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 11: delete_user
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def delete_user(
    username: str,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Xóa user theo username.

    Args:
        username: Tên đăng nhập cần xóa
        api_url: URL Argilla server
        api_key: API key

    Returns:
        {"status": "deleted", "username": str}
    """
    client = _get_client(api_url, api_key)
    UserFactory.delete_user(client, username)
    return {"status": "deleted", "username": username}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 12: manage_workspace
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def manage_workspace(
    username: str,
    workspace_name: str,
    action: str = "add",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quản lý thành viên workspace: thêm hoặc xóa annotator.

    Args:
        username: Tên user cần quản lý
        workspace_name: Tên workspace
        action: "add" (thêm vào workspace) | "remove" (xóa khỏi workspace)
        api_url: URL Argilla server
        api_key: API key

    Returns:
        {"username": str, "workspace": str, "status": "added"|"removed"}
    """
    client = _get_client(api_url, api_key)

    if action == "add":
        return UserFactory.add_to_workspace(client, username, workspace_name)
    elif action == "remove":
        return UserFactory.remove_from_workspace(client, username, workspace_name)
    else:
        raise ValueError(f"Action không hợp lệ: '{action}'. Dùng: add | remove")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 13: annotator_stats
# ─────────────────────────────────────────────────────────────────────────────
@server.tool()
def annotator_stats(
    data: Any = None,
    dataset_name: str = "zem_dataset",
    workspace: str = "admin",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Thống kê số lượng annotations theo từng annotator trong dataset.

    Args:
        data: Không dùng
        dataset_name: Tên dataset
        workspace: Workspace
        api_url: URL Argilla server
        api_key: API key

    Returns:
        File reference chứa danh sách annotators và thống kê:
        [{"annotator_id": str, "submitted": int, "discarded": int, "pending": int, "total": int}]
        (sắp xếp giảm dần theo tổng annotations)
    """
    client = _get_client(api_url, api_key)
    stats = UserFactory.annotator_stats(
        client=client,
        dataset_name=dataset_name,
        workspace=workspace,
    )
    return server.save_output(stats)


# =============================================================================
# Entry Point – phải đặt SAU khi tất cả tool đã đăng ký
# =============================================================================
if __name__ == "__main__":
    server.run()
