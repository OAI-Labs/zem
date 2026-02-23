"""
DatasetFactory – tạo hoặc lấy Argilla Dataset với cấu hình fields & questions.

Hỗ trợ:
  - TextField, ChatField
  - LabelQuestion, MultiLabelQuestion, RatingQuestion, TextQuestion, SpanQuestion
"""
from typing import Any, Dict, List, Optional
from loguru import logger


# Mapping tên string → class Argilla field/question
FIELD_TYPES = {
    "text": "TextField",
    "chat": "ChatField",
}

QUESTION_TYPES = {
    "label": "LabelQuestion",
    "multi_label": "MultiLabelQuestion",
    "rating": "RatingQuestion",
    "text": "TextQuestion",
    "span": "SpanQuestion",
}


class DatasetFactory:
    """
    Factory tạo và quản lý Argilla Dataset.
    """

    @staticmethod
    def _build_field(rg, field_cfg: Dict[str, Any]):
        """Chuyển dict config thành Argilla Field object."""
        ftype = field_cfg.get("type", "text").lower()
        name = field_cfg["name"]
        title = field_cfg.get("title", name)
        required = field_cfg.get("required", True)

        if ftype == "text":
            return rg.TextField(name=name, title=title, required=required)
        elif ftype == "chat":
            return rg.ChatField(name=name, title=title, required=required)
        else:
            raise ValueError(f"Loại field không hỗ trợ: '{ftype}'. Dùng: {list(FIELD_TYPES.keys())}")

    @staticmethod
    def _build_question(rg, q_cfg: Dict[str, Any]):
        """Chuyển dict config thành Argilla Question object."""
        qtype = q_cfg.get("type", "label").lower()
        name = q_cfg["name"]
        title = q_cfg.get("title", name)
        description = q_cfg.get("description", "")
        required = q_cfg.get("required", True)

        if qtype == "label":
            labels = q_cfg.get("labels", [])
            return rg.LabelQuestion(
                name=name, title=title, description=description,
                required=required, labels=labels,
            )
        elif qtype == "multi_label":
            labels = q_cfg.get("labels", [])
            return rg.MultiLabelQuestion(
                name=name, title=title, description=description,
                required=required, labels=labels,
            )
        elif qtype == "rating":
            values = q_cfg.get("values", list(range(1, 6)))
            return rg.RatingQuestion(
                name=name, title=title, description=description,
                required=required, values=values,
            )
        elif qtype == "text":
            return rg.TextQuestion(
                name=name, title=title, description=description,
                required=required,
            )
        elif qtype == "span":
            field = q_cfg.get("field", "text")
            labels = q_cfg.get("labels", [])
            return rg.SpanQuestion(
                name=name, title=title, description=description,
                required=required, field=field, labels=labels,
            )
        else:
            raise ValueError(f"Loại question không hỗ trợ: '{qtype}'. Dùng: {list(QUESTION_TYPES.keys())}")

    @classmethod
    def create_or_get(
        cls,
        client,
        name: str,
        workspace: str = "admin",
        fields: Optional[List[Dict[str, Any]]] = None,
        questions: Optional[List[Dict[str, Any]]] = None,
        guidelines: str = "",
    ):
        """
        Tạo hoặc lấy dataset đã tồn tại.

        Args:
            client: Argilla client (từ ArgillaConnectionFactory)
            name: Tên dataset
            workspace: Workspace chứa dataset
            fields: List config cho fields
            questions: List config cho questions
            guidelines: Hướng dẫn annotate

        Returns:
            rg.Dataset instance
        """
        try:
            import argilla as rg
        except ImportError:
            raise ImportError("Cài đặt: pip install 'xfmr-zem[argilla]'")

        # Kiểm tra dataset đã tồn tại chưa
        try:
            existing = client.datasets(name=name, workspace=workspace)
            if existing:
                logger.info(f"Dataset '{name}' (workspace='{workspace}') đã tồn tại.")
                return existing[0]
        except Exception:
            pass

        # Xây dựng fields
        built_fields = []
        for f in (fields or []):
            built_fields.append(cls._build_field(rg, f))

        # Xây dựng questions
        built_questions = []
        for q in (questions or []):
            built_questions.append(cls._build_question(rg, q))

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
        logger.info(f"Đã tạo dataset '{name}' trong workspace '{workspace}'.")
        return dataset
