"""
RecordFactory – chuẩn bị và validate records cho Argilla.

Hỗ trợ field_map linh hoạt: ánh xạ key trong data pipeline → field trong Argilla.
"""
from typing import Any, Dict, List, Optional
from loguru import logger


class RecordFactory:
    """
    Factory chuyển đổi dữ liệu thô thành Argilla Records.
    """

    @staticmethod
    def from_list(
        data: List[Dict[str, Any]],
        field_map: Optional[Dict[str, str]] = None,
        metadata_fields: Optional[List[str]] = None,
        suggestion_map: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        Chuyển list dicts thành List[rg.Record].

        Args:
            data: Danh sách records thô (từ pipeline)
            field_map: Ánh xạ key gốc → tên field Argilla
                       Ví dụ: {"input": "text", "response": "completion"}
                       None = dùng key gốc
            metadata_fields: Các key được đưa vào metadata (không phải fields)
            suggestion_map: Dict gợi ý nhãn sẵn cho questions
                            Ví dụ: {"label": {"value": "positive", "score": 0.9}}

        Returns:
            List[rg.Record]
        """
        try:
            import argilla as rg
        except ImportError:
            raise ImportError("Cài đặt: pip install 'xfmr-zem[argilla]'")

        if field_map is None:
            field_map = {}
        if metadata_fields is None:
            metadata_fields = []
        if suggestion_map is None:
            suggestion_map = {}

        records = []
        skipped = 0

        for i, item in enumerate(data):
            try:
                # Build fields
                fields = {}
                for key, value in item.items():
                    if key in metadata_fields:
                        continue
                    argilla_key = field_map.get(key, key)
                    fields[argilla_key] = str(value) if value is not None else ""

                # Build metadata
                metadata = {k: item[k] for k in metadata_fields if k in item}

                # Build suggestions (gợi ý nhãn từ model)
                suggestions = []
                for q_name, sug_cfg in suggestion_map.items():
                    if isinstance(sug_cfg, dict):
                        suggestions.append(
                            rg.Suggestion(
                                question_name=q_name,
                                value=sug_cfg.get("value"),
                                score=sug_cfg.get("score"),
                                agent=sug_cfg.get("agent", "auto"),
                            )
                        )

                record = rg.Record(
                    fields=fields,
                    metadata=metadata,
                    suggestions=suggestions if suggestions else None,
                )
                records.append(record)

            except Exception as e:
                logger.warning(f"Bỏ qua record #{i}: {e}")
                skipped += 1

        logger.info(f"Đã tạo {len(records)} records (bỏ qua {skipped} lỗi).")
        return records
