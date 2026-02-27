"""
ExportFactory – export annotated records từ Argilla về file.

Hỗ trợ:
  - JSONL
  - Parquet
  - HuggingFace Hub (optional)
"""
import os
import json
from typing import Any, List, Optional
from loguru import logger


class ExportFactory:
    """
    Factory export annotated records về các định dạng khác nhau.
    """

    @staticmethod
    def to_jsonl(records: List[Any], path: str) -> str:
        """
        Export records sang file JSONL.

        Args:
            records: List rg.Record đã fetch từ Argilla
            path: Đường dẫn file output

        Returns:
            Đường dẫn file đã ghi
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rows = []
        for rec in records:
            row = {
                "id": str(rec.id) if rec.id else None,
                "fields": rec.fields,
                "metadata": rec.metadata if rec.metadata else {},
                "responses": [],
            }
            if rec.responses:
                for resp in rec.responses:
                    row["responses"].append({
                        "user_id": str(resp.user_id) if resp.user_id else None,
                        "status": resp.status.value if resp.status else None,
                        "values": {
                            q: v.value
                            for q, v in (resp.values or {}).items()
                        },
                    })
            rows.append(row)

        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(f"Đã export {len(rows)} records → {path} (JSONL)")
        return path

    @staticmethod
    def to_parquet(records: List[Any], path: str) -> str:
        """
        Export records sang file Parquet.

        Args:
            records: List rg.Record đã fetch từ Argilla
            path: Đường dẫn file output

        Returns:
            Đường dẫn file đã ghi
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Cần pandas: pip install pandas")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rows = []
        for rec in records:
            base = {
                "id": str(rec.id) if rec.id else None,
                **{f"field_{k}": v for k, v in (rec.fields or {}).items()},
                **{f"meta_{k}": v for k, v in (rec.metadata or {}).items()},
            }
            if rec.responses:
                for resp in rec.responses:
                    row = dict(base)
                    row["annotator"] = str(resp.user_id) if resp.user_id else None
                    row["status"] = resp.status.value if resp.status else None
                    for q, v in (resp.values or {}).items():
                        row[f"answer_{q}"] = v.value
                    rows.append(row)
            else:
                rows.append(base)

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        logger.info(f"Đã export {len(rows)} records → {path} (Parquet)")
        return path

    @staticmethod
    def to_huggingface(
        records: List[Any],
        repo_id: str,
        token: Optional[str] = None,
        split: str = "train",
    ) -> str:
        """
        Push records lên HuggingFace Hub dạng dataset.

        Args:
            records: List rg.Record
            repo_id: HuggingFace repo (vd: "org/dataset-name")
            token: HF token (hoặc từ env HF_TOKEN)
            split: Dataset split name

        Returns:
            URL của dataset trên Hub
        """
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError("Cần huggingface datasets: pip install datasets")

        token = token or os.getenv("HF_TOKEN")
        rows = []
        for rec in records:
            row = {
                "id": str(rec.id) if rec.id else None,
                **(rec.fields or {}),
                **(rec.metadata or {}),
            }
            if rec.responses:
                for resp in rec.responses:
                    for q, v in (resp.values or {}).items():
                        row[q] = v.value
            rows.append(row)

        hf_dataset = HFDataset.from_list(rows)
        url = hf_dataset.push_to_hub(repo_id, split=split, token=token)
        logger.info(f"Đã push {len(rows)} records → HuggingFace Hub: {repo_id}")
        return f"https://huggingface.co/datasets/{repo_id}"
