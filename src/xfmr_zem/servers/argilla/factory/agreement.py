"""
AgreementFactory – tính độ đồng thuận giữa các annotators (Inter-Annotator Agreement).

Hỗ trợ:
  - Cohen's Kappa   : 2 annotators, categorical
  - Fleiss' Kappa   : N annotators, categorical
  - Krippendorff Alpha : N annotators, mọi thang đo
  - Overlap %       : % records được ít nhất K annotators label
  - Label Distribution : phân phối nhãn
"""
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


class AgreementFactory:
    """
    Factory tính các chỉ số đồng thuận Inter-Annotator Agreement (IAA).
    """

    # ── Diễn giải Kappa / Alpha ─────────────────────────────────────────────

    @staticmethod
    def _interpret_kappa(score: float) -> str:
        """Diễn giải điểm Kappa theo Landis & Koch (1977)."""
        if score < 0:
            return "Kém (< 0: đồng thuận tệ hơn ngẫu nhiên)"
        elif score < 0.20:
            return "Không đáng kể (0–0.20)"
        elif score < 0.40:
            return "Yếu (0.20–0.40)"
        elif score < 0.60:
            return "Trung bình (0.40–0.60)"
        elif score < 0.80:
            return "Khá tốt (0.60–0.80)"
        else:
            return "Xuất sắc (0.80–1.00)"

    # ── Cohen's Kappa ────────────────────────────────────────────────────────

    @staticmethod
    def cohen_kappa(
        labels_a: List[Any],
        labels_b: List[Any],
    ) -> Dict[str, Any]:
        """
        Tính Cohen's Kappa giữa 2 annotators.

        Args:
            labels_a: Danh sách nhãn của annotator A
            labels_b: Danh sách nhãn của annotator B

        Returns:
            dict với score, method, n_samples, interpretation
        """
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            raise ImportError("Cài: pip install 'xfmr-zem[argilla]'")

        if len(labels_a) != len(labels_b):
            raise ValueError(
                f"Số nhãn không khớp: annotator_a={len(labels_a)}, annotator_b={len(labels_b)}"
            )

        score = float(cohen_kappa_score(labels_a, labels_b))
        result = {
            "method": "cohen_kappa",
            "score": round(score, 4),
            "n_samples": len(labels_a),
            "interpretation": AgreementFactory._interpret_kappa(score),
        }
        logger.info(f"Cohen's Kappa = {score:.4f} ({result['interpretation']})")
        return result

    # ── Fleiss' Kappa ────────────────────────────────────────────────────────

    @staticmethod
    def fleiss_kappa(
        ratings_matrix: List[List[int]],
        labels: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Tính Fleiss' Kappa cho N annotators.

        Args:
            ratings_matrix: Ma trận ratings. 
                            Mỗi hàng = 1 record, mỗi cột = số annotators gán nhãn đó.
                            Hoặc có thể là list of lists [annotator][record] = label.
            labels: Danh sách nhãn (dùng khi ratings_matrix là dạng list nhãn)

        Returns:
            dict với score, method, n_samples, n_annotators, interpretation
        """
        import numpy as np

        n_records = len(ratings_matrix)
        matrix = np.array(ratings_matrix, dtype=float)

        # Kiểm tra có dạng count matrix không
        if matrix.ndim == 2:
            n_annotators = int(matrix.sum(axis=1).mean())
            n_categories = matrix.shape[1]
        else:
            raise ValueError("ratings_matrix phải là 2D: [n_records × n_categories]")

        # Fleiss' Kappa formula
        p_j = matrix.sum(axis=0) / (n_records * n_annotators)  # proportion mỗi category
        P_e = float(np.sum(p_j ** 2))

        P_i = (np.sum(matrix ** 2, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
        P_bar = float(np.mean(P_i))

        if P_e == 1.0:
            kappa = 0.0
        else:
            kappa = (P_bar - P_e) / (1 - P_e)

        result = {
            "method": "fleiss_kappa",
            "score": round(kappa, 4),
            "n_samples": n_records,
            "n_categories": n_categories,
            "n_annotators": n_annotators,
            "interpretation": AgreementFactory._interpret_kappa(kappa),
        }
        logger.info(f"Fleiss' Kappa = {kappa:.4f} ({result['interpretation']})")
        return result

    # ── Krippendorff Alpha ───────────────────────────────────────────────────

    @staticmethod
    def krippendorff_alpha(
        reliability_data: List[List[Optional[Any]]],
        data_type: str = "nominal",
    ) -> Dict[str, Any]:
        """
        Tính Krippendorff Alpha cho N annotators với mọi thang đo.

        Args:
            reliability_data: Ma trận [n_annotators × n_records].
                              Dùng None cho missing data.
            data_type: "nominal" | "ordinal" | "interval" | "ratio"

        Returns:
            dict với score, method, data_type, interpretation
        """
        try:
            import krippendorff
        except ImportError:
            raise ImportError("Cài: pip install krippendorff")

        import numpy as np

        # Chuyển None → np.nan
        data = np.array(
            [[np.nan if v is None else v for v in row] for row in reliability_data],
            dtype=float,
        )

        alpha = float(krippendorff.alpha(reliability_data=data, level_of_measurement=data_type))

        result = {
            "method": "krippendorff_alpha",
            "score": round(alpha, 4),
            "data_type": data_type,
            "n_annotators": data.shape[0],
            "n_samples": data.shape[1],
            "interpretation": AgreementFactory._interpret_kappa(alpha),
        }
        logger.info(f"Krippendorff Alpha = {alpha:.4f} (type={data_type})")
        return result

    # ── Annotation Overlap % ─────────────────────────────────────────────────

    @staticmethod
    def overlap_percentage(
        records: List[Any],
        min_annotators: int = 2,
    ) -> Dict[str, Any]:
        """
        Tính % records được ít nhất `min_annotators` người label.

        Args:
            records: List rg.Record từ Argilla
            min_annotators: Ngưỡng tối thiểu

        Returns:
            dict với overlap_pct, total_records, covered_records, min_annotators
        """
        total = len(records)
        covered = sum(
            1 for rec in records
            if rec.responses and len(rec.responses) >= min_annotators
        )
        pct = round(covered / total * 100, 2) if total > 0 else 0.0

        result = {
            "method": "overlap",
            "total_records": total,
            "covered_records": covered,
            "min_annotators": min_annotators,
            "overlap_pct": pct,
        }
        logger.info(f"Overlap = {pct}% ({covered}/{total} records với ≥{min_annotators} annotators)")
        return result

    # ── Label Distribution ───────────────────────────────────────────────────

    @staticmethod
    def label_distribution(
        records: List[Any],
        question_name: str,
    ) -> Dict[str, Any]:
        """
        Tính phân phối nhãn trên một question.

        Args:
            records: List rg.Record từ Argilla
            question_name: Tên question cần phân tích

        Returns:
            dict với distribution (label → count), total_annotations, per_annotator
        """
        from collections import Counter, defaultdict

        distribution: Counter = Counter()
        per_annotator: Dict[str, Counter] = defaultdict(Counter)
        total = 0

        for rec in records:
            if not rec.responses:
                continue
            for resp in rec.responses:
                if not resp.values or question_name not in resp.values:
                    continue
                val = resp.values[question_name].value
                # Handle list (multi-label)
                labels = val if isinstance(val, list) else [val]
                for label in labels:
                    distribution[str(label)] += 1
                    annotator = str(resp.user_id) if resp.user_id else "unknown"
                    per_annotator[annotator][str(label)] += 1
                    total += 1

        result = {
            "method": "label_distribution",
            "question_name": question_name,
            "total_annotations": total,
            "distribution": dict(distribution),
            "per_annotator": {k: dict(v) for k, v in per_annotator.items()},
        }
        logger.info(f"Label distribution cho '{question_name}': {dict(distribution)}")
        return result

    # ── Dispatcher ───────────────────────────────────────────────────────────

    @classmethod
    def compute(
        cls,
        method: str,
        records: Optional[List[Any]] = None,
        question_name: str = "label",
        data_type: str = "nominal",
        min_annotators: int = 2,
    ) -> Dict[str, Any]:
        """
        Dispatcher tổng quát. Tự động extract labels từ records và tính IAA.

        Args:
            method: "cohen_kappa" | "fleiss_kappa" | "krippendorff" | "overlap" | "distribution"
            records: List rg.Record từ Argilla
            question_name: Tên question để lấy nhãn
            data_type: Thang đo cho krippendorff
            min_annotators: Cho overlap

        Returns:
            dict kết quả IAA
        """
        if method == "overlap":
            return cls.overlap_percentage(records, min_annotators)

        if method == "distribution":
            return cls.label_distribution(records, question_name)

        # Thu thập annotations theo annotator
        from collections import defaultdict
        annotator_labels: Dict[str, Dict[int, Any]] = defaultdict(dict)

        for i, rec in enumerate(records):
            if not rec.responses:
                continue
            for resp in rec.responses:
                if not resp.values or question_name not in resp.values:
                    continue
                annotator = str(resp.user_id) if resp.user_id else f"anon_{i}"
                annotator_labels[annotator][i] = resp.values[question_name].value

        annotators = list(annotator_labels.keys())
        n_records = len(records)

        if method == "cohen_kappa":
            if len(annotators) < 2:
                raise ValueError(f"Cohen's Kappa cần ≥2 annotators, có {len(annotators)}")
            a1 = annotators[0]
            a2 = annotators[1]
            # Chỉ lấy records cả 2 cùng annotate
            common_ids = sorted(
                set(annotator_labels[a1].keys()) & set(annotator_labels[a2].keys())
            )
            if not common_ids:
                raise ValueError("Không có records nào được cả 2 annotators label")
            la = [annotator_labels[a1][i] for i in common_ids]
            lb = [annotator_labels[a2][i] for i in common_ids]
            return cls.cohen_kappa(la, lb)

        elif method == "fleiss_kappa":
            # Xây count matrix
            import numpy as np
            all_labels = sorted(set(
                v for ann_dict in annotator_labels.values()
                for v in ann_dict.values()
            ))
            label_idx = {l: i for i, l in enumerate(all_labels)}
            matrix = np.zeros((n_records, len(all_labels)), dtype=float)
            for ann_labels in annotator_labels.values():
                for rec_idx, label in ann_labels.items():
                    matrix[rec_idx, label_idx[label]] += 1
            return cls.fleiss_kappa(matrix.tolist())

        elif method == "krippendorff":
            import numpy as np
            all_record_ids = list(range(n_records))
            rel_data = []
            for ann in annotators:
                row = [annotator_labels[ann].get(i, None) for i in all_record_ids]
                # Convert str labels to int nếu cần (nominal OK với str)
                rel_data.append(row)
            return cls.krippendorff_alpha(rel_data, data_type=data_type)

        else:
            raise ValueError(
                f"Method không hỗ trợ: '{method}'. "
                "Dùng: cohen_kappa | fleiss_kappa | krippendorff | overlap | distribution"
            )
