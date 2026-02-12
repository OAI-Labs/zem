from typing import List, Optional
from opik.evaluation.models import OpikBaseModel
from opik.evaluation.metrics import (
    Hallucination,
    AnswerRelevance,
    ContextRecall,
    ContextPrecision,
    Moderation,
    LevenshteinRatio,
    BaseMetric,
    score_result
)

class ExactMatchMetric(BaseMetric):
    def __init__(self, name: str = "exact_match"):
        self.name = name

    def score(self, output: str, reference: str, **kwargs) -> score_result.ScoreResult:
        # Simple exact match logic
        out_str = str(output).strip()
        ref_str = str(reference).strip()
        is_match = out_str == ref_str
        
        return score_result.ScoreResult(
            name=self.name,
            value=1.0 if is_match else 0.0,
            reason="Output matches reference exactly" if is_match else f"Expected '{ref_str}', got '{out_str}'"
        )

class HallucinationMetric(BaseMetric):
    def __init__(self, model: Optional[OpikBaseModel] = None, name: str = "hallucination"):
        self.name = name
        self.metric = Hallucination(model=model)

    def score(self, **kwargs) -> score_result.ScoreResult:
        return self.metric.score(**kwargs)

class AnswerRelevanceMetric(BaseMetric):
    def __init__(self, model: Optional[OpikBaseModel] = None, name: str = "answer_relevance"):
        self.name = name
        self.metric = AnswerRelevance(model=model)

    def score(self, **kwargs) -> score_result.ScoreResult:
        return self.metric.score(**kwargs)

class ContextRecallMetric(BaseMetric):
    def __init__(self, model: Optional[OpikBaseModel] = None, name: str = "context_recall"):
        self.name = name
        self.metric = ContextRecall(model=model)

    def score(self, **kwargs) -> score_result.ScoreResult:
        return self.metric.score(**kwargs)

class ContextPrecisionMetric(BaseMetric):
    def __init__(self, model: Optional[OpikBaseModel] = None, name: str = "context_precision"):
        self.name = name
        self.metric = ContextPrecision(model=model)

    def score(self, **kwargs) -> score_result.ScoreResult:
        return self.metric.score(**kwargs)

class ModerationMetric(BaseMetric):
    def __init__(self, model: Optional[OpikBaseModel] = None, name: str = "moderation"):
        self.name = name
        self.metric = Moderation(model=model)

    def score(self, **kwargs) -> score_result.ScoreResult:
        return self.metric.score(**kwargs)

class LevenshteinRatioMetric(BaseMetric):
    def __init__(self, name: str = "levenshtein_ratio"):
        self.name = name
        self.metric = LevenshteinRatio()

    def score(self, **kwargs) -> score_result.ScoreResult:
        return self.metric.score(**kwargs)

class MetricFactory:
    """
    Factory to retrieve a list of Opik metrics based on the dataset/task type.
    """
    
    @staticmethod
    def get_metrics(dataset_type: str, model: Optional[OpikBaseModel] = None) -> List[BaseMetric]:
        if dataset_type == "text_generation":
            # Metrics suitable for RAG or Context-based generation
            return [
                HallucinationMetric(model=model),
                AnswerRelevanceMetric(model=model),
                ContextRecallMetric(model=model),
                ContextPrecisionMetric(model=model),
                ModerationMetric(model=model),
            ]
        elif dataset_type == "multiple_choice":
            return [
                ExactMatchMetric(),
                LevenshteinRatioMetric(),
                ContextPrecisionMetric(model=model),
                ModerationMetric(model=model),
            ]
        else:
            return []
