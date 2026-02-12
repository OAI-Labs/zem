from typing import List, Optional, Any
from opik.evaluation.models import OpikBaseModel
from opik.evaluation.metrics import (
    Hallucination,
    AnswerRelevance,
    ContextRecall,
    ContextPrecision,
    Moderation,
    LevenshteinRatio,
    GEval,
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

class GEvalMetric(BaseMetric):
    def __init__(self, model: Optional[OpikBaseModel] = None, name: str = "g_eval"):
        self.name = name
        self.metric = GEval(model=model, name=name)

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
    Factory to retrieve a list of Opik metrics based on the list of metric names.
    """
    
    @staticmethod
    def get_metrics(metric_names: List[str], model: Optional[Any] = None) -> List[BaseMetric]:
        metrics = []
        for name in metric_names:
            if name == "exact_match":
                metrics.append(ExactMatchMetric())
            elif name == "hallucination":
                metrics.append(HallucinationMetric(model=model))
            elif name == "answer_relevance":
                metrics.append(AnswerRelevanceMetric(model=model))
            elif name == "context_recall":
                metrics.append(ContextRecallMetric(model=model))
            elif name == "context_precision":
                metrics.append(ContextPrecisionMetric(model=model))
            elif name == "moderation":
                metrics.append(ModerationMetric(model=model))
            elif name == "g_eval":
                metrics.append(GEvalMetric(model=model))
            elif name == "levenshtein_ratio":
                metrics.append(LevenshteinRatioMetric())
        return metrics
