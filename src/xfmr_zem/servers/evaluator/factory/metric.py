from typing import List, Optional, Any
import json
import re
from opik.evaluation.models import OpikBaseModel
from pydantic import BaseModel, Field
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

from loguru import logger
import sys
logger.remove()

logger.add(sys.stderr, level = "DEBUG")

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

class CustomMetricResponse(BaseModel):
    score: float = Field(description="The score between 0.0 and 1.0")
    reason: str = Field(description="The reason for the score")

class CustomMetric(BaseMetric):
    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model

    def score(self, input: str, output: str, context: Any = None, expected_output: str = None, **kwargs) -> score_result.ScoreResult:
        # Prepare context string
        context_str = ""
        if isinstance(context, list):
            context_str = "\n".join([str(c) for c in context])
        elif context:
            context_str = str(context)

        prompt = f"""[SYSTEM INSTRUCTION]
You are an expert AI evaluator. Your task is to assess the quality of the 'Actual Output' based on the provided 'Input', 'Context', and 'Expected Output'.

Metric Name: {self.name}
Scoring Scale: 0.0 (Worst) to 1.0 (Best)

[DATA TO EVALUATE]
Input:
{input}

Context:
{context_str}

Expected Output (Reference):
{expected_output}

Actual Output (Target):
{output}

[OUTPUT REQUIREMENT]
1. Evaluate the 'Actual Output' objectively.
2. Return the result strictly in valid JSON format.
3. Do not include any markdown formatting (like ```json), explanations, or additional text outside the JSON object.

Use exactly the following JSON structure:
{{
  "score": <float>,   // A value between 0.0 and 1.0
  "reason": "<string>" // A concise explanation for the score
}}
[/SYSTEM INSTRUCTION]
"""
        try:
            result = self.model.generate_string(prompt, response_format=CustomMetricResponse)
            # Extract JSON using custom logic
            logger.debug(f"Custom Metric - generated result: {result}")
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                first_brace = result.find("{")
                last_brace = result.rfind("}")
                json_string = result[first_brace : last_brace + 1]
                data = json.loads(json_string)

            score = float(data.get("score", 0.0))
            reason = data.get("reason", "No reason provided.")
        except Exception as e:
            score = 0.0
            reason = f"Error evaluating custom metric: {e}"

        return score_result.ScoreResult(
            name=self.name,
            value=score,
            reason=reason
        )

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
            else:
                metrics.append(CustomMetric(name=name, model=model))
        return metrics
