from typing import List, Optional, Any
import json
from string import Formatter
from opik.evaluation.models import OpikBaseModel
from pydantic import BaseModel, Field
from xfmr_zem.servers.evaluator.factory.prompt.evaluate_prompt import DEFAULT_EVALUATE_PROMPT
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
        out_str = str(output).lower().strip()
        ref_str = str(reference).lower().strip()
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
    score: int = Field(description="The score between 0 and 100")
    reason: str = Field(description="The reason for the score")

class CustomMetric(BaseMetric):
    REQUIRED_PROMPT_KEYS = {
        "name",
        "criteria",
        "input",
        "context",
        "expected_output",
        "output",
    }

    def __init__(self, name: str, criteria: str, judge_prompt: str = None, model: Any = None):
        self.name = name
        self.criteria = criteria
        self.model = model
        self.judge_prompt = judge_prompt or DEFAULT_EVALUATE_PROMPT
        self._validate_prompt_placeholders()

    def score(self, input: str, output: str, context: Any = None, expected_output: str = None, **kwargs) -> score_result.ScoreResult:
        # Prepare context string
        context_str = ""
        if isinstance(context, list):
            context_str = "\n".join([str(c) for c in context])
        elif context:
            context_str = str(context)

        prompt_kwargs = {
            "name": self.name,
            "criteria": self.criteria,
            "input": input,
            "context": context_str,
            "expected_output": expected_output or "",
            "output": output,
        }
        missing_keys = self.REQUIRED_PROMPT_KEYS - prompt_kwargs.keys()
        if missing_keys:
            raise ValueError(f"Missing required custom metric prompt keys: {sorted(missing_keys)}")

        prompt = self.judge_prompt.format(**prompt_kwargs)

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

    def _validate_prompt_placeholders(self):
        formatter = Formatter()
        placeholders = {
            field_name
            for _, field_name, _, _ in formatter.parse(self.judge_prompt)
            if field_name
        }
        missing = self.REQUIRED_PROMPT_KEYS - placeholders
        if missing:
            raise ValueError(
                f"Judge prompt is missing placeholders for required keys: {sorted(missing)}"
            )

class MetricFactory:
    """
    Factory to retrieve a list of Opik metrics from structured input.
    """

    @staticmethod
    def get_metrics(metric_specs: List[Any], model: Optional[Any] = None) -> List[BaseMetric]:
        valid_metrics = {
            "exact_match": lambda: ExactMatchMetric(),
            "hallucination": lambda: HallucinationMetric(model=model),
            "answer_relevance": lambda: AnswerRelevanceMetric(model=model),
            "context_recall": lambda: ContextRecallMetric(model=model),
            "context_precision": lambda: ContextPrecisionMetric(model=model),
            "moderation": lambda: ModerationMetric(model=model),
            "g_eval": lambda: GEvalMetric(model=model),
            "levenshtein_ratio": lambda: LevenshteinRatioMetric(),
        }
        metrics = []
        valid_metric_names = "\n".join(sorted(valid_metrics.keys()))

        for spec in metric_specs:
            if isinstance(spec, str):
                spec = {"name": spec}

            if not isinstance(spec, dict):
                raise ValueError(
                    "Metric entry must be a dictionary with at least a 'name' key."
                )

            name = spec.get("name")
            if not name:
                raise ValueError(
                    "Metric dictionary missing required 'name' key."
                )

            criteria = spec.get("criteria")
            judge_prompt = spec.get("judge_prompt")

            if criteria:
                metrics.append(CustomMetric(name=name, criteria=criteria, judge_prompt=judge_prompt, model=model))
                continue

            factory = valid_metrics.get(name)
            if not factory:
                raise ValueError(
                    f"Metric '{name}' is invalid; add description or modify name to match valid metrics:\n{valid_metric_names}"
                )

            metrics.append(factory())

        return metrics

if (__name__ == "__main__"):
    list_metric = [
        {"name": "exact_match"},
        {"name": "accuracy",
         "criteria": "Scores how accurate the response is compared to the expected output."},
        {"name": "bruh"}
    ]

    metrics = MetricFactory.get_metrics(list_metric)
    for metric in metrics:
        print(metric.name)
