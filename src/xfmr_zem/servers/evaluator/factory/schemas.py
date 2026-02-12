from typing import List, Dict, Any, TypedDict, Optional, Union

# --- Schemas ---

class MultipleChoiceItem(TypedDict):
    input: str
    choices: List[str]
    output: str  # The correct answer (lists of the chosen labels)
    system_prompt: Optional[str]

class TextGenerationItem(TypedDict):
    input: str
    context: Optional[str]
    output: str
    system_prompt: Optional[str]

# Union type for validation
DatasetItem = Union[MultipleChoiceItem, TextGenerationItem]

# System prompt for each type of question
MCQ_SYSTEM_PROMPT = """You are a helpful assistant. 
Please answer the multiple-choice question by selecting the correct option.
There will be many options to choose, labeled from 0 to num_choices - 1"""

TEXTGEN_SYSTEM_PROMPT = "You are a helpful assistant. Please answer the question based on the provided context."
