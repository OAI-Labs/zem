DEFAULT_EVALUATE_PROMPT = """[SYSTEM INSTRUCTION]
You are an expert AI evaluator. 
Your task is to assess the quality of the 'Actual Output' based on the provided 'Input', 'Context', and 'Expected Output'.

Metric Name: {name}
Metric Criteria: {criteria}
Scoring Scale: 0 (Worst) to 100 (Best)

[DATA TO EVALUATE]
Input:
{input}

Context:
{context}

Expected Output (Reference):
{expected_output}

Actual Output (Target):
{output}

[OUTPUT REQUIREMENT]
1. Evaluate the 'Actual Output' objectively.
2. Return the result strictly in valid JSON format.
3. Do not include any markdown formatting (like ```json), explanations, or additional text outside the JSON object.

You must strictly follow the exact following JSON format for response (don't response other things):
{{
  "score": <integer>,   // A value between 0 and 100
  "reason": "<string>" // A concise explanation for the score
}}

FOR Example:
Example 1 (for reference only, not to be included in the prompt):
{{
    "score": 83,
    "reason": "The actual output is mostly correct but misses some details compared to the expected
}}

Example 2 (for reference only, not to be included in the prompt):
{{
    "score": 17,
    "reason": "The actual output is mostly incorrect and does not match the expected output."
}}

[/SYSTEM INSTRUCTION]
"""