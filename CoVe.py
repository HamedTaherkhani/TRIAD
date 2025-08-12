from typing import Dict, List, Tuple
from llm_requester import LLMRequester, init_llm, TokenUsage
import re

def separate_python_code_blocks(text: str) -> List[str]:
    """
    Extracts all Python code blocks from a string.

    Args:
        text (str): The input text containing multiple Python code blocks.

    Returns:
        List[str]: A list of strings, each string being the content of a Python code block.
    """
    # Regular expression to match code blocks that start with ```python and end with ```
    pattern = r"```python\s*(.*?)```"

    # Use re.DOTALL to allow '.' to match newline characters
    code_blocks = re.findall(pattern, text, re.DOTALL)

    # Optionally, you can strip trailing or leading whitespace for each block
    return [block.strip() for block in code_blocks]


def plan_verification_questions(prompt: str, baseline_response: str, llm_requester:LLMRequester) -> List[str]:
    """
    Step 2: Given the original prompt and the baseline response, generate a set of
    verification questions that check for mistakes or hallucinations.
    """
    planning_prompt = f"""You are an expert in verifying code solutions.
Below is the original prompt and a baseline code solution.
Your task is to generate a list of questions that can verify the correctness of the baseline code.
Each question should ask for a specific aspect or test case to confirm the code's correctness or a question about the reasoning steps of the given solution. Devise the most 10 critical questions.
Do not refer to the code directly. Instead, generate general verification questions.

PROMPT:
\"\"\"{prompt}\"\"\"

BASELINE SOLUTION:
\"\"\"{baseline_response}\"\"\"

Provide your verification questions as a numbered list.
"""
    planning_response = llm_requester.get_completion(messages=[{
                    "role": "user",
                    "content": planning_prompt
                }],temperature=0,n=1)
    # Ensure planning_response is a string (adjust based on your API)
    if isinstance(planning_response, (list, tuple)):
        planning_response = planning_response[0]
    # print('planning response:\n', planning_response)
    questions = []
    for line in planning_response.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            # Remove numbering and punctuation.
            q = line.split('.', 1)[-1].strip()
            questions.append(q)
    if not questions:
        questions = [line.strip() for line in planning_response.splitlines() if line.strip()]
    return questions

def execute_verifications(prompt: str, verification_questions: List[str], answer:str, llm_requester:LLMRequester) -> str:
    """
    Step 3: For each verification question, obtain an answer using Gemini.
    """
    verification_prompt = f"""You are a fact-checking assistant for code generation.
Answer the following verification questions according to the problem and the solution to help confirm the correctness of the Python code solution.
For each question provide answer and explanations. Answer all the questions. Pay attention to the problem and the provided examples in the problem to infer the correct answer. The problem is the ground truth but the solution may have bugs.
#Problem#:
{prompt}
#Solution#:
{answer}
#Verification Questions#:
"""
    for idx, question in enumerate(verification_questions, start=1):
        verification_prompt += f"""
Verification Question {idx}: {question}
"""
    verification_prompt += "\n#Answer#:\n"
    # print('Verification Prompt##:\n', verification_prompt)
    answer = llm_requester.get_completion(messages=[{
                    "role": "user",
                    "content": verification_prompt
                }], n=1, temperature=0)
    # print(answer)
    # If answer comes as a list or tuple, use the first element
    if isinstance(answer, (list, tuple)):
        answer = answer[0]
    return answer

def generate_final_verified_response(prompt: str, baseline_response: str, verif_questions, verification_answers: str, llm_requester:LLMRequester) -> str:
    """
    Step 4: Using the prompt, the baseline response, and the verification Q&A pairs,
    generate the final verified response.
    """
    verification_prompt = ''
    for idx, question in enumerate(verif_questions, start=1):
        verification_prompt += f"""
    Verification Question {idx}: {question}
    """
    final_prompt = f"""You are a senior code expert tasked with verifying and improving code solutions.
Below you are provided with:
1. The original problem prompt.
2. A baseline code solution.
3. A list of verification questions and their answers related to the baseline.

Using the verification responses, identify any mistakes or hallucinated parts in the baseline solution,
and provide a revised, correct Python implementation that addresses any issues detected. Also provide an explanation of the mistakes in the baseline solution based on the verification answers. If there is not any mistakes in the baseline solution, print ##NO mistakes found in the original solution##. Also indicate if the mistake is minor or major.
##Examples of major mistakes:
**The response provides the wrong answer to the user.
**The code does not have the functionality as described in the response (major unexpected side effects and/or does not perform the action described).
Example: [javascript how to sort array numerically] -> "Use Array.sort() to sort an array" (sort() turns numbers into strings and sorts alphabetically).
**The code does not handle likely edge cases.
**The code implements incorrect logic.

##Examples of minor mistakes:
**The code includes minor problems that are straightforward to fix (e.g. missing imports or small syntax errors).
**The code does not handle uncommon edge cases.
**The code has readability issues.
**There are redundant operations or variables in the code.
**The code has performance issues.

At the end, print ##Minor issues## or ##Major issues## or ##No issues##

Original Prompt:
\"\"\"{prompt}\"\"\"

Baseline Code Solution:
\"\"\"{baseline_response}\"\"\"

Verification Questions:
\"\"\"{verification_prompt}\"\"\"

Verification Answers:
\"\"\"{verification_answers}\"\"\"

Please provide the final revised code solution in the original language in the baseline solution.

Put the final revised code solution inside ```python and ``` tags after ###Final Response:
###Final Response
```python
your code here

```
"""
    # print('final prompt:')
    # print(final_prompt)
    final_response = llm_requester.get_completion(messages=[{
                    "role": "user",
                    "content": final_prompt
                }], n=1, temperature=0)
    if isinstance(final_response, (list, tuple)):
        final_response = final_response[0]
    return final_response

def chain_of_verification_pipeline(prompt: str, answer: str,model,backend) -> Tuple[str, TokenUsage]:
    """
    Run the complete chain-of-verification pipeline.
    Returns a dictionary containing all relevant outputs.
    """
    llm_requester = init_llm(model=model, backend=backend)
    results = {}
    results["prompt"] = prompt
    # Use the provided answer as the baseline.
    results["baseline_response"] = answer
    # print('baseline answer:')
    # print(answer)
    # Generate verification questions.
    verif_questions = plan_verification_questions(prompt, answer, llm_requester)
    results["verification_questions"] = verif_questions
    # print('verification questions:')
    # print(verif_questions)
    # time.sleep(2)
    # Execute the verification queries.
    verif_answers = execute_verifications(prompt, verif_questions, answer, llm_requester)
    results["verification_answers"] = verif_answers
    # print('===verification answers:===')
    # print(verif_answers)
    # time.sleep(2)
    # Generate the final verified response.
    final_verified = generate_final_verified_response(prompt, answer, verif_questions, verif_answers, llm_requester)
    results["final_verified_response"] = final_verified
    # no_issues = "No issues"
    # minor_issues = "Minor issues"
    # major_issues = "Major issues"
    # if no_issues in final_verified:
    #     results['issue_type'] = 'No issues'
    # elif minor_issues in final_verified:
    #     results['issue_type'] = 'Minor issues'
    # elif major_issues in final_verified:
    #     results['issue_type'] = 'Major issues'
    # else:
    #     results['issue_type'] = 'ND'
    try:
        verified_code = separate_python_code_blocks(final_verified.split('###Final Response')[1])
    except IndexError:
        verified_code = separate_python_code_blocks(final_verified)
    verified_code = '\n'.join(verified_code)
    # print("=== Final Verified Response ===")
    # print(verified_code)
    return verified_code, llm_requester.get_total_usage()
