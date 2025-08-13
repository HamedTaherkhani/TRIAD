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
    planning_prompt = f"""
    You are an expert software tester and code auditor specializing in verifying the correctness of code solutions.

    You will be given:
    1. The **original prompt** describing the coding task.
    2. A **baseline solution** (LLM-generated code) that attempts to solve the task.

    Your job is to produce a **list of the 10 most critical and insightful verification questions**.  
    These questions will be used to **evaluate whether the baseline solution is correct, robust, and aligned with the given prompt**.

    Your questions should:
    - Test **core correctness** (does it produce the right outputs for given inputs?).
    - Cover **edge cases** (unusual or extreme inputs that may break the solution).
    - Examine **reasoning and logic** (is the algorithm correct and complete?).
    - Check **error handling and robustness**.
    - Identify **assumption validation** (does the code rely on unstated or unsafe assumptions?).

    Be specific — each question should target one concrete aspect of the solution.  
    Avoid vague questions like "Is the code correct?" — instead, ask targeted, testable questions.

    ---

    **PROMPT:**
    \"\"\"{prompt}\"\"\"

    **BASELINE SOLUTION:**
    \"\"\"{baseline_response}\"\"\"

    **Output format:**
    A numbered list of exactly 10 clear, concise, and targeted verification questions.
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

def execute_verifications(prompt: str, verification_questions: List[str], answer:str, llm_requester:LLMRequester) -> Tuple[str, bool]:
    """
    Step 3: For each verification question, obtain an answer using Gemini.
    """
    verification_prompt = f"""
    You are an expert code verifier and fact-checking assistant for Python code solutions.

    You will be given:
    1. **Problem statement** – the exact coding task to solve (this is the *ground truth*).
    2. **Proposed solution** – a Python code implementation that may contain bugs.
    3. **Verification questions** – targeted checks designed to confirm correctness and alignment with the problem.

    Your tasks:
    1. **Answer ALL verification questions** one by one.
       - For each question:
         - Provide a **direct answer** ("Yes", "No", "Partially", or a specific fact).
         - Provide an explanation using:
           - The problem statement (ground truth)
           - The proposed code
           - Logical reasoning about expected vs. actual behavior
    2. After answering all questions, assess **overall correctness** of the code:
       - Decide whether the code has **major issues** that make it incorrect, incomplete, or unsafe to use.
       - If major issues exist, explicitly state:
         ```
         ##MAJOR ISSUES##
         ```
       - If no major issues exist, explicitly state:
         ```
         ##NO MAJOR ISSUES##
         ```
    ---

    **Problem Statement (Ground Truth)**:
    \"\"\"{prompt}\"\"\"

    **Proposed Solution (May Contain Bugs)**:
    ```python
    {answer}
    ```

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
    no_major_issues = "##NO MAJOR ISSUES##".lower() in answer.lower()
    return answer, no_major_issues

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
    final_prompt = f"""
    You are a **senior Python expert** responsible for **verifying, correcting, and improving** a code solution.

    You are given:
    1. **Original problem prompt** – the ground truth requirements.
    2. **Baseline code solution** – an initial attempt at solving the problem (may contain mistakes).
    3. **Verification questions** – targeted checks for correctness.
    4. **Verification answers** – results of these checks, including explanations of correctness or errors.

    Your tasks:
    1. **Analyze** the verification answers to:
       - Identify any **mistakes, incorrect logic, missing requirements, or hallucinated functionality** in the baseline code.
       - Confirm if there are **no mistakes**.
    2. If mistakes exist:
       - Produce a **fully corrected Python solution** that meets *all* requirements in the original problem prompt.
       - Ensure the revised code is:
         - Correct and bug-free.
         - Robust to edge cases.
    3. **Explain** the mistakes found in the baseline solution, referencing the verification answers as evidence.
    4. If no mistakes exist output the Baseline Code Solution in the response.
    ---

    **Original Prompt (Ground Truth)**:
    \"\"\"{prompt}\"\"\"

    **Baseline Code Solution**:
    ```python
    {baseline_response}

Verification Questions:
\"\"\"{verification_prompt}\"\"\"

Verification Answers:
\"\"\"{verification_answers}\"\"\"

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
    verif_answers,no_major_issues = execute_verifications(prompt, verif_questions, answer, llm_requester)
    if no_major_issues:
        print("No major issues found.")
        return answer, llm_requester.get_total_usage()
    print("Major issues found. Trying to solve the issues in the code...")
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
