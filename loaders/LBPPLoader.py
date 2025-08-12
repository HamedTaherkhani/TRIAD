from datasets import load_dataset
import json
import pickle
import zlib
import base64
import re
from tqdm import tqdm
from function_executor import run_test_cases
from reusable_classes import Function

def decode_str(str_to_decode: str) -> str | list | dict:
    return json.loads(pickle.loads(zlib.decompress(base64.b64decode(str_to_decode.encode("utf-8")))))

def find_import_statements(python_code: str) -> list:
    # Regular expressions to match both 'import' and 'from ... import ...' statements
    import_pattern = r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?'
    from_import_pattern = r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_]*\s+import\s+[a-zA-Z_][a-zA-Z0-9_]*'

    # Find all matches using the regex patterns
    imports = re.findall(import_pattern, python_code, re.MULTILINE)
    from_imports = re.findall(from_import_pattern, python_code, re.MULTILINE)

    # Combine both types of imports into a single list
    return imports + from_imports + ['import numpy as np', 'import pandas as pd', 'import pytest', 'import time', 'from copy import deepcopy', 'import math', 'from math import sqrt', 'import datetime']

def extract_function_signature(text: str) -> str | None:
    """
    Extract the Python function signature from a block of text.

    Args:
        text (str): The input text.

    Returns:
        str | None: The function signature if found, otherwise None.
    """
    # Regex for a Python function signature (multi-line aware)
    pattern = re.compile(
        r'def\s+\w+\s*\(.*?\)(\s*->\s*[\w\[\],.: ]+)?',
        re.DOTALL
    )
    match = pattern.search(text)
    if match:
        # Clean up whitespace and return
        return ' '.join(match.group(0).split())
    return None

class LBPPLoaderPython:
    def __init__(self):
        python = load_dataset("CohereLabs/lbpp", name="python", trust_remote_code=True, split="test")
        self.solutions = []
        self.tests = []
        self.prompts = []
        self.functions = []
        for instance in tqdm(python):
            # print(instance.keys())
            imports = find_import_statements(decode_str(instance["completion"]))
            imports_text = '\n'.join(imports) + '\n'
            cod = imports_text + "\n" + decode_str(instance["completion"])
            tests = decode_str(instance['test_list'])
            res = run_test_cases(cod,tests)
            # print(res)
            prompt = f"{imports_text}\n{instance['signature']}\n   \"\"\"{instance['instruction']}\"\"\""
            task_id = instance['task_id']
            if not all(res):
                continue
            # function_signature = extract_function_signature(instance['instruction'])
            # print(function_signature)
            # print(instance['signature'])
            self.solutions.append(cod)
            self.tests.append(tests)
            self.prompts.append(prompt)
            self.functions.append(
                Function(
                    prompt=prompt,
                    generated_testcases=[],
                    solution=cod,
                    original_tests=tests,
                    generated_solutions=[],
                    task_id=task_id,
                    dataset='LBPPPython'
                )

            )

    def get_solutions(self):
        return self.solutions
    def get_tests(self):
        return self.tests
    def get_prompts(self):
        return self.prompts
    def get_functions(self):
        return self.functions

class LBPPLoaderJava:
    def __init__(self):
        java = load_dataset("CohereLabs/lbpp", name="java", trust_remote_code=True, split="test")
        self.solutions = []
        self.tests = []
        self.prompts = []
    def get_solutions(self):
        return self.solutions
    def get_tests(self):
        return self.tests
    def get_prompts(self):
        return self.prompts
