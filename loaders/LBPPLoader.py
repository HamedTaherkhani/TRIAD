from datasets import load_dataset
import json
import pickle
import zlib
import base64
import re

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
    return imports + from_imports + ['import numpy as np', 'import pandas as pd']

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
        for instance in python:
            imports = find_import_statements(decode_str(instance["completion"]))
            imports_text = '\n'.join(imports) + '\n'
            # function_signature = extract_function_signature(instance['instruction'])
            # print(function_signature)
            # print(instance['signature'])
            self.solutions.append(decode_str(instance["completion"]))
            self.tests.append("\n".join(decode_str(instance['test_list'])))
            self.prompts.append(f"{imports_text}\n{instance['signature']}\n   \"\"\"{instance['instruction']}\"\"\"")

    def get_solutions(self):
        return self.solutions
    def get_tests(self):
        return self.tests
    def get_prompts(self):
        return self.prompts

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
