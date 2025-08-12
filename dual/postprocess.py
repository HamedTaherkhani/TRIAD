# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from reusable_classes import Function
import json
STOP_TOKEN = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']


import ast

def extract_function_name(code):
    # Parse the code into an AST
    tree = ast.parse(code)

    # Traverse the AST to find the first function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name

    return None

class PostProcessor:
    @staticmethod
    def map_task_id_for_solution(functions:list[Function], dataset_name):
        result = []
        for ix, func in enumerate(functions):
            for comp in func.generated_solutions:
                result.append({
                    'task_id': func.task_id,
                    'prompt': func.prompt,
                    'test':'\n'.join(func.original_tests),
                    'entry_point': extract_function_name(func.solution),
                    'completion': comp
                })
        return result

    @staticmethod
    def map_task_id_for_test_case(generated_tests: list[Function], dataset_name):
        test_cases_by_task = defaultdict(list)
        for idx, func in enumerate(generated_tests):
            generated_tests = [ff.text for ff in func.generated_testcases]
            task_id = func.task_id
            test_cases_by_task[task_id] = generated_tests
        return test_cases_by_task

    @staticmethod
    def solution_extract(content):
        for identifier in STOP_TOKEN:
            if identifier in content:
                content = content.split(identifier)[0]
        return content
    
    @staticmethod
    def test_case_extract(content, entry_point):
        def _truncate(content):
            for identifier in STOP_TOKEN:
                if identifier in content:
                    content = content.split(identifier)[0]
            return content.strip()
        
        split_by_assert = [f'assert {part}'.strip() for part in f'assert {content}'.split('assert ') if (entry_point.strip() in part) and len(part.strip()) > 0]
        truncated_test_cases = [_truncate(i) for i in split_by_assert]
        checked_assertions = [i for i in truncated_test_cases if PostProcessor._check_test_case_validation(i)]
        return checked_assertions

    @staticmethod
    def _check_test_case_validation(test_case):
        if len(test_case.strip()) < 1:
            return False
        if 'assert' not in test_case:
            return False
        try:
            multi_line_test_case = test_case.replace("\n", "\n    ")
            assert_in_a_block = f'try:\n    {multi_line_test_case}\nexcept:\n    pass\n'
            compile(assert_in_a_block, '', 'exec')
            return True
        except Exception:
            return False