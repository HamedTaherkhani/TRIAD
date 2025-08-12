
PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring. Do not make any comments on the test cases. Generate 10 to 20 test cases.\n"""
PY_TEST_GENERATION_FEW_SHOT = """Examples:
    func signature:
    def add3Numbers(x, y, z):
        \"\"\" Add three numbers together.
        This function takes three numbers as input and returns the sum of the three numbers.
        \"\"\"
    unit tests:
    assert add3Numbers(1, 2, 3) == 6
    assert add3Numbers(-1, 2, 3) == 4
    assert add3Numbers(1, -2, 3) == 2
    assert add3Numbers(1, 2, -3) == 0
    assert add3Numbers(-3, -2, -1) == -6
    assert add3Numbers(0, 0, 0) == 0\n
    """
separate_unit_output_prompt = '''
Question: 

Assume there exists a python function `{signature}` to solve the task: {description}

A user calls this functions with the input: {entry_point}({unit_input}).
Based on the task objective of the function and the user's input, what is the output of the function if implemented **correctly**?

Make sure your data type of the final answer matches the expected output type of the function.
First explain your reasoning and in the end format your final answer as:

Output: ```<your final answer>```
'''

input_completion_prompt = '''
Given a Python function `{signature}` to solve the following task:
{description}

Write a valid input based on this task description, i.e., an acceptable input consistent with task description that a correct program should be able to execute.
Provide a reasoning for your answer and present your response in the format below:
```
<reasoning>

Arguments: {entry_point}(<all arguments>)
```
Note that you MUST directly write ALL input arguments of the function in the correct order. Skip writing any names of arguments.
'''
test_setup_prompt = """
Given a Python function `{signature}` to solve the following task:
{description}

Write 20 valid python unit test setups (unit test classes) for this task description. The test setups must be diverse and include different scenarios and inputs. Write any other necessary setup for the unit tests to run.

Don't write the assertion statements in the unit tests. Instead write a comment placeholder on the lines that the assertions should be writen. Write #***Assertion statement*** in the lines that should have assertions as a comment. Make sure that to put it as comment (#) so that we won't get syntax error.
Assume the function is implemented in the current file and don't import anything the function from another file. If needed create files or directories in the setUp and remove them in tearDown method. Don't assume there are already any files or directories on system. Delete any created files or resources in tearDown method.
Put each unit test between separate ```python and ``` tags. Make sure every testcase has setUp method and also another method that defines a test case. Write every object initialization or defining any variable in the setUp. Don't write more than 1 test function in each test case.
"""

test_setup_prompt_classeval = """
Given a Python class `{signature}` to solve the following task:
{description}

Write 20 valid python unit test setups (unit test classes) for this python class. The test setups must be diverse and include different scenarios and inputs. Write any other necessary setup for the unit tests to run.

Don't write the assertion statements in the unit tests. Instead write a comment placeholder on the lines that the assertions should be writen. Write #***Assertion statement*** in the lines that should have assertions.
Assume the function is implemented in the current file and don't import anything the function from another file.
Put each unit test between separate ```python and ``` tags. Make sure every testcase has setUp method and also another method that defines a test case. Write every object initialization or defining any variable in the setUp. Don't write more than 1 test function in each test case.
"""
test_holistic_prompt = """
Given a function signature, write 20 valid python unit tests for the given function. The tests must be diverse and include different scenarios and inputs. 
Assume the function is implemented in the current file and don't import anything the function from another file.
Put each unit test between separate ```python and ``` tags. Make sure every testcase has setUp method and also another method that defines a test case. Write every object initialization or defining any variables in the setUp method. Don't write more than 1 test function in each test case.
generate tests for this function:
{signature}
"""

self_consistency_prompt = """You have the following Python unittest code stub that includes 
placeholder lines '#***Assertion statement***'. Your task is to complete the test stub by replacing the '#***Assertion statement***' with an actual assertion statement.

Please think step-by-step (chain-of-thought) about what should be the expected output of the assertions in the unit test based on the given function's description or python class description. Think what the expected output should be given the test stub. Then replace each '#***Assertion statement***' with a single valid unittest assertion statement. produce the valid final unit test. 
Use assertRaises to catch the expected exceptions and errors given the input and the functionality of the function.
Return Python code in your final answer in ```python and ``` tags. Remove all #***Assertion statement*** comments and replace it with a single assertion statement. Don't implement the function.

for example given the following test stub and function description:

def search(i: int, P: list[list[int]]):
   \"\"\"You are given a list L containing n integer vectors with numbers in [1, v], where v>1. Each vector can have a different size,
denoted as s_i. You are also given a number c, which we call a context window. Assume that s_i <= c.
The goal is to create a new list of integer vectors P with a uniform size of c, using the vectors in L, while minimizing the amount of
vectors in P and the free space at the end of each row in P.
The rules for creating P are as follows:
Fit each vector in L into a row in P.
Separate each vector from the list L in the same row in P with a 0.
Pad any remaining space at the end of a row in P, which cannot be filled by any of the remaining vectors in L, with zeros.
A single vector from L cannot be split across multiple rows in P.
If there is a vector in L whose size is larger than c, return an empty list.
If there are multiple ways of minimizing the number of vectors of P, return the one where earlier vectors of L are packed in earlier vectors of P, and as far to the left as possible. To be more precise, if the ith vector of L is packed in the j_i-th vector of P at position k_i, we want the lexicographically least sequence (j_0, k_0), (j_1, k_1), ... of all packings that minimize the number of vectors of P.
So for instance given:
L = [[1,2,3], [4,5,6], [7], [8]]
c = 6
The resulting vector P would not be:
[
[1, 2, 3, 0, 0, 0]
[4, 5, 6, 0, 0, 0]
[7, 0, 8, 0, 0, 0]
]
But rather:
[
[1, 2, 3, 0, 7, 0]
[4, 5, 6, 0, 8, 0]
]
Write it in Python.\"\"\"

import unittest
class TestSearchExactFit(unittest.TestCase):
    def setUp(self):
        self.i = 6
        self.P = [[1, 2, 3], [4, 5, 6], [7], [8]]

    def test_exact_fit_case(self):
        result = search(self.i, self.P)
        #***Assertion statement***

Complete the test stub like:
import unittest
class TestSearchExactFit(unittest.TestCase):
    def setUp(self):
        self.i = 6
        self.P = [[1, 2, 3], [4, 5, 6], [7], [8]]

    def test_exact_fit_case(self):
        result = search(self.i, self.P)
        self.assertEqual(result, self.P) 

Now for the following function or class description:

{prompt}

Complete the given test stub:

{stub}

"""
test_setup_lbpp_fewshot="""

For example given this function:

def search(i: int, P: list[list[int]]):
   \"\"\"You are given a list L containing n integer vectors with numbers in [1, v], where v>1. Each vector can have a different size,
denoted as s_i. You are also given a number c, which we call a context window. Assume that s_i <= c.
The goal is to create a new list of integer vectors P with a uniform size of c, using the vectors in L, while minimizing the amount of
vectors in P and the free space at the end of each row in P.
The rules for creating P are as follows:
Fit each vector in L into a row in P.
Separate each vector from the list L in the same row in P with a 0.
Pad any remaining space at the end of a row in P, which cannot be filled by any of the remaining vectors in L, with zeros.
A single vector from L cannot be split across multiple rows in P.
If there is a vector in L whose size is larger than c, return an empty list.
If there are multiple ways of minimizing the number of vectors of P, return the one where earlier vectors of L are packed in earlier vectors of P, and as far to the left as possible. To be more precise, if the ith vector of L is packed in the j_i-th vector of P at position k_i, we want the lexicographically least sequence (j_0, k_0), (j_1, k_1), ... of all packings that minimize the number of vectors of P.
So for instance given:
L = [[1,2,3], [4,5,6], [7], [8]]
c = 6
The resulting vector P would not be:
[
[1, 2, 3, 0, 0, 0]
[4, 5, 6, 0, 0, 0]
[7, 0, 8, 0, 0, 0]
]
But rather:
[
[1, 2, 3, 0, 7, 0]
[4, 5, 6, 0, 8, 0]
]
Write it in Python.\"\"\"

unittests:

```python
import unittest
class TestSearchExactFit(unittest.TestCase):
    def setUp(self):
        self.i = 6
        self.P = [[1, 2, 3], [4, 5, 6], [7], [8]]

    def test_exact_fit_case(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
```python
import unittest
class TestSearchSingleVectorLargerThanC(unittest.TestCase):
    def setUp(self):
        self.i = 3
        self.P = [[1, 2, 3, 4]]

    def test_single_vector_too_large(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
```python
import unittest
class TestSearchEmptyL(unittest.TestCase):
    def setUp(self):
        self.i = 4
        self.P = []

    def test_empty_input_list(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
```python
import unittest
class TestSearchSingleVectorSmallerThanC(unittest.TestCase):
    def setUp(self):
        self.i = 5
        self.P = [[1, 2]]

    def test_single_vector_padding_needed(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
```python
import unittest
class TestSearchMultipleFitInOneRow(unittest.TestCase):
    def setUp(self):
        self.i = 6
        self.P = [[1, 2], [3], [4]]

    def test_multiple_vectors_fit_together(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
```python
import unittest
class TestSearchMultipleRowsRequired(unittest.TestCase):
    def setUp(self):
        self.i = 5
        self.P = [[1, 2, 3], [4, 5], [6, 7]]

    def test_multiple_rows_output(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
```python
import unittest
class TestSearchZeroInVectors(unittest.TestCase):
    def setUp(self):
        self.i = 5
        self.P = [[0, 1], [2, 3], [4]]

    def test_vectors_with_zero_values(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
```python
import unittest
class TestSearchLargeCValue(unittest.TestCase):
    def setUp(self):
        self.i = 10
        self.P = [[1, 2, 3], [4], [5, 6, 7, 8, 9]]

    def test_large_c_with_padding(self):
        result = search(self.i, self.P)
        #***Assertion statement***
```
"""

test_setup_prompt_lbpp = """
Given a python function signature, write 20 valid python unit test stubs for the given function. The test stubs must be diverse and include different scenarios and inputs. Write any other necessary setup for the unit tests to run.
Don't write the assertion statements in the unit tests. Instead write a comment placeholder on the lines that the assertions should be writen. Write #***Assertion statement*** in the lines that should have assertions.
Assume the function is implemented in the current file and don't import anything the function from another file.
Put each unit test between separate ```python and ``` tags. Make sure every testcase has setUp method and also another method that defines a test case. Write every object initialization or defining any variables in the setUp method. Don't write more than 1 test function in each test case.

generate test stub for this function:
{signature}
""" + test_setup_lbpp_fewshot

PY_TEST_GENERATION_FEW_SHOT_BigCodeBench = """
Examples:

Func signature:
import re
import pandas as pd
from datetime import datetime

def task_func(log_file):
    \"\"\"
    Extracts logging information such as message type, timestamp, and the message itself from a log file and
    stores the data in a CSV format. This utility is ideal for converting plain text logs into a more s
    tructured format that can be easily analyzed. The log is the format of 'TYPE: [TIMESTAMP (YYYY-MM-DD HH:MM:SS)] - MESSAGE'.

    Parameters:
    log_file (str): The file path to the log file that needs to be parsed.

    Returns:
    str: The file path to the newly created CSV file which contains the structured log data.

    Requirements:
    - re
    - pandas
    - datetime

    Raises:
    ValueError: If the timestamp in any log entry is invalid or if no valid log entries are found.

    Example:
    >>> output_path = task_func('server.log')
    >>> print(output_path)
    log_data.csv
    \"\"\"

unit tests:

```python
import unittest
import os

class TestValidLogFile(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_valid.log'
        with open(self.log_file, 'w') as f:
            f.write('INFO: [2023-01-01 12:00:00] - Server started\n')
            f.write('ERROR: [2023-01-01 12:05:00] - An error occurred\n')

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        csv_file = 'log_data.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_task_func_creates_csv(self):
        output = task_func(self.log_file)
        ***Assertion statement***
```

```python
import unittest
import os

class TestInvalidTimestamp(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_invalid_timestamp.log'
        with open(self.log_file, 'w') as f:
            f.write('INFO: [2023-13-01 12:00:00] - Invalid month\n')

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_invalid_timestamp_raises_value_error(self):
        pass
        ***Assertion statement***
```

```python
import unittest
import os

class TestNoValidLogEntries(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_no_valid.log'
        with open(self.log_file, 'w') as f:
            f.write('This is an invalid log entry\n')

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_no_valid_entries_raises_value_error(self):
        pass
        ***Assertion statement***
```

```python
import unittest
import os

class TestEmptyLogFile(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_empty.log'
        open(self.log_file, 'w').close()

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_empty_log_file_raises_value_error(self):
        pass
        ***Assertion statement***
```

```python
import unittest
import os

class TestMixedValidAndInvalidEntries(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_mixed.log'
        with open(self.log_file, 'w') as f:
            f.write('INFO: [2023-01-01 12:00:00] - Server started\n')
            f.write('INVALID ENTRY\n')
            f.write('ERROR: [2023-01-01 12:05:00] - An error occurred\n')

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        csv_file = 'log_data.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_mixed_entries_processes_valid_only(self):
        output = task_func(self.log_file)
        ***Assertion statement***
        with open(output, 'r') as f:
            content = f.read()
            ***Assertion statement***
```

```python
import unittest
import os

class TestDuplicateEntries(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_duplicates.log'
        with open(self.log_file, 'w') as f:
            f.write('INFO: [2023-01-01 12:00:00] - Server started\n')
            f.write('INFO: [2023-01-01 12:00:00] - Server started\n')

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        csv_file = 'log_data.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_duplicate_entries_processed_correctly(self):
        output = task_func(self.log_file)
        ***Assertion statement***
        with open(output, 'r') as f:
            lines = f.readlines()
            ***Assertion statement***
```

```python
import unittest
import os

class TestDifferentTimestampFormats(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_timestamp_formats.log'
        with open(self.log_file, 'w') as f:
            f.write('INFO: [2023/01/01 12:00:00] - Incorrect date format\n')

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        csv_file = 'log_data.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_timestamp_format_validation(self):
        pass
        ***Assertion statement***
```

```python
import unittest
import os

class TestLargeLogFile(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_large.log'
        with open(self.log_file, 'w') as f:
            for i in range(1000):
                f.write(f'INFO: [2023-01-01 12:{i%60:02d}:00] - Message {i}\n')

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        csv_file = 'log_data.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_large_log_file_processing(self):
        output = task_func(self.log_file)
        ***Assertion statement***
        with open(output, 'r') as f:
            lines = f.readlines()
            ***Assertion statement***

```
"""

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
        "from functools import *"
    ]}