#!/usr/bin/env python3
import sys
import argparse
import os
import pickle
import re
from typing import List
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
from openai import OpenAI
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# ----------------------------------------------------------------------------
# Local imports for your project structure
# ----------------------------------------------------------------------------
# from openai import OpenAI  # or your actual openai usage
from function_executor import run_unit_tests_parallel
# from loaders.ClassEvalLoader import ClassEvalLoader  # The class that loads your dataset
from loaders import BigCodeLoader, LBPPLoaderPython
from prompts import test_holistic_prompt, test_setup_prompt, test_setup_prompt_classeval, PY_TEST_GENERATION_FEW_SHOT_BigCodeBench, test_setup_prompt_lbpp, self_consistency_prompt
from reusable_classes import Function, TestCase
from llm_requester import FireworksAPIRequester, LLMRequester, OpenaiRequester, backends, init_llm
###############################################################################
# TestCodeGenerator
###############################################################################


class TestCodeGenerator:
    """
    A class that handles:
      - Extracting the function signature from a prompt
      - Generating initial test stubs (with placeholders)
      - Enriching those stubs with valid assertions (chain-of-thought)
      - Using a simplistic frequency-based approach for self-consistency
    """

    def __init__(self, llm_interface: LLMRequester):
        self.llm_interface = llm_interface

    @staticmethod
    def get_function_signature_from_string(func_str: str) -> str:
        """
        Extracts the function signature from a string using a simple regex.
        Returns something like 'def my_function(x, y)'
        or a fallback if it can't find a match.
        """
        pattern = r"def\s+(\w+)\s*\((.*?)\)\s*:"
        match = re.search(pattern, func_str)
        if match:
            func_name = match.group(1)
            params = match.group(2)
            return f"def {func_name}({params})"
        return "def your_function(...)"  # fallback

    @staticmethod
    def get_class_name(func_str: str) -> str:
        match = re.search(r'class\s+(\w+)\s*:', func_str)

        if match:
            class_name = match.group(1)
            return class_name
        else:
            print("No class found.")
            return ''

    def generate_stubs_for_problem(self, problem: Function) -> Function:
        """
        Calls the LLM to generate a single string containing 5-10 unittest stubs.
        Splits those stubs into a list of strings (one per stub) and stores them
        in problem.generated_tests.
        """

        # Use the test_setup_prompt. E.g.:
        #   test_setup_prompt = """Given a Python function {signature} ... Write 5-10 valid python unit test setups..."""
        if problem.dataset == "ClassEval":
            class_name = self.get_class_name(problem.prompt)
            prompt_str = test_setup_prompt_classeval.format(signature=class_name, description=problem.prompt)
        elif problem.dataset == "LBPPPython":
            prompt_str = test_setup_prompt_lbpp.format(signature=problem.prompt)
        elif problem.dataset == "BigCodeBenchHard":
            signature = self.get_function_signature_from_string(problem.prompt)
            prompt_str = test_setup_prompt.format(signature=signature, description=problem.prompt) + PY_TEST_GENERATION_FEW_SHOT_BigCodeBench
        else:
            raise Exception(f"Unknown dataset: {problem.dataset}")

        completions = self.llm_interface.get_completion(messages=[{'content':prompt_str}], n=1, temperature=0)
        if not completions:
            problem.generated_testcases = []
            return problem

        raw_output = completions[0]
        # print(raw_output)
        # print('*'*100)
        code_blocks = re.findall(r"```python(.*?)```", raw_output, re.DOTALL)

        # Optional: strip leading/trailing whitespace from each block
        code_blocks = ["import unittest\n" + block.strip() for block in code_blocks]
        testcases = []
        for block in code_blocks:
            testcases.append(TestCase(
                text=block,
            ))
        problem.generated_testcases = testcases
        return problem

    def chain_of_thought_prompt(self, stub: str, prompt: str) -> str:
        """
        Returns a chain-of-thought style prompt asking the LLM to add valid assertions.
        """
        return self_consistency_prompt.format(stub=stub, prompt=prompt)

    def self_consistency_v1(self, testcases: List[str]):
        frequency = {}
        counter = 0
        for c in testcases:
            counter += 1
            code_blocks = re.findall(r"```python(.*?)```", c, re.DOTALL)[0]
            # print(code_blocks)
            frequency[code_blocks] = frequency.get(code_blocks, 0) + 1
        # print('*'*100)
        best_code = max(frequency, key=frequency.get)
        print(f'counter is {counter}')
        print(f'frequency length is {len(frequency.keys())}')
        return best_code

    def self_consistency(self, testcases: List[str]):
        # Step 1: Build the list of (testcase_string, set_of_assertions)
        test_assertions = []
        for testcase in testcases:
            try:
                code_block = re.findall(r"```python(.*?)```", testcase, re.DOTALL)[0]
            except IndexError:
                continue
            lines = code_block.splitlines()
            # Gather all lines containing 'assert'
            assertion_lines = set(line.strip() for line in lines if 'assert' in line)
            test_assertions.append((code_block, assertion_lines))

        # Step 2: Count how many times each unique set of assertions appears
        # (use frozenset so it's hashable as a dict key)
        assertion_dict = defaultdict(list)
        for testcase, assertion_lines in test_assertions:
            assertion_dict[frozenset(assertion_lines)].append(testcase)

        # Step 3: The number of unique tests is simply how many unique sets of assertions we have
        num_unique_tests = len(assertion_dict)

        # Step 4: Find the set of assertions with the largest list of matching testcases
        #         Then pick one of them (e.g. the first) to identify as "most repeated"
        try:
            max_assertion_set = max(assertion_dict, key=lambda k: len(assertion_dict[k]))
            most_repeated_test = assertion_dict[max_assertion_set][0]
            repeats = len(assertion_dict[max_assertion_set])
        except ValueError:
            most_repeated_test = ''
            max_assertion_set = []
            repeats =0
        # Return or print the results
        print("Number of unique tests (based on assertions):", num_unique_tests)
        # print("Test with the highest frequency:", most_repeated_test)
        print("Number of repeats for that test:", repeats)

        # Optionally return them if you need them programmatically
        return most_repeated_test

    def enrich_problem_tests(self, problem: Function, n_completions=1) -> Function:
        """
        For each stub in problem.generated_tests, generate multiple completions with
        chain-of-thought prompting, pick the best final code using self-consistency,
        and store it back in problem.generated_tests.
        """
        for stub in problem.generated_testcases:
            # We prompt for multiple completions to apply self-consistency
            prompt_str = self.chain_of_thought_prompt(stub.text, problem.prompt)
            completions = self.llm_interface.get_completion(
                messages=[{'content': prompt_str}], n=n_completions, temperature=0
            )
            if not completions:
                continue

            # Self-consistency: pick the identical test that appears most frequently
            best_code = self.self_consistency(completions)
            stub.text = best_code

        # problem.generated_testcases = new_tests
        return problem

    def generate_hollistic_tests(self,problem: Function)-> Function:
        prompt_str = test_holistic_prompt.format(signature=problem.prompt)
        completions = self.llm_interface.get_completion(messages=[{'content': prompt_str, 'role':'user'}], n=1, temperature=0)
        if not completions:
            problem.generated_testcases = []
            return problem

        raw_output = completions[0]
        # print(raw_output)
        # print('*'*100)
        # ---------------------------------------------------------------------
        # Split the LLM output into multiple test stubs
        # For example, we might look for lines that start with 'class Test'
        # or something similar. Adjust as needed based on the LLM's format.
        # ---------------------------------------------------------------------
        code_blocks = re.findall(r"```python(.*?)```", raw_output, re.DOTALL)

        # Optional: strip leading/trailing whitespace from each block
        code_blocks = [block.strip() for block in code_blocks]
        print(code_blocks)
        testcases = []
        for block in code_blocks:
            testcases.append(TestCase(
                text=block,
            ))
        problem.generated_testcases = testcases
        return problem


###############################################################################
# Parallel Operations
###############################################################################
def parallel_generate_stubs(problem: Function, model: str, backend:str) -> Function:
    """
    Helper function for step 1 (so it can be used with ProcessPoolExecutor).
    Creates a local LLMInterface and TestCodeGenerator, returns the problem with stubs.
    """
    llm = init_llm(model, backend)
    generator = TestCodeGenerator(llm)
    return generator.generate_stubs_for_problem(problem)

def parallel_generate_tests_hollistic(problem: Function, model: str, backend:str):
    try:
        llm = init_llm(model, backend)
        generator = TestCodeGenerator(llm)
        return generator.generate_hollistic_tests(problem)
    except Exception as ex:
        raise ex

def parallel_enrich_tests(problem: Function, model: str, n_completions: int, backend:str) -> Function:
    """
    Helper function for step 2 (so it can be used with ProcessPoolExecutor).
    Creates a local LLMInterface and TestCodeGenerator, returns the problem with final tests.
    """
    llm = init_llm(model, backend)
    generator = TestCodeGenerator(llm)
    return generator.enrich_problem_tests(problem, n_completions)

def evaluate_problems_and_update(problems: List[Function]):
    all_results = []
    num_passed = 0
    total = 0
    for problem in problems:
        results = run_unit_tests_parallel(test_list=[p.text for p in problem.generated_testcases], code_str=problem.solution)
        # print(problem.solution)
        # print([p.text for p in problem.generated_testcases])
        # print(results)
        # print('*'*100)
        for idx, test in enumerate(problem.generated_testcases):
            problem.generated_testcases[idx].is_valid = results[idx][0]
        total += len(results)
        num_passed += len([r for r in results if r[0]])
        all_results.append(results)
    print(f'num passed = {num_passed}')
    print(f'total = {total}')
    print(f'valid ratio = {num_passed / total}')

###############################################################################
# Main Script
###############################################################################
def main(args):
    """
    1) Parse arguments for dataset name (e.g. 'ClassEval') and model name (e.g. 'gpt-4').
    2) Load the dataset (ClassEvalLoader).
    3) Step 1: Generate 20 stubs for each problem in parallel.
    4) Save to "with_stubs.pkl".
    5) Step 2: Read "with_stubs.pkl", enrich stubs with assertions in parallel, save "final_tests.pkl".
    """
    # -------------------------------------------------------------------------
    # 1) Load the dataset
    # -------------------------------------------------------------------------
    if args.dataset == "LBPPPython":
        problems:List[Function] = LBPPLoaderPython().get_functions()
    elif args.dataset == "BigCodeBenchHard":
        problems:List[Function] = BigCodeLoader(hard=1).get_functions()
    else:
        # If you had other loaders, you could add them here
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    # problems = problems[:5]
    # print(f"Loaded {len(problems)} problems from dataset '{args.dataset}'.")
    if args.approach  == "self-consistency":
        step1_output = f'generated_tests/stub/{args.dataset}-{args.model}.pkl'
        if os.path.exists(step1_output):
            print(f'loading test stubs from {step1_output}...')
            with open(step1_output, 'rb') as a_file:
                updated_problems: list[Function] = pickle.load(a_file)
        else:
            # -------------------------------------------------------------------------
            # 2) Step 1: Generate stubs in parallel
            # -------------------------------------------------------------------------
            print("\n=== Step 1: Generating 20 unittest stubs for each problem ===")
            with ThreadPoolExecutor(max_workers=10) as executor:
                updated_problems = list(
                    tqdm(
                        executor.map(
                            parallel_generate_stubs,
                            problems,
                            [args.model] * len(problems),
                            [args.backend] * len(problems),
                        ),
                    total = len(problems),  # Required for tqdm to know the length
                    desc = "Processing problems"
                    )

                )
            evaluate_problems_and_update(updated_problems)
            with open(step1_output, "wb") as f:
                pickle.dump(updated_problems, f)
            print(f"Step 1 done. Generated stubs saved to '{step1_output}'.")
            print('evaluating stubs...')

        # -------------------------------------------------------------------------
        # 3) Step 2: Enrich stubs with assertions, using chain-of-thought prompts
        # -------------------------------------------------------------------------
        print("\n=== Step 2: Enrich stubs with assertions (chain-of-thought) ===")
        # Reload from pickle (demonstrates how you'd typically do it in a real pipeline)
        # for sample in updated_problems:
        #     if sample.generated_tests:
        #         print(len(sample.generated_tests))
        os.makedirs(f'generated_tests/final_tests/{args.approach}', exist_ok=True)
        step2_output = f"generated_tests/final_tests/{args.approach}/{args.dataset}-{args.model}.pkl"
        if os.path.exists(step2_output):
            print('loading final tests from ' + step2_output)
            with open(step2_output, 'rb') as a_file:
                final_problems: list[Function] = pickle.load(a_file)
        else:
            n_completions = 5
            with ThreadPoolExecutor(max_workers=10) as executor:
                final_problems = list(
                    tqdm(
                        executor.map(
                            parallel_enrich_tests,
                            updated_problems,
                            [args.model] * len(updated_problems),
                            [n_completions] * len(updated_problems),
                            [args.backend] * len(updated_problems),
                        ),
                        total=len(updated_problems),
                        desc="Processing problems"
                    )
                )

            # Save final results
            evaluate_problems_and_update(final_problems)
            with open(step2_output, "wb") as f:
                pickle.dump(final_problems, f)
            print(f"Step 2 done. Final tests with assertions saved to '{step2_output}'.")
            print('running evaluation on final tests')
            # for problem in final_problems:
            #     for test in problem.generated_testcases:
            #         print(test.text)
            #     print('*'*100)

    elif args.approach == 'holistic':
        os.makedirs(f'generated_tests/final_tests/{args.approach}', exist_ok=True)
        step2_output = f"generated_tests/final_tests/{args.approach}/{args.dataset}-{args.model}.pkl"
        with ThreadPoolExecutor(max_workers=10) as executor:
            final_problems = list(
                tqdm(
                    executor.map(
                        parallel_generate_tests_hollistic,
                        problems,
                        [args.model] * len(problems),
                        [args.backend] * len(problems),
                    ),
                    total=len(problems),
                    desc="Processing problems"
                )
            )
        evaluate_problems_and_update(final_problems)
        with open(step2_output, "wb") as f:
            pickle.dump(final_problems, f)

    print("\nAll done!\n")


if __name__ == "__main__":

    approaches = ["self-consistency", "holistic"]
    parser = argparse.ArgumentParser(description="Run the script with specified dataset and LLM.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset (e.g. LBPPPython)",
                        choices=VALID_DATASETS)
    parser.add_argument("--model", type=str, required=True, help="Name of LLM model (e.g. gpt-4 or gpt-3.5-turbo)",
                        choices=VALID_LLMS)
    parser.add_argument("--approach", type=str, required=True,choices=approaches)
    parser.add_argument('--backend', type=str, required=True, choices=backends)
    args = parser.parse_args()
    main(args)
