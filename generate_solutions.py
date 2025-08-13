import ast
import argparse
import sys
import re
import os
import multiprocessing
from tqdm import tqdm
from loaders import BigCodeLoader, LBPPLoaderPython
from llm_requester import OpenaiRequester, HuggingfaceRequester, GeminiRequester, VertexAIRequester, AntropicRequester, FireworksAPIRequester, backends, init_llm, TokenUsage
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
import pickle
from reusable_classes import Function
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from CoVe import chain_of_verification_pipeline
IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"
def get_function_name(func_str):
    pattern = r"def\s+(\w+)\s*\((.*?)\)\s*:"
    match = re.search(pattern, func_str)

    if match:
        func_name = match.group(1)
        return func_name
    else:
        return None


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


def generate_solutions(dataset_name, llm_name, approach, backend, max_workers=8):
    os.makedirs(f'generated_solutions/vanilla', exist_ok=True)
    out_path = f'generated_solutions/vanilla/{dataset_name}-{llm_name}.pkl'
    total_token_usage = TokenUsage()
    if os.path.exists(out_path):
        print('Loading baseline responses from ', out_path)
        with open(out_path, 'rb') as f:
            data: List[Function] = pickle.load(f)
    else: ## generate baseline response
        print('Generating baseline responses ...')
        if dataset_name == 'LBPPPython':
            dataset: List[Function] = LBPPLoaderPython().get_functions()
        elif dataset_name == 'BigCodeBenchHard':
            dataset: List[Function] = BigCodeLoader(hard=1).get_functions()
        else:
            raise Exception(f"Dataset {dataset_name} is not supported.")

        def process_function(func):
            llm_requester = init_llm(model=llm_name, backend=backend)
            responses = llm_requester.get_completion(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You are a python developer who implements correct implementations. "
                            "Don't write imports like from your_module import sth. "
                            f"Implement the following code:\n {func.prompt}. Don't generate test cases and just implement the code."
                        ),
                    }
                ],
                temperature=0.4,
                n=7
            )

            solutions = []
            for s in responses:
                # print(s)
                sols = '\n'.join(separate_python_code_blocks(s))
                print(sols)
                print('*' * 100)
                solutions.append(sols)
            func.generated_solutions = solutions
            return func, llm_requester.get_total_usage()

        # Run in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_function, func) for func in dataset]
            data = []
            for f in as_completed(futures):
                func, usage = f.result()
                data.append(func)
                total_token_usage += usage
        # Save results
        with open(out_path, "wb") as file:
            pickle.dump(data, file)
    if approach == 'CoVe':
        print('Verifying solutions using CoVe...')
        # data = data[:1]
        os.makedirs(f'generated_solutions/CoVe', exist_ok=True)
        out_path = f'generated_solutions/CoVe/{dataset_name}-{llm_name}.pkl'

        for func in tqdm(data):
            verified_sols = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(chain_of_verification_pipeline, func.prompt,sol, llm_name,backend) for sol in func.generated_solutions]
                for f in as_completed(futures):
                    verified_sol, token_usage = f.result()
                    verified_sols.append(verified_sol)
                    total_token_usage += token_usage
                print('len(verified_sols)', len(verified_sols))
                func.verified_solutions = verified_sols
            print('total_token_usage: \n', total_token_usage)
        with open(out_path, "wb") as file:
            pickle.dump(data, file)

    print(f'Total tokens used: {total_token_usage}')

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process a specified dataset.')
    # Add dataset argument
    parser.add_argument('--dataset', type=str, required=True, choices=VALID_DATASETS, help=f'The dataset to process. Options: {VALID_DATASETS}')

    # Add LLM argument
    parser.add_argument('--llm', type=str, required=True, choices=VALID_LLMS, help=f'The LLM to use. Options are {VALID_LLMS}')
    parser.add_argument('--approach', type=str, required=True,choices=['CoVe', 'vanilla'])
    parser.add_argument('--backend', type=str, required=True, choices=backends)
    # Parse arguments
    args = parser.parse_args()

    llm_name = args.llm
    generate_solutions(args.dataset, llm_name, args.approach, args.backend)

if __name__ == '__main__':
    main()