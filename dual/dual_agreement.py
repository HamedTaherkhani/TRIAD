import pickle
import sys
import traceback
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Any
from reusable_classes import Function
from function_executor import run_test_cases
from generate_solutions import IMPORT_HEADER
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from function_executor_codet import check_correctness_with_test_cases
from tqdm import tqdm
import math
from loaders import HumanEvalLoader, MBPPLoader, BigCodeLoader, LBPPLoaderPython, LBPPLoaderJava, kornia_loader
from evaluation import compute_validity_rate


# -------------------------------------------------------------------------------
# Main processing: load, select best, run both original & generated tests, update, and compute metrics
# -------------------------------------------------------------------------------
def perform_dual_agreement(
    tests_path: str, code_path: str, timeout, dataset_name
) -> None:
    with open(tests_path, "rb") as f:
        generated_tests: List[Function] = pickle.load(f)

    with open(code_path, "rb") as f:
        generated_sols: List[Function] = pickle.load(f)
    generated_tests = generated_tests[:10]
    generated_sols = generated_sols[:10]

    print(f'tests_path len: {len(generated_tests)}')
    print(f'code_path len: {len(generated_sols)}')
    import logging

    from postprocess import PostProcessor
    from execution import evaluate_with_test_code, evaluate_with_test_cases
    from io_utils import Tools
    from agreement import DataManager, DualAgreement
    from evaluation import pass_at_K, get_result_of_sorted_solutions

    logging.basicConfig(
        format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)

    handled_solutions = PostProcessor.map_task_id_for_solution(generated_sols, dataset_name)
    handled_test_cases = PostProcessor.map_task_id_for_test_case(generated_tests, dataset_name)

    ground_truth_exec_result = evaluate_with_test_code(samples=handled_solutions, timeout=timeout, dataset_name=dataset_name)
    # ground_truth_exec_result = [00]
    count = 0
    for aa in ground_truth_exec_result:
        if aa['passed']:
            count += 1
    print(f'total count ground truth: {len(ground_truth_exec_result)}')
    print(f'total pass ground truth: {count}')
    dual_exec_result = evaluate_with_test_cases(solutions=handled_solutions, test_cases_dict=handled_test_cases, timeout=timeout, dataset_name=dataset_name)
    # print(ground_truth_exec_result[0])
    print('dual result')
    print(f'len dual result: {len(dual_exec_result)}')
    print(dual_exec_result[0])
    count_duel = 0
    for aa in dual_exec_result:
        for ii in aa['passed']:
            if ii:
                count_duel += 1
    print(f'total correct dual result: {count_duel}')
    # Tools.dump_pickle(os.path.join(args.cache_dir, 'ground_truth_exec_result.pkl'), ground_truth_exec_result)
    # Tools.dump_pickle(os.path.join(args.cache_dir, 'dual_exec_result.pkl'), dual_exec_result)
    print('*'*100)
    print(len(dual_exec_result))
    print(len(handled_solutions))
    print(len(handled_test_cases))
    print('*'*100)
    data_manager = DataManager(dual_exec_result, handled_solutions, handled_test_cases)
    set_consistency = DualAgreement(data_manager)
    ranked_result,passed_solution_test_case_pairs_by_task = set_consistency.get_sorted_solutions_without_iter()
    if dataset_name == 'BigCodeBenchHard':
        is_unittest=True
    else:
        is_unittest=False
    compute_validity_rate(ranked_result=ranked_result, ground_truth_exec_result=ground_truth_exec_result, functions=generated_tests,is_unittest=is_unittest)
    # print(ranked_result)
    # logger.info('pass rates of ranked solutions')
    # get_result_of_sorted_solutions(ground_truth_exec_result, ranked_result)
    # logger.info('pass rates of random solutions')
    # pass_at_K(ground_truth_exec_result)

# -------------------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the script with specified dataset and LLM.")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        required=True,
        help=f"Specify the dataset to use. Choices are: {VALID_DATASETS}."
    )

    # Add the 'LLM' argument with restricted choices, allowing future extensions
    parser.add_argument(
        "--llm",
        type=str,
        choices=VALID_LLMS,
        required=True,
        help=f"Specify the LLM to use. Choices are: {VALID_LLMS}."
    )
    parser.add_argument(
        "--test_approach",
        type=str,
        required=True,
        choices=['self-consistency', 'holistic'],
    )
    parser.add_argument(
        '--code_approach',
        type=str,
        required=True,
        choices=['self-consistency', 'vanilla', 'CoVe'],
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        required=False,
    )

    args = parser.parse_args()
    tests_path = f'generated_tests/final_tests/{args.test_approach}/{args.dataset}-{args.llm}.pkl'
    code_path = f'generated_solutions/{args.code_approach}/{args.dataset}-{args.llm}.pkl'
    file_name = f'output/dual_agreement/{args.dataset}-{args.llm}_{args.test_approach}_{args.code_approach}.txt'
    print(f'Writing the output to {file_name}')
    with open(file_name, 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        real_dataset = []

        # process_functions(input_path, output_pickle, valtest_scores_dir, args.approach)
        perform_dual_agreement(tests_path=tests_path, code_path=code_path, timeout=args.timeout, dataset_name=args.dataset)