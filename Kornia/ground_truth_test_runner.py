import os
import json
import subprocess
import shutil
import logging
from datetime import datetime
import sys
import ast
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from replacer_function import replace_function
# from replacer_class import replace_function_in_class

# Configure logging with thread safety
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"ground_truth_test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a thread-safe logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create console handler for multithreading info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Thread locks for synchronization
print_lock = threading.Lock()
file_write_lock = threading.Lock()

def process_single_task(task_id, dataset_entry, code_technique, test_technique, model_name, repo_name, 
                       generated_code, generated_tests, print_file_content=False):
    """
    Process a single task - designed to be thread-safe for parallel execution.
    
    Parameters:
    task_id (str): Task ID
    dataset_entry (dict): Dataset entry for this task
    code_technique (str): Code generation technique
    test_technique (str): Test generation technique  
    model_name (str): Model name
    repo_name (str): Repository name
    generated_code_dict (dict): Generated code dictionary
    generated_tests_dict (dict): Generated tests dictionary
    print_file_content (bool): Whether to print file content
    
    Returns:
    tuple: (task_id, result_dict) or (task_id, None) if failed
    """
    try:
        # Extract metadata from dataset entry
        function_name = dataset_entry.get("function", "")
        path = dataset_entry.get("ground Truth", "").split("#")[0]
        class_name = dataset_entry.get("class", None)
        repo = dataset_entry.get("repo", repo_name)
        
        # Check if we have generated code and tests for this task
        # if task_id not in generated_code_dict:
        #     print("###")
        #     print(generated_code_dict)
        #     logger.warning(f"No generated code found for task {task_id} in repo {repo}")
        #     return task_id, None
        
        # if task_id not in generated_tests_dict:
        #     logger.warning(f"No generated tests found for task {task_id} in repo {repo}")
        #     return task_id, None
        
        # Set BASE_PATH based on the actual repository for this specific task
        if repo != "pytorch3d":
            BASE_PATH = "/home/hamed/PycharmProjects/VALTEST/Kornia"
        else:
            BASE_PATH = "/local/data0/moved_data/"
        
        # Construct full path to ground truth file
        if repo != "pytorch3d":
            ground_truth_path = os.path.join(BASE_PATH, repo, path)
        else:
            ground_truth_path = os.path.join(BASE_PATH, repo, path)
            
        # Check if ground truth file exists
        # if not os.path.exists(ground_truth_path):
        #     logger.warning(f"Ground truth file not found: {ground_truth_path}")
        #     return task_id, None
        
        # Thread-safe printing
        with print_lock:
            print(f"\n{'='*60}")
            print(f"Processing task: {task_id}")
            print(f"Function: {function_name}")
            print(f"Class: {class_name if class_name else 'None (standalone function)'}")
            print(f"Repository: {repo}")
            print(f"{'='*60}")
        
        # Create test file with generated code + ground truth function + generated tests
        test_file_path = create_ground_truth_test_file(
            generated_code=generated_code,
            ground_truth_path=ground_truth_path,
            function_name=function_name,
            generated_tests=generated_tests,
            task_id=task_id,
            class_name=class_name,
            code_technique=code_technique
        )
        
        if not test_file_path:
            logger.error(f"Failed to create test file for task {task_id}")
            return task_id, None
        
        # Print file content if requested (thread-safe)
        if print_file_content:
            with print_lock:
                print(f"\n{'='*80}")
                print(f"GENERATED FILE CONTENT FOR TASK {task_id}")
                print(f"File: {test_file_path}")
                print(f"{'='*80}")
                try:
                    with open(test_file_path, 'r') as f:
                        file_content = f.read()
                    print(file_content)
                    print(f"{'='*80}")
                    print(f"END OF FILE CONTENT FOR TASK {task_id}")
                    print(f"{'='*80}")
                except Exception as e:
                    print(f"Error reading file content: {e}")
                    print(f"{'='*80}")
        
        # Clear pycache (thread-safe)
        clear_pycache()
        
        # Run pytest on the test file
        test_output, test_errors = run_pytest(test_file_path, rep=repo)
        
        # Thread-safe printing of test results
        with print_lock:
            print(f"\nTest run completed for {task_id}")
            print("="*50)
            print("PYTEST STDOUT:")
            print(test_output)
            print("="*50)
            print("PYTEST STDERR:")
            print(test_errors)
            print("="*50)
        
        # Parse individual test results
        individual_results = parse_individual_test_results(test_output)
        
        # Save results (thread-safe file writing)
        save_test_results(
            task_id=task_id,
            function_name=function_name,
            test_results=individual_results,
            technique=f"{code_technique}_{test_technique}",
            model_name=model_name,
            repo_name=repo_name
        )
        
        # Prepare result data
        result_data = {
            "function_name": function_name,
            "class_name": class_name,
            "test_file_path": test_file_path,
            "individual_results": individual_results,
            "total_tests": len(individual_results),
            "passed_tests": sum(individual_results.values()),
            "test_output": test_output,
            "test_errors": test_errors
        }
        
        # Print individual test results (thread-safe)
        with print_lock:
            if individual_results:
                total_tests = len(individual_results)
                passed_tests = sum(individual_results.values())
                print(f"Task {task_id}: {passed_tests}/{total_tests} tests passed")
                for test_name, result in individual_results.items():
                    print(f"  {test_name}: {'PASS' if result else 'FAIL'}")
            else:
                print(f"Task {task_id}: No tests found or error occurred")
        
        logger.info(f"Completed task {task_id}")
        return task_id, result_data
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        with print_lock:
            print(f"Error processing task {task_id}: {e}")
        return task_id, None

def clear_pycache():
    """Clear __pycache__ directory"""
    pycache_dir = "__pycache__"
    if os.path.exists(pycache_dir):
        shutil.rmtree(pycache_dir)
        logger.info(f"{pycache_dir} cleared.")

def run_pytest(test_file, python_path="/home/hamed/PycharmProjects/VALTEST/",
               test_case=None, conda_env="/home/aliredaq/anaconda3/envs/myenv/", is_conda=False, rep=None):
    """
    Run pytest using Docker instead of virtual environments for supported repositories.
    """
    if test_case:
        full_test_file = f"{test_file}::{test_case}"
    else:
        full_test_file = test_file
    
    # Check if this is a repository that has Docker support
    if rep == "kornia":
        # Use Docker for kornia
        # Mount the code directory to /local/data0/dockerfiles/kornia/kornia
        docker_mount_path = f"/home/hamed/PycharmProjects/VALTEST/Kornia/{rep}"
        
        # Get the relative path of the test file from the repository root
        repo_path = os.path.join(python_path, "Kornia", rep)
        test_file_relative = os.path.relpath(full_test_file, repo_path)
        print()
        # Docker command to run pytest
        command = f'docker run --rm -v {repo_path}:/app/{rep} {rep} bash -c "cd /app/{rep} && python -m pytest {test_file_relative} --color=no --cache-clear -v -s"'
        logger.info(f"Running Docker command: {command}")
    else:
        return
        # Fall back to original venv/conda logic for other repositories
        if rep == "DeepReg":
            is_conda = True
        python_path += rep + "/"
        if rep == "pytorch3d":
            python_path = "/local/data0/moved_data/"
        
        vi_env = "venv"
        if rep == "nlp-architecht":
            vi_env = "nvenv"

        if is_conda:
            conda_setup = "/home/aliredaq/anaconda3/etc/profile.d/conda.sh"
            command = f'source {conda_setup} && conda activate {rep.lower()} && PYTHONPATH={python_path} && cd {python_path}{rep} && python3 -m pytest {full_test_file} --color=no --cache-clear -v' 
        else:
            command = f'source {python_path}/{rep}/{vi_env}/bin/activate && PYTHONPATH={python_path}{rep} python -m pytest {full_test_file} --color=no --cache-clear -v -s' 
            logger.info(f"Running command: {command}")
    
    try:  
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=180)
        return result.stdout, result.stderr
    except Exception as e:
        logger.error(f"Error running pytest: {e}")
        return "error", str(e)

def parse_individual_test_results(test_output):
    """
    Parse pytest output to extract individual test case results.
    
    Parameters:
    test_output (str): The stdout from pytest execution
    
    Returns:
    dict: Dictionary mapping test case names to results (1 for pass, 0 for fail)
    """
    test_results = {}
    lines = test_output.split('\n')
    
    # First, look for the standard pytest format with PASSED/FAILED
    for i, line in enumerate(lines):
        line = line.strip()
        # Look for test result lines in pytest output
        # Format: test_file.py::test_function_name [anything] [PASSED/FAILED on this or next line]
        if '::' in line and ('test_' in line):
            # Extract the test path (part before any spaces)
            test_path = line.split(' ')[0] if ' ' in line else line
            
            if '::' in test_path:
                test_name = test_path.split('::')[-1]
                
                # Look for PASSED/FAILED on this line or the next few lines
                result_found = False
                for j in range(min(5, len(lines) - i)):  # Check current and next 4 lines
                    check_line = lines[i + j].strip()
                    if 'PASSED' in check_line:
                        test_results[test_name] = 1
                        result_found = True
                        break
                    elif 'FAILED' in check_line or 'ERROR' in check_line:
                        test_results[test_name] = 0
                        result_found = True
                        break
                
                # If no explicit PASSED/FAILED found, check for error patterns
                if not result_found:
                    # Look for error indicators in the next several lines
                    for j in range(min(10, len(lines) - i)):
                        check_line = lines[i + j].strip()
                        if any(error_pattern in check_line for error_pattern in [
                            'SyntaxError', 'ImportError', 'ModuleNotFoundError', 'AttributeError',
                            'TypeError', 'ValueError', 'NameError', 'IndentationError',
                            'Traceback', 'Exception', 'AssertionError'
                        ]):
                            test_results[test_name] = 0
                            result_found = True
                            break
                    
                    # If still no result found, mark as failed (safer assumption)
                    if not result_found:
                        test_results[test_name] = 0
                        logger.warning(f"No clear result found for test {test_name}, marking as failed")
    
    # Check for overall test collection/execution failures
    full_output = '\n'.join(lines).lower()
    if not test_results:
        # If no individual tests were parsed, check for collection failures
        if any(pattern in full_output for pattern in [
            'syntaxerror', 'importerror', 'modulenotfounderror', 'no tests collected',
            'error collecting', 'collection failed', 'traceback'
        ]):
            # Try to extract test names from the file and mark them as failed
            for line in lines:
                if 'def test_' in line:
                    test_name = line.split('def ')[1].split('(')[0]
                    test_results[test_name] = 0
        
        # If still no tests found, check for any test function definitions in the error output
        if not test_results:
            for line in lines:
                if '::test_' in line:
                    test_name = line.split('::test_')[1].split()[0]
                    test_results[test_name] = 0
    
    return test_results

def load_generated_code(code_technique, model_name, repo_name="dlbench"):
    """
    Load generated code from the code file.
    
    Parameters:
    code_technique (str): The mutation technique used for code generation
    model_name (str): Name of the model
    repo_name (str): Repository name
    
    Returns:
    dict: Dictionary mapping task_id to code
    """
    logger.info(f"Loading code from technique: {code_technique}")
    
    generation_dir = "/home/aliredaq/Desktop/Dual_execution/generation_output"
    print("$$$$$$$$$$$$$$$$$$$$")
    
    code_file = os.path.join(generation_dir, f"code_{code_technique}_{model_name}_{repo_name}_new2.jsonl")
    print(code_file)
    code_dict = {}
    if os.path.exists(code_file):
        with open(code_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id", "")
                    code = data.get("response_code", "")
                    if task_id and code and "429 RESOURCE_EXHAUSTED" not in code:
                        code_dict[task_id] = code
                except Exception as e:
                    logger.error(f"Error parsing code file line: {e}")
                    continue
    else:
        logger.warning(f"Code file not found: {code_file}")
    
    return code_dict

def load_generated_tests(test_technique, model_name, repo_name="dlbench"):
    """
    Load generated test cases from the test file.
    
    Parameters:
    test_technique (str): The mutation technique used for test generation
    model_name (str): Name of the model
    repo_name (str): Repository name
    
    Returns:
    dict: Dictionary mapping task_id to test cases
    """
    logger.info(f"Loading tests from technique: {test_technique}")
    
    generation_dir = "/home/aliredaq/Desktop/Dual_execution/generation_output"
    test_file = os.path.join(generation_dir, f"test_{test_technique}_{model_name}_{repo_name}_new2.jsonl")
    
    test_dict = {}
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id", "")
                    test_code = data.get("response_code", "")
                    if task_id and test_code and "429 RESOURCE_EXHAUSTED" not in test_code:
                        test_dict[task_id] = test_code
                except Exception as e:
                    logger.error(f"Error parsing test file line: {e}")
                    continue
    else:
        logger.warning(f"Test file not found: {test_file}")
    
    return test_dict

def extract_function_from_ground_truth(ground_truth_path, function_name, class_name=None):
    """
    Extract the specific function from ground truth file.
    
    Parameters:
    ground_truth_path (str): Path to the ground truth file
    function_name (str): Name of the function to extract
    class_name (str, optional): Name of the class if it's a class method
    
    Returns:
    str: The extracted function code
    """
    try:
        with open(ground_truth_path, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        if class_name:
            # Find the class first
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Now find the function within the class
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef) and class_node.name == function_name:
                            # Get function code with correct indentation
                            function_code = code.splitlines()[class_node.lineno-1:class_node.end_lineno]
                            return '\n'.join(function_code)
        else:
            # Find the standalone function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    function_code = code.splitlines()[node.lineno-1:node.end_lineno]
                    return '\n'.join(function_code)
        
        logger.error(f"Function {function_name} not found in {ground_truth_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting function: {e}")
        return None

def normalize_function_name_in_code(code, original_function_name, mutated_function_name):
    """
    Rename a function in code from mutated name back to original name.
    
    Parameters:
    code (str): The code containing the function to rename
    original_function_name (str): The original function name
    mutated_function_name (str): The current (mutated) function name
    
    Returns:
    str: Code with the function renamed
    """
    try:
        tree = ast.parse(code)
        
        class FunctionRenamer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == mutated_function_name:
                    logger.info(f"Renaming function from {mutated_function_name} to {original_function_name}")
                    node.name = original_function_name
                return self.generic_visit(node)
        
        renamer = FunctionRenamer()
        new_tree = renamer.visit(tree)
        return ast.unparse(new_tree)
        
    except Exception as e:
        logger.error(f"Error normalizing function name: {e}")
        return code

def direct_replace_function_in_code(code, function_name, ground_truth_function, class_name=None):
    """
    Directly replace a function in code using AST manipulation.
    
    Parameters:
    code (str): The code containing the function to replace
    function_name (str): Name of the function to replace
    ground_truth_function (str): The ground truth function code
    class_name (str, optional): Name of the class if it's a class method
    
    Returns:
    str: Code with the function replaced
    """
    try:
        # Parse the original code
        tree = ast.parse(code)
        
        # Parse the ground truth function
        gt_tree = ast.parse(ground_truth_function)
        gt_function = None
        
        # Extract the function definition from ground truth
        for node in ast.walk(gt_tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                gt_function = node
                break
        
        if not gt_function:
            logger.error(f"Could not find function {function_name} in ground truth code")
            return code
        
        # Replace the function in the main code
        class FunctionReplacer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # If this is the target function, replace it
                if node.name == function_name:
                    # If we're looking for a class method, check we're in the right class
                    if class_name is None:  # Standalone function
                        logger.info(f"Replacing standalone function {function_name}")
                        return gt_function
                    # For class methods, we handle this in visit_ClassDef
                return self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                if class_name and node.name == class_name:
                    # Replace method within this class
                    new_body = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == function_name:
                            logger.info(f"Replacing method {function_name} in class {class_name}")
                            new_body.append(gt_function)
                        else:
                            new_body.append(item)
                    node.body = new_body
                return self.generic_visit(node)
        
        # Apply the transformation
        replacer = FunctionReplacer()
        new_tree = replacer.visit(tree)
        
        # Convert back to code
        return ast.unparse(new_tree)
        
    except Exception as e:
        logger.error(f"Error in direct function replacement: {e}")
        return code

def create_ground_truth_test_file(generated_code, ground_truth_path, function_name, generated_tests, task_id, class_name=None, code_technique="original"):
    """
    Create a test file starting with the whole ground truth file and replacing only the specific function with generated implementation.
    
    Parameters:
    generated_code (str): Generated code from LLM
    ground_truth_path (str): Path to the ground truth file
    function_name (str): Name of the function being tested
    generated_tests (str): Generated test cases
    task_id (str): Task ID for naming
    class_name (str, optional): Name of the class if it's a class method
    code_technique (str): Code generation technique used
    
    Returns:
    str: Path to the created test file
    """
    try:
        # Extract the actual function name from the function_name field
        patterns = [
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', # Match "def function_name"
            r'`([a-zA-Z_][a-zA-Z0-9_]*)`',     # Match function name in backticks
            r"'([a-zA-Z_][a-zA-Z0-9_]*)'",     # Match function name in single quotes
            r'"([a-zA-Z_][a-zA-Z0-9_]*)"',     # Match function name in double quotes
        ]
        
        actual_function_name = None
        
        # First try the specific patterns
        for pattern in patterns:
            match = re.search(pattern, function_name)
            if match:
                actual_function_name = match.group(1)
                logger.info(f"Extracted actual function name using pattern '{pattern}': {actual_function_name}")
                break
        
        # If no pattern matches, check if it's already a clean function name
        if not actual_function_name:
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function_name.strip()):
                actual_function_name = function_name.strip()
                logger.info(f"Using function_name as-is: {actual_function_name}")
            else:
                # Clean the string by removing non-alphanumeric characters except underscore
                cleaned = re.sub(r'[^a-zA-Z0-9_]', '', function_name.strip())
                if cleaned and (cleaned[0].isalpha() or cleaned[0] == '_'):
                    actual_function_name = cleaned
                    logger.warning(f"Cleaned function name from '{function_name}' to '{actual_function_name}'")
                else:
                    logger.error(f"Could not extract valid function name from '{function_name}'")
                    return None
        
        # Create the test file name
        test_file_name = f"test_this_{actual_function_name}.py"
        test_file_path = os.path.join(os.path.dirname(ground_truth_path), test_file_name)
        
        # Start with the whole ground truth file
        with open(ground_truth_path, 'r') as f:
            ground_truth_code = f.read()
        
        # Extract the generated function from the generated code
        generated_function = None
        generated_function_name = None
        
        # try:
        #     tree = ast.parse(generated_code)
        #     for node in ast.walk(tree):
        #         if isinstance(node, ast.FunctionDef):
        #             # Find the function that we want to replace
        #             if node.name == actual_function_name:
        #                 generated_function = ast.unparse(node)
        #                 generated_function_name = node.name
        #                 break
        #             else:
        #                 # For mutation techniques, the function name might be different
        #                 # Try to find any function that could be the target
        #                 generated_function = ast.unparse(node)
        #                 generated_function_name = node.name
        #                 logger.info(f"Found generated function: {generated_function_name}")
        #                 break
        # except Exception as e:
        #     logger.error(f"Error parsing generated code: {e}")
        #     logger.warning(f"Skipping task {task_id} due to malformed generated code")
        #     return None
        
        # if not generated_function:
        #     logger.error(f"Could not find any function in generated code for {task_id}")
        #     return None
        
        # If the function names are different (mutation technique), normalize the generated function name
        # if generated_function_name != actual_function_name:
        #     logger.info(f"Normalizing function name from {generated_function_name} to {actual_function_name}")
        #     generated_function = normalize_function_name_in_code(generated_function, actual_function_name, generated_function_name)
        
        # Replace the function in the ground truth code with the generated implementation
        # modified_code = direct_replace_function_in_code(ground_truth_code, actual_function_name, generated_function, class_name)
        modified_code = ground_truth_code
        # Process the test cases to remove 'from' imports and fix function calls
        processed_tests = process_test_cases(generated_tests, actual_function_name)
        
        # Create the combined content
        combined_content = f"""{modified_code}

# Generated Test Cases for {actual_function_name}
# Task ID: {task_id}
{processed_tests}
"""
        print(combined_content)
        # Write the final combined content
        with open(test_file_path, 'w') as f:
            f.write(combined_content)
        
        logger.info(f"Created test file: {test_file_path}")
        return test_file_path
        
    except Exception as e:
        logger.error(f"Error creating test file for {task_id}: {e}")
        return None

def process_test_cases(test_code, function_name):
    """
    Process test cases to remove problematic imports while preserving necessary ones like MagicMock.
    
    Parameters:
    test_code (str): The generated test code
    function_name (str): The original function name
    
    Returns:
    str: Processed test code
    """
    import ast
    import re
    
    if not test_code:
        return ""
    
    try:
        # Extract the actual function name from the function_name field
        patterns = [
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', # Match "def function_name"
            r'`([a-zA-Z_][a-zA-Z0-9_]*)`',     # Match function name in backticks
            r"'([a-zA-Z_][a-zA-Z0-9_]*)'",     # Match function name in single quotes
            r'"([a-zA-Z_][a-zA-Z0-9_]*)"',     # Match function name in double quotes
        ]
        
        original_name = None
        
        # First try the specific patterns
        for pattern in patterns:
            match = re.search(pattern, function_name)
            if match:
                original_name = match.group(1)
                logger.info(f"Extracted original function name using pattern '{pattern}': {original_name}")
                break
        
        # If no pattern matches, check if it's already a clean function name
        if not original_name:
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function_name.strip()):
                original_name = function_name.strip()
                logger.info(f"Using function_name as-is: {original_name}")
            else:
                # Clean the string by removing non-alphanumeric characters except underscore
                cleaned = re.sub(r'[^a-zA-Z0-9_]', '', function_name.strip())
                if cleaned and (cleaned[0].isalpha() or cleaned[0] == '_'):
                    original_name = cleaned
                    logger.warning(f"Cleaned function name from '{function_name}' to '{original_name}'")
                else:
                    logger.error(f"Could not extract valid function name from '{function_name}'")
                    return test_code
        
        # Parse the test code to process imports properly
        try:
            tree = ast.parse(test_code)
            filtered_lines = []
            needs_magicmock = False
            needs_sys = False
            
            # First pass: check if MagicMock or sys is used in the code
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == 'MagicMock':
                    needs_magicmock = True
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'sys':
                    needs_sys = True
            
            # Add missing imports at the top if needed
            if needs_magicmock:
                filtered_lines.append("from unittest.mock import MagicMock")
            if needs_sys:
                filtered_lines.append("import sys")
            
            for node in tree.body:
                should_include = True
                
                # Remove problematic "from X import Y" statements
                if isinstance(node, ast.ImportFrom):
                    # Skip imports that try to import the function being tested or from invalid modules
                    if (node.module and 
                        ('your_module' in str(node.module) or 
                         'your_package' in str(node.module) or
                         node.module in ['your_module', 'your_package'] or
                         any(alias.name == original_name for alias in (node.names or [])))):
                        should_include = False
                        logger.info(f"Removing problematic import: from {node.module} import {[alias.name for alias in node.names]}")
                    # Also check if any import alias contains the target function name  
                    elif (node.names and any(alias.name == original_name for alias in node.names)):
                        should_include = False
                        logger.info(f"Removing import of target function: from {node.module} import {[alias.name for alias in node.names]}")
                    # Keep important imports like MagicMock, but avoid duplicates
                    elif (node.module == 'unittest.mock' and needs_magicmock):
                        should_include = False  # We already added it above
                        logger.info(f"Skipping duplicate MagicMock import")
                
                # Include all other statements (regular imports, code, etc.)
                if should_include:
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        filtered_lines.append("\n".join(test_code.splitlines()[node.lineno - 1:node.end_lineno]))
            
            processed_code = "\n\n".join(filtered_lines)
            
            # Post-process to remove any remaining problematic import lines
            lines = processed_code.split('\n')
            clean_lines = []
            for line in lines:
                # Skip lines that import the target function from invalid modules
                if ('from your_module import' in line or 
                    'from your_package import' in line or
                    f'import {original_name}' in line):
                    logger.info(f"Removing problematic line: {line.strip()}")
                    continue
                clean_lines.append(line)
            
            return '\n'.join(clean_lines)
            
        except SyntaxError:
            logger.warning(f"Could not parse test code as Python, returning original")
            return test_code
            
    except Exception as e:
        logger.error(f"Error processing test cases: {e}")
        return test_code

def save_test_results(task_id, function_name, test_results, technique, model_name, repo_name):
    """
    Save test results to JSONL file.
    
    Parameters:
    task_id (str): Task ID
    function_name (str): Function name
    test_results (dict): Individual test results
    technique (str): Technique used
    model_name (str): Model name
    repo_name (str): Repository name
    """
    jsonl_file = os.path.join(os.getcwd(),"cross_test" ,model_name,f"ground_truth_test_results_{technique}_{model_name}_{repo_name}.jsonl")
    os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
    # Create JSON record for this task
    json_record = {
        "task_id": task_id,
        "function_name": function_name,
        "technique": technique,
        "model": model_name,
        "repository": repo_name,
        "individual_test_results": test_results,
        "total_tests": len(test_results),
        "passed_tests": sum(test_results.values()),
        "failed_tests": len(test_results) - sum(test_results.values()),
        "pass_rate": (sum(test_results.values()) / len(test_results)) * 100 if test_results else 0
    }
    
    # Append to JSONL file (thread-safe)
    with file_write_lock:
        with open(jsonl_file, 'a') as f:
            json.dump(json_record, f)
            f.write('\n')
    
    logger.info(f"Results saved to JSONL: {jsonl_file}")

def save_final_summary_log(all_results, technique, model_name, repo_name):
    """
    Save final summary log in JSONL format.
    
    Parameters:
    all_results (dict): Results from all processed tasks
    technique (str): Technique used
    model_name (str): Model name
    repo_name (str): Repository name
    """
    # Calculate totals
    total_tasks = len(all_results)
    total_individual_tests = sum(result["total_tests"] for result in all_results.values())
    total_passed_tests = sum(result["passed_tests"] for result in all_results.values())
    total_failed_tests = total_individual_tests - total_passed_tests
    
    # Create summary record
    summary_record = {
        "timestamp": datetime.now().isoformat(),
        "technique": technique,
        "model": model_name,
        "repository": repo_name,
        "summary": {
            "total_tasks_processed": total_tasks,
            "total_individual_tests": total_individual_tests,
            "total_passed_tests": total_passed_tests,
            "total_failed_tests": total_failed_tests,
            "overall_pass_rate": (total_passed_tests/total_individual_tests)*100 if total_individual_tests > 0 else 0,
            "tasks_with_all_tests_passed": sum(1 for result in all_results.values() if result["total_tests"] > 0 and result["passed_tests"] == result["total_tests"]),
            "tasks_with_some_tests_failed": sum(1 for result in all_results.values() if result["total_tests"] > 0 and result["passed_tests"] < result["total_tests"]),
            "tasks_with_no_tests": sum(1 for result in all_results.values() if result["total_tests"] == 0)
        },
        "detailed_results": [
            {
                "task_id": task_id,
                "function_name": result["function_name"],
                "total_tests": result["total_tests"],
                "passed_tests": result["passed_tests"],
                "failed_tests": result["total_tests"] - result["passed_tests"],
                "pass_rate": (result["passed_tests"] / result["total_tests"]) * 100 if result["total_tests"] > 0 else 0,
                "individual_results": result["individual_results"]
            }
            for task_id, result in all_results.items()
        ]
    }
    
    # Save to summary file
    summary_file = os.path.join(os.getcwd(), "cross_test", model_name, f"ground_truth_final_summary_{technique}_{model_name}_{repo_name}.jsonl")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with file_write_lock:
        with open(summary_file, 'w') as f:
            json.dump(summary_record, f, indent=2)
            f.write('\n')
    
    logger.info(f"Final summary saved to: {summary_file}")
    
    # Print summary in the format requested (e.g., "4 passed 2 failed")
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Technique: {technique}")
    print(f"Model: {model_name}")
    print(f"Repository: {repo_name}")
    print(f"Total Tasks: {total_tasks}")
    print(f"Total Tests: {total_individual_tests}")
    print(f"Result: {total_passed_tests} passed, {total_failed_tests} failed")
    print(f"Pass Rate: {(total_passed_tests/total_individual_tests)*100:.2f}%" if total_individual_tests > 0 else "No tests to calculate rate")
    print(f"{'='*60}")

def run_ground_truth_tests(dataset_file_path, code_technique, test_technique, model_name="o4-mini", repo_name="dlbench", task_ids=None, print_file_content=False, single_id=None, num_threads=1):
    """
    Run tests with generated code where specific function is replaced with ground truth implementation.
    
    Parameters:
    dataset_file_path (str): Path to the dataset file
    code_technique (str): Code generation technique
    test_technique (str): Test generation technique
    model_name (str): Model name
    repo_name (str): Repository name
    task_ids (list): Specific task IDs to process (optional)
    print_file_content (bool): Whether to print the generated file content
    single_id (str): Single task ID to process (if specified, only this ID will be processed)
    num_threads (int): Number of threads to use for parallel processing (default: 1)
    
    Returns:
    dict: Results for all processed tasks
    """
    logger.info(f"Running ground truth tests with code technique: {code_technique}, test technique: {test_technique}")
    logger.info(f"Using {num_threads} thread(s) for processing")
    
    # Load dataset
    dataset_entries = {}
    with open(dataset_file_path, "r") as df:
        for line in df:
            try:
                entry = json.loads(line)
                dataset_entries[entry["task_id"]] = entry
            except Exception as e:
                logger.error(f"Error parsing dataset line: {e}")
                continue
    
    # Load generated code and tests once (shared across all tasks)
    generated_code_dict = load_generated_code(code_technique, model_name, repo_name)
    generated_tests_dict = load_generated_tests(test_technique, model_name, repo_name)
    
    # Filter tasks to process
    tasks_to_process = []
    for task_id, dataset_entry in dataset_entries.items():
        # If single_id is specified, only process that specific ID
        if single_id and task_id != single_id:
            continue
        
        # Skip if not in task_ids (when filtering is applied)
        if task_ids and task_id not in task_ids:
            continue
            
        tasks_to_process.append((task_id, dataset_entry))
    
    print(f"Found {len(tasks_to_process)} tasks to process")
    logger.info(f"Found {len(tasks_to_process)} tasks to process")
    
    all_results = {}
    
    if num_threads == 1:
        # Single-threaded execution (original behavior)
        logger.info("Running in single-threaded mode")
        for task_id, dataset_entry in tasks_to_process:
            result = process_single_task(
                task_id, dataset_entry, code_technique, test_technique, 
                model_name, repo_name, generated_code_dict, generated_tests_dict, 
                print_file_content
            )
            if result[1] is not None:
                all_results[result[0]] = result[1]
    else:
        # Multi-threaded execution
        logger.info(f"Running in multi-threaded mode with {num_threads} threads")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks to the thread pool
            future_to_task = {
                executor.submit(
                    process_single_task,
                    task_id, dataset_entry, code_technique, test_technique,
                    model_name, repo_name, generated_code_dict, generated_tests_dict,
                    print_file_content
                ): task_id for task_id, dataset_entry in tasks_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    if result[1] is not None:
                        all_results[result[0]] = result[1]
                except Exception as e:
                    logger.error(f"Error in thread processing task {task_id}: {e}")
                    with print_lock:
                        print(f"Error in thread processing task {task_id}: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("GROUND TRUTH TESTING SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        total_tasks = len(all_results)
        total_individual_tests = sum(result["total_tests"] for result in all_results.values())
        total_passed_tests = sum(result["passed_tests"] for result in all_results.values())
        
        print(f"Total tasks processed: {total_tasks}")
        print(f"Total individual tests: {total_individual_tests}")
        print(f"Total passed tests: {total_passed_tests}")
        print(f"Total failed tests: {total_individual_tests - total_passed_tests}")
        print(f"Overall pass rate: {(total_passed_tests/total_individual_tests)*100:.2f}%" if total_individual_tests > 0 else "No tests run")
        
        # Print detailed breakdown
        print(f"\nDetailed breakdown:")
        for task_id, result in all_results.items():
            if result["individual_results"]:
                task_total = result["total_tests"]
                task_passed = result["passed_tests"]
                print(f"  {task_id}: {task_passed}/{task_total} tests passed")
            else:
                print(f"  {task_id}: No tests found")
        
        # Save final summary log in JSONL format
        technique = f"{code_technique}_{test_technique}"
        save_final_summary_log(all_results, technique, model_name, repo_name)
    
    return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ground truth tests with generated code and test cases for DLBench.')
    parser.add_argument('--repo', type=str, default='dlbench', help='Repository name (default: dlbench)')
    parser.add_argument('--ids', type=str, nargs='*', help='Specific task IDs to process (optional)')
    parser.add_argument('--model', type=str, default='o4-mini', help='Model name (default: o4-mini)')
    # parser.add_argument('--code-technique', type=str, default='original', 
    #                     help='Code technique name (default: original)')
    # parser.add_argument('--test-technique', type=str, default='original', 
    #                     help='Test technique name (default: original)')
    parser.add_argument('--print-content', action='store_true',
                        help='Print the content of generated test files')
    parser.add_argument('--single-id', type=str, default=None,
                        help='Process only a single task ID (takes priority over --ids)')
    parser.add_argument('--all', action='store_true',
                        help='Process all tasks in the dataset (ignores --ids and --single-id)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to use for parallel processing (default: 1)')
    
    args = parser.parse_args()
    
    # Configuration
    repo_name = args.repo.lower()
    dataset_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "dataset", "DLBench", "dlbench.jsonl")
    # current_directory = os.getcwd()
    print(f'Reading dataset from {dataset_file}')
    # dataset_file = "/home/aliredaq/Desktop/Dual_execution/dataset/DLBench/dlbench.jsonl"
    
    # Determine which tasks to process
    task_ids = None
    single_id = None
    # code_techniques = ["original", "active_to_passive" "declarative_to_interrogative", "verb_to_similar_verb", "rephrase_prompt"
    # , "task_function_name"]
    code_techniques = ["v1", "original", ]
    test_techniques = ["original", "v1"]
    # test_techniques = ["lowercase_to_uppercase"]
    # test_techniques = ["task_function_name", "adversarial_function_name"]
    if args.all:
        # Process all tasks in dataset
        task_ids = None
        single_id = None
        print("Processing ALL tasks in the dataset")
    elif args.single_id:
        # Process only the specified single ID
        single_id = args.single_id
        task_ids = None
        print(f"Processing SINGLE task: {single_id}")
    elif args.ids:
        # Process specified list of IDs
        task_ids = args.ids
        single_id = None
        print(f"Processing SPECIFIC tasks: {task_ids}")
    else:
        # Default behavior - process all tasks
        task_ids = None
        single_id = None
        print("Processing ALL tasks in the dataset (default behavior)")
    
    # Run ground truth tests
    for test_technique in test_techniques:
        for code_technique in code_techniques:
            results = run_ground_truth_tests(
                dataset_file_path=dataset_file,
                code_technique=code_technique,
                test_technique=test_technique,
                model_name=args.model,
                repo_name=repo_name,
                task_ids=task_ids,
                print_file_content=args.print_content,
                single_id=single_id,
                num_threads=args.threads
            )
    
            # return results

if __name__ == "__main__":
    main()