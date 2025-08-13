import json
import copy
import tqdm
import concurrent.futures as cfuts
import random
import argparse
from src import model, utils
import os

def form_messages(problems, is_test_case=False):
    """
    Create a list of message pairs for the model
    
    Args:
        problems (list): List of (prompt, task_id) tuples
        is_test_case (bool): Whether to generate test cases or code
        
    Returns:
        list: List of (message, task_id) tuples
    """
    system_prompt_code = """Generate the python code solution based on the context. 
Return the function with import packages if needed. 
Do not include any test cases or usage. 
ONLY RETURN FINAL Code in ```python
<code>
``` 
DO NOT PROVIDE ANY THINKING OR EXPLANATION.
"""

    system_prompt_test = """Generate 5 test cases for the following problem. The test cases should be in pytest format.
Return the test cases in ```python\n <code> ``` format. You should only return the test cases, not the function itself. You should return runnable code without placeholders.
...
```"""

    return [
        (
            [
                {'role': 'system', 'content': system_prompt_code + "ONLY RETURN FINAL CODE IN ```python\n <code> ```. DO NOT ADD ANYOTHER CONTEXT" if not is_test_case else system_prompt_test},
                {'role': 'user', 'content': problem[0]}
            ],
            problem[1]  # task_id
        )
        for problem in problems
    ]

def run_model(message, model_name):
    """
    Run the model with the given message
    
    Args:
        message (list): The message to send to the model
        model_name (str): The name of the model to use
        
    Returns:
        tuple: (response, prompt_tokens, completion_tokens)
    """
    if model_name.startswith('deepseek'):
        return model.call_deepseek(message)
    
    if model_name.startswith("gemini"):
        return model.call_gemini(message)
    return model.call_chat_gpt(message, model=model_name)

def run(messages, output_path, model_name, is_test_case=False):
    """
    Run the model on a list of messages and save the results
    
    Args:
        messages (list): List of (message, task_id) tuples
        output_path (str): Path to save the results
        model_name (str): Name of the model to use
        is_test_case (bool): Whether to generate test cases
        
    Returns:
        None
    """
    def process_message(message, task_id):
        """Process a single message and return the result"""
        try:
            response, prompt_tokens, completion_tokens = run_model(message, model_name)
            print(f"Task {task_id} generated response of length {len(response)}")
            
            # Extract code from response
            code = utils.process_generation_to_code(response)
            
            return {
                'task_id': task_id,
                'prompt': message,
                'response': response,
                'response_code': code
            }
        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            return {
                'task_id': task_id,
                'prompt': message,
                'response': f"Error: {str(e)}",
                'response_code': ""
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with cfuts.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_message, message[0], message[1]) 
            for message in messages
        ]
        
        responses = []
        for future in tqdm.tqdm(cfuts.as_completed(futures), total=len(futures)):
            responses.append(future.result())

    # Sort responses by task_id
    try:
        # Try to sort by the numerical part of the task_id
        responses.sort(key=lambda x: int(x['task_id'].split('/')[-1]))
    except:
        # Fall back to string sorting
        responses.sort(key=lambda x: x['task_id'])

    # Write results to file
    with open(output_path, 'w') as f:
        for res in responses:
            f.write(json.dumps(res) + '\n')

def get_dlbench_template():
    """
    Create a simple template for DLBench prompts
    
    Returns:
        str: Template string
    """
    return "{problem}\n\nWrite a Python function named `{function_name}` that solves this task."

def generate_prompt(mutation, function_name, template):
    """
    Generate a prompt from a mutation and function name
    
    Args:
        mutation (str): The mutated problem description
        function_name (str): The function name to use
        template (str): The template to use
        
    Returns:
        str: Formatted prompt
    """
    return mutation

def generate_dlbench_code(args):
    """
    Generate code for DLBench dataset
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        None
    """
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Define file paths
    type_str = 'code' if not args.test_case else 'test'
    dataset = 'dlbench'
    
    # Load original data from mutation_output
    original_file = f'mutation_output/original_{args.mutation_model}_{dataset}.jsonl'
    try:
        with open(original_file) as f:
            original_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Original file {original_file} not found.")
        return
    
    # Get template
    template = get_dlbench_template()
    
    # Create problems list from original data
    problems = []
    for data in original_data:
        task_id = data['task_id']
        # Use the original problem, not mutation
        problem = data['original']
        function_name = data.get('original_function_name', 'func')
        
        # Format the prompt
        formatted_prompt = generate_prompt(problem, function_name, template)
        problems.append((formatted_prompt, task_id))
    
    # If a limit is set, only use that many problems
    if args.limit > 0:
        problems = problems[:args.limit]
    
    def run_version_and_type(version, is_test_case):
        """Run a single version of the generation for either code or test"""
        type_str = 'test' if is_test_case else 'code'
        output_path = f'generation_output/dlbench/{type_str}_original_{args.model}_v{version}.jsonl'
        
        print(f"Starting version {version} ({type_str}): Generating {'test cases' if is_test_case else 'code'} for {len(problems)} DLBench problems")
        print(f"Using model: {args.model}")
        print(f"Output will be saved to: {output_path}")
        
        # Create messages and run the model
        messages = form_messages(problems, is_test_case=is_test_case)
        run(messages, output_path, args.model, is_test_case=is_test_case)
        
        print(f"Version {version} ({type_str}) complete! Results saved to {output_path}")
        return f"v{version}_{type_str}"
    
    # Run 7 versions in parallel for both code and test
    print(f"Starting parallel generation of 7 versions for both code and test cases...")
    with cfuts.ThreadPoolExecutor(max_workers=min(14, args.max_workers)) as executor:
        futures = []
        
        # Submit jobs for both code and test for each version
        for version in range(1, 3):
            futures.append(executor.submit(run_version_and_type, version, False))  # Code
            futures.append(executor.submit(run_version_and_type, version, True))   # Test
        
        completed_jobs = []
        for future in cfuts.as_completed(futures):
            completed_job = future.result()
            completed_jobs.append(completed_job)
            print(f"✓ {completed_job} completed ({len(completed_jobs)}/14)")
    
    print(f"All 7 versions for both code and test completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate code or test cases for DLBench dataset (7 versions of original data)')
    parser.add_argument('--model', type=str, default='gemini', help='Model to use for generation')
    parser.add_argument('--mutation_model', type=str, default='gpt-4o', help='Model used for mutations (to locate original file)')
    parser.add_argument('--test_case', action='store_true', help='Generate test cases instead of code')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of problems to process (0 for all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_workers', type=int, default=6, help='Maximum number of concurrent workers')
    
    args = parser.parse_args()
    generate_dlbench_code(args)