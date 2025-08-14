from typing import List
class TestCase:
    def __init__(self, text, is_valid: int = None, prediction_is_valid: int = None, prediction_y_prob: int = None, validated_text=None):
        self.text = text
        self.is_valid = is_valid
        self.prediction_is_valid = prediction_is_valid
        self.prediction_y_prob = prediction_y_prob
        self.validated_text = validated_text

    def __str__(self):
        return f'Text: "{self.text}"\nIs Valid: {self.is_valid}\n'

    def __repr__(self):
        return self.__str__()


class Function:
    def __init__(self, prompt: str, generated_testcases: list[TestCase], solution: str, dataset:str, task_id: str,original_tests: list[str], generated_solutions: List[str]=None, verified_solutions: list[str] = None):
        self.prompt = prompt
        self.generated_testcases = generated_testcases
        self.solution = solution
        self.original_tests = original_tests
        self.generated_solutions = generated_solutions
        self.task_id = task_id
        self.dataset = dataset
        self.verified_solutions = verified_solutions

    def __str__(self):
        # Create a string representation of test cases
        generated_testcases_str = "\n".join([str(tc) for tc in self.generated_testcases])
        generated_solutions_str = "\n".join(self.generated_solutions)
        return (
            f"Prompt:\n{self.prompt}\n"
            f"Solution:\n{self.solution}\n"
            f"Generated Test Cases:\n{generated_testcases_str}"
            f"Generated Solutions:\n{generated_solutions_str}"
        )

    def __repr__(self):
        return self.__str__()