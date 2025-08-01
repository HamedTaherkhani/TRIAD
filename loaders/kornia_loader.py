import json
import os

class kornia_loader:
    def __init__(self):
        import json
        data = []
        with open("Kornia/dlbench.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d['repo'] == 'kornia':
                    data.append(d)
        self.prompts = []
        self.tests = []
        self.sols = []
        self.ids = []
        for item in data:
            self.prompts.append(item['prompt'])
            self.tests.append(item['test_cases'])
            self.ids.append(item['task_id'])
            # relative_path = item['ground Truth']
            # current_path = os.getcwd()
            # absolute_path = os.path.join(current_path, relative_path)
            # with open(absolute_path, 'r') as file:
            #     content = file.read()
            self.sols.append("")
    def get_prompts(self):
        return self.prompts
    def get_tests(self):
        return self.tests
    def get_sols(self):
        return self.sols
    def get_ids(self):
        return self.ids
