import re
import textwrap

def split_unittest_code(code: str) -> list[str]:
    try:
        # Step 1: Remove `from your_module import ...` lines
        code = re.sub(r"from\s+your_module\s+import\s+.*\n", "", code)

        # Step 2: Extract setUp method
        setup_match = re.search(r"def\s+setUp\(self\):[\s\S]*?(?=def\s+test_)", code)
        if not setup_match:
            raise ValueError("No setUp method found")
        setup_method = textwrap.dedent(setup_match.group(0))

        # Step 3: Extract all test methods
        test_methods = re.findall(r"(def\s+test_[\s\S]*?)(?=def\s+test_|class|$)", code)

        if not test_methods:
            raise ValueError("No test methods found")

        # Step 4: Split test methods into chunks of 2
        chunks = [test_methods[i:i+2] for i in range(0, len(test_methods), 2)]

        test_cases = []
        for idx, chunk in enumerate(chunks, start=1):
            class_name = f"TestCheapestConnectionPart{idx}"
            test_case = [
                "import unittest",
                "",
                f"class {class_name}(unittest.TestCase):",
                "",
                textwrap.indent(setup_method, "    ")
            ]
            for method in chunk:
                test_case.append(textwrap.indent(textwrap.dedent(method), "    "))
            test_cases.append("\n".join(test_case))

        return test_cases

    except Exception as e:
        # If anything goes wrong, return the original code in a list
        return [code]