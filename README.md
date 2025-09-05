# Consistency Meets Verification: Enhancing Test Generation Quality in Large Language Models Without Ground-Truth Solutions

## 🚀 Features

- **Multiple Test Generation Approaches**:
  - Self-consistency: Uses multiple LLM completions and selects the most frequent result
  - Two-stage: Generates test stubs first, then enriches them with assertions
  - Holistic: Generates complete test suites in a single pass

- **Code Verification Methods**:
  - Vanilla: Direct LLM code generation
  - CoVe (Chain of Verification): Multi-step verification pipeline with question generation and correction

- **Comprehensive Evaluation**:
  - Test coverage analysis (line and branch coverage)
  - Mutation testing with multiple operators
  - Dual agreement evaluation between tests and code solutions

- **Multiple Dataset Support**:
  - BigCodeBenchHard
  - LBPPPython

- **LLM Backend Support**:
  - OpenAI API
  - Fireworks API
  - VLLM (local inference)

## 📋 Prerequisites

- Python 3.8+
- CUDA (for VLLM backend)
- Required environment variables (see Configuration section)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd TRIAD
```

2. **Create and activate virtual environment**:
```bash
# using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
```bash
# For running bigcodebench you need a separate env
python3.10 env .bigcode_venv
source .bigcode_venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_bigcode.txt

```
3.**Set up environment variables**:
Create a `.env` file in the project root:
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Fireworks API
FIREWORKS_API_KEY=your_fireworks_api_key

```

## 🚀 Quick Start

### 1. Run the Complete Pipeline

Use the provided script to run the entire pipeline:

```bash
# Edit run_pipeline.sh to set your parameters
DATASET="LBPPPython"
MODEL="qwen3-coder-480b-a35b-instruct"
BACKEND="fireworks"
TEST_APPROACH="self-consistency"
CODE_APPROACH="vanilla"

# Run the complete pipeline
bash run_pipeline.sh
```

### 2. Run Individual Components

#### Test Generation
```bash
python test_generator.py \
    --dataset LBPPPython \
    --model qwen3-coder-480b-a35b-instruct \
    --backend fireworks \
    --approach self-consistency \
```

#### Code Generation
```bash
python generate_solutions.py \
    --dataset LBPPPython \
    --llm qwen3-coder-480b-a35b-instruct \
    --backend fireworks \
    --approach vanilla \
```

#### Dual Agreement Evaluation
```bash
export PYTHONPATH=`pwd`
python dual/dual_agreement.py \
    --dataset LBPPPython \
    --llm qwen3-coder-480b-a35b-instruct \
    --test_approach self-consistency \
    --code_approach vanilla
```

#### Test Coverage Analysis
```bash
python evaluation/test_coverage.py \
    --dataset LBPPPython \
    --llm qwen3-coder-480b-a35b-instruct \
    --test_approach self-consistency \
    --code_approach vanilla
```

#### Mutation Testing
```bash
python evaluation/mutation_testing.py \
    --dataset LBPPPython \
    --llm qwen3-coder-480b-a35b-instruct \
    --test_approach self-consistency \
    --code_approach vanilla
```

## 📊 Available Datasets

| Dataset | Description | Loader |
|---------|-------------|---------|
| `BigCodeBenchHard` | Hard coding problems from BigCodeBench | `BigCodeLoader` |
| `LBPPPython` | Python problems from LBPP | `LBPPLoaderPython` |

## 🤖 Supported Models

| Model | Backend   | Description |
|-------|-----------|-------------|
| `gpt-5-mini` | OpenAI    | OpenAI's latest model |
| `qwen3-coder-480b-a35b-instruct` | fireworks | Qwen3 Coder model |
| `gemma3` | VLLM      | Google's Gemma 3 |

## 🔧 Configuration

### Test Generation Approaches

1. **Self-consistency** (`self-consistency`):
   - Generates multiple test completions
   - Uses frequency-based selection for consistency
   - Best for reliable test generation

2. **Two-stage** (`two-stage`):
   - First generates test stubs
   - Then enriches with assertions
   - Good balance of structure and detail

3. **Holistic** (`holistic`):
   - Generates complete test suites in one pass
   - Fastest approach
   - Good for simple problems

### Code Generation Approaches

1. **Vanilla** (`vanilla`):
   - Direct LLM code generation
   - Fastest approach
   - Baseline for comparison

2. **CoVe** (`CoVe`):
   - Chain of Verification pipeline
   - Generates verification questions
   - Corrects code based on verification results
   - Most thorough but slowest

## 📁 Project Structure

```
TRIAD/
├── dual/                    # Dual agreement evaluation
│   ├── dual_agreement.py   # Main dual agreement logic
│   ├── agreement.py        # Agreement computation
│   └── execution.py        # Test execution utilities
├── evaluation/             # Evaluation modules
│   ├── test_coverage.py    # Coverage analysis
│   ├── mutation_testing.py # Mutation testing
│   └── mutation_testing_unittest.py
├── loaders/                # Dataset loaders
├── prompts/                # LLM prompts
├── test_generator.py       # Test generation main script
├── generate_solutions.py   # Code generation main script
├── CoVe.py                 # Chain of Verification implementation
├── function_executor.py    # Test execution utilities
├── llm_requester.py        # LLM interface abstraction
└── reusable_classes.py     # Common data structures
```

## 📈 Output Structure

The framework generates outputs in the following structure:

```
output/
├── generated_tests/
│   ├── stub/                    # Test stubs
│   └── final_tests/            # Final generated tests
│       ├── self-consistency/
│       ├── two-stage/
│       └── holistic/
├── generated_solutions/
│   ├── vanilla/                # Vanilla solutions
│   └── CoVe/                   # CoVe-verified solutions
├── verified/                   # Dual agreement results
└── evaluation_results/         # Coverage and mutation results
```

## 🔍 Evaluation Metrics

### Test Coverage
- **Line Coverage**: Percentage of code lines executed by tests
- **Branch Coverage**: Percentage of code branches executed by tests

### Mutation Testing
- **Mutation Score**: Percentage of mutants killed by tests
- **Operators**: Various mutation operators (arithmetic, logical, etc.)


### Common Issues

1.**API Rate Limits**:
   - Add delays between requests
   - Use different API keys
   - Switch to local VLLM backend

2.**Import Errors**:
   - Ensure all dependencies are installed
   - Check Python path: `export PYTHONPATH=`pwd``

### Debug Mode

Enable verbose logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0) License**.

You are free to:
- **Share** — Copy and redistribute the material in any medium or format.
- **Adapt** — Remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

[View Full License](https://creativecommons.org/licenses/by/4.0/)


## 📚 Citation

```bibtex

```

## 🙏 Acknowledgments

- BigCodeBench dataset
- LBPP dataset

**Note**: This framework is designed for research purposes. Please ensure you have appropriate API access and computational resources before running large-scale experiments.
