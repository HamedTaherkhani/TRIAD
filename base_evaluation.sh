#!/bin/bash
DATASET="BigCodeBenchHard"
MODEL="qwen3-coder-480b-a35b-instruct"
TEST_APPROACH="holistic"
CODE_APPROACH="base"

export PYTHONPATH=`pwd`

python evaluation/test_coverage.py \
    --dataset "$DATASET" \
    --llm "$MODEL" \
    --test_approach "$TEST_APPROACH" \
    --code_approach "$CODE_APPROACH"

python evaluation/mutation_testing.py \
    --dataset "$DATASET" \
    --llm "$MODEL" \
    --test_approach "$TEST_APPROACH" \
    --code_approach "$CODE_APPROACH"