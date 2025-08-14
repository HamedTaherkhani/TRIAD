#!/bin/bash

DATASET="LBPPPython"
MODEL="qwen3-coder-480b-a35b-instruct"
BACKEND="fireworks"
TEST_APPROACH="holistic"
CODE_APPROACH="vanilla"
NUM_INSTANCES=5
python test_generator.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --approach "$TEST_APPROACH"

python generate_solutions.py \
    --dataset "$DATASET" \
    --llm "$MODEL" \
    --backend "$BACKEND" \
    --approach "$CODE_APPROACH"

export PYTHONPATH=`pwd`
python dual/dual_agreement.py \
    --dataset "$DATASET" \
    --llm "$MODEL" \
    --test_approach "$TEST_APPROACH" \
    --code_approach "$CODE_APPROACH"


