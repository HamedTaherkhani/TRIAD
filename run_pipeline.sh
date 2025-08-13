#!/bin/bash

DATASET="LBPPPython"
MODEL="llama4-maverick-instruct-basic"
BACKEND="fireworks"
TEST_APPROACH="self-consistency"
CODE_APPROACH="CoVe"

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


