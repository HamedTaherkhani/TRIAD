## start_vllm.sh
#MODEL="/home/hamed/PycharmProjects/TRIAD/huggingface/gemma_3_1b/"   # or whatever you use
#PORT=8000
#TP=1                                     # set >1 if using multiple GPUs
#ALIAS="local-vllm"                        # what clients will pass as model
#
##python -m vllm.entrypoints.openai.api_server \
##  --model "$MODEL" \
##  --served-model-name "$ALIAS" \
##  --host 0.0.0.0 --port "$PORT" \
##  --tensor-parallel-size "$TP" \
##  --max-model-len 8192 \
##  --max-num-batched-tokens 8192 \
##  > vllm_server.log 2>&1 &
#
#python -m vllm.entrypoints.openai.api_server \
#  --model "$MODEL" \
#  --served-model-name "$ALIAS" \
#  --host 0.0.0.0 --port "$PORT" \
#  --tensor-parallel-size "$TP" \
#  > vllm_server.log 2>&1 &
#
## simple wait loop until the server is ready
#until curl -s "http://localhost:$PORT/health" | grep -q ok; do
#  sleep 1
#done
#echo "vLLM ready on :$PORT as model '$ALIAS'"

MODEL="google/gemma-3-12b-it"
PORT=8000
TP=1
ALIAS="local-vllm"

# helpful allocator setting to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# kill a previous server if any
# pkill -f "vllm.entrypoints.api_server" || true

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$ALIAS" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --enforce-eager \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 49152 \
  --max-num-seqs 6 \
  --gpu-memory-utilization 0.85


# wait for health
until curl -s "http://localhost:$PORT/health" | grep -q OK; do sleep 1; done
echo "vLLM ready on :$PORT as model '$ALIAS'"

