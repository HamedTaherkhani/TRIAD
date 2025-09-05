#!/bin/bash
#SBATCH --job-name=triad_vllm_pipeline
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH -c 32
#SBATCH --mem=42000
#SBATCH --output=output.log
#SBATCH --account=rrg-hvpham
set -euo pipefail
echo "Running on $(hostname)"
echo "SLURM job id: $SLURM_JOB_ID"

# -------------------------
# Activate environment
# -------------------------
module --force purge
source /home/hamedth/miniconda3/etc/profile.d/conda.sh
conda activate Triad

# Helpful allocator setting to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------
# vLLM server config
# -------------------------
# Path to your HF model snapshot (update if needed)
MODEL="/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80/"
PORT=8005
TP=1

# IMPORTANT: make the served name match what your pipeline passes as --model
SERVED_NAME="local-vllm"

# OpenAI-compatible endpoint env for your scripts
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="dummy"             # vLLM ignores key, but many clients require it
export OPENAI_MODEL="${SERVED_NAME}"

# (Optional) pin to the first GPU of the allocation
# export CUDA_VISIBLE_DEVICES=0

# -------------------------
# Start vLLM server
# -------------------------
VLLM_LOG="vllm_server.log"

echo "Starting vLLM server for '${MODEL}' on :$PORT as '${SERVED_NAME}'..."
python -m vllm.entrypoints.openai.api_server \
 --model "$MODEL" \
 --served-model-name "$SERVED_NAME" \
 --host 0.0.0.0 --port "$PORT" \
 --tensor-parallel-size "$TP" \
 --enforce-eager \
 --dtype bfloat16 \
 --max-model-len 19000 \
 --max-num-batched-tokens 120000 \
 --max-num-seqs 12 \
 --gpu-memory-utilization 0.8 \
 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID (logs: $VLLM_LOG)"
cleanup() { 
  echo "Stopping vLLM (PID $VLLM_PID)..." 
  kill $VLLM_PID 2>/dev/null || true 
  wait $VLLM_PID 2>/dev/null || true 
  }
trap cleanup EXIT
# -------------------------
# Wait until the server is healthy
# -------------------------
echo -n "Waiting for vLLM health check"
URL="http://127.0.0.1:${PORT}/health"
DEADLINE=$((SECONDS + 600))  # 10 minutes

while (( SECONDS < DEADLINE )); do
  # returns just the HTTP code; no body
  code=$(curl -s -o /dev/null -w "%{http_code}" "$URL" || echo 000)
  echo "$code"
  if [[ "$code" == "200" ]]; then
    echo
    echo "vLLM ready on :$PORT as model '$SERVED_NAME'"
    break
  fi

  # if the server died, stop waiting and show why
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo
    echo "vLLM process exited unexpectedly. Tail of log:"
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi

  echo -n "."
  sleep 1
done

if (( SECONDS >= DEADLINE )); then
  echo
  echo "Timed out waiting for vLLM health. Tail of log:"
  tail -n 200 "$VLLM_LOG" || true
  exit 1
fi
# -------------------------
# Run your pipeline
# -------------------------
PIPELINE_LOG="pipeline.log"
echo "Launching run_pipeline.sh (logging to $PIPELINE_LOG)..."
bash ./run_pipeline.sh > "$PIPELINE_LOG" 2>&1
PIPELINE_RC=$?

echo "Pipeline exit code: $PIPELINE_RC"
exit $PIPELINE_RC
