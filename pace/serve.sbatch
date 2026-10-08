#!/bin/bash
#SBATCH -J kv-serve
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%A_%a.out
# E3 / E4 (C5): one array task per server config, a fresh `vllm serve` for every
# (workload, repetition), so no cache state carries over between runs.
#   task 0 full       no prefix caching, no connector (full prefill every request)
#   task 1 prefix     vLLM's automatic prefix caching
#   task 2 connector  ChunkReuseConnector (method 1), prefix caching off, store in host memory
# Needs pace/serve_prep.sbatch first.
#   sbatch --array=0-2 --export=ALL,MODE=e3 pace/serve.sbatch
#   sbatch --array=0-2 --export=ALL,MODE=e4 pace/serve.sbatch
#   sbatch --array=0-2 --export=ALL,MODE=e3,REPS=1,LIMIT=20,WORKLOADS=work/e3_k10.jsonl pace/serve.sbatch  # smoke
source pace/env.sh
set +u; deactivate 2>/dev/null || true; source "${VLLM_VENV:-$HOME/scratch/kvreuse-vllm-venv312}/bin/activate"; set -u
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"  # this copy's kvreuse, whatever -e install the venv has

MODE="${MODE:-e3}"
CONFIGS=(full prefix connector)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
REPS="${REPS:-3}"
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))  # unique per array task; two jobs can share a node
export VLLM_ENGINE_READY_TIMEOUT_S=1800  # the connector preloads the store during engine start-up
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
if [ "$MODE" = e3 ]; then
    STORE="${STORE:-$HOME/scratch/kvreuse-store-conv26}"
    WORKLOADS="${WORKLOADS:-work/e3_k2.jsonl work/e3_k4.jsonl work/e3_k6.jsonl work/e3_k8.jsonl work/e3_k10.jsonl}"
    CONC="${CONC:-1}"; SEEN=false
else
    STORE="${STORE:-$HOME/scratch/kvreuse-store-all}"
    WORKLOADS="${WORKLOADS:-work/e4_p0.jsonl work/e4_p0.25.jsonl work/e4_p0.5.jsonl work/e4_p0.75.jsonl work/e4_p0.9.jsonl}"
    CONC="${CONC:-16}"; SEEN=true
fi

if [ "$CONFIG" = connector ]; then
    # Stage the store in node-local memory once per job: preloading 33 GB cold from Lustre
    # takes >10 min per server start, from /dev/shm it is memory speed.
    STAGE="/dev/shm/kvreuse-store-$SLURM_JOB_ID"
    trap 'rm -rf "$STAGE"' EXIT
    mkdir -p "$STAGE"
    t0=$(date +%s)
    ls "$STORE" | xargs -P 16 -I{} cp "$STORE/{}" "$STAGE/"
    echo "staged $(du -sh "$STAGE" | cut -f1) in $(( $(date +%s) - t0 )) s"; df -h /dev/shm | tail -1
    STORE="$STAGE"
fi

FLAGS=(--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --seed 0 --port "$PORT"
       --enable-prompt-tokens-details)
case "$CONFIG" in
    full)   FLAGS+=(--no-enable-prefix-caching) ;;
    prefix) FLAGS+=(--enable-prefix-caching) ;;
    connector)
        FLAGS+=(--no-enable-prefix-caching --kv-transfer-config "{\"kv_connector\": \"ChunkReuseConnector\",
            \"kv_connector_module_path\": \"kvreuse.vllm_connector\", \"kv_role\": \"kv_both\",
            \"kv_connector_extra_config\": {\"store_dir\": \"$STORE\", \"kv_device\": \"cpu\",
            \"preload\": true, \"require_seen\": $SEEN, \"profile\": ${PROFILE:-false}}}") ;;
esac

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import vllm; print('vllm', vllm.__version__)"
echo "mode $MODE config $CONFIG reps $REPS conc $CONC port $PORT start $(date)"
mkdir -p results/serve logs/serve

for wl in $WORKLOADS; do
    for rep in $(seq 0 $(( REPS - 1 ))); do
        name="$(basename "$wl" .jsonl)_${CONFIG}_r${rep}"
        log="logs/serve/${name}_${SLURM_JOB_ID}.log"
        setsid vllm serve "$MODEL" "${FLAGS[@]}" > "$log" 2>&1 &
        pid=$!
        for _ in $(seq 1 180); do  # up to 15 min (connector preload of the full store is slow)
            curl -sf "localhost:$PORT/health" >/dev/null && break
            kill -0 "$pid" 2>/dev/null || { echo "server died: $log"; tail -40 "$log"; exit 1; }
            sleep 5
        done
        curl -sf "localhost:$PORT/health" >/dev/null || { echo "server not up: $log"; kill -- -"$pid"; exit 1; }
        python scripts/bench_serve.py --url "http://localhost:$PORT" --workload "$wl" --concurrency "$CONC" \
            $LIMIT_ARG --out "results/serve/$name.jsonl" \
            --tag mode="$MODE" config="$CONFIG" rep="$rep" gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
        grep "ChunkReuseConnector stats" "$log" | tail -1 || true
        kill -- -"$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
        sleep 5
    done
done
echo "done $(date)"
