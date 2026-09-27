#!/usr/bin/env bash
# Rent one GPU on RunPod, run scripts/gpu_job.sh on it, bring results/gpu/ back,
# and delete the pod. Needs runpodctl configured with your key.
#
#   scripts/runpod_gpu.sh              # tests main
#   REF=my-branch scripts/runpod_gpu.sh
#
# The pod cannot delete itself (its own key gets 403), so a timer on this
# machine deletes it after MAX_MIN minutes whatever happens.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-"COMMUNITY|NVIDIA GeForce RTX 3090;COMMUNITY|NVIDIA RTX A5000;SECURE|NVIDIA RTX A5000"}
IMAGE=${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}
MAX_MIN=${MAX_MIN:-45}
REF=${REF:-main}

job=$(base64 < scripts/gpu_job.sh | tr -d '\n')
env_json=$(printf '{"JOB":"%s","REF":"%s"}' "$job" "$REF")

pod=""
IFS=';' read -ra specs <<< "$GPUS"
for try in $(seq 30); do
  for spec in "${specs[@]}"; do
    out=$(runpodctl pod create --name fa-gpu --cloud-type "${spec%%|*}" --gpu-id "${spec#*|}" \
      --image "$IMAGE" --container-disk-in-gb 30 --env "$env_json" \
      --docker-args "bash -c 'echo \$JOB | base64 -d | bash'" 2>&1) || true
    pod=$(printf '%s' "$out" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)
    [ -n "$pod" ] && { echo "pod $pod on ${spec#*|}, ${spec%%|*}"; break 2; }
  done
  sleep 30
done
[ -n "$pod" ] || { echo "no GPU available"; exit 1; }

( sleep $((MAX_MIN * 60)); runpodctl pod delete "$pod" ) >/dev/null 2>&1 &
guard=$!
cleanup () { runpodctl pod delete "$pod" >/dev/null 2>&1 || true; pkill -P $guard 2>/dev/null; kill $guard 2>/dev/null || true; }
trap cleanup EXIT

logs () { runpodctl pod logs "$pod" 2>/dev/null | python3 -c "
import sys, json
for l in sys.stdin:
    try: print(json.loads(l)['line'])
    except Exception: pass"; }

until logs | grep -q "SEND_code is:"; do sleep 20; done
logs | grep -E "torch |ENV_DONE|passed|failed|xfail|TESTS_DONE" | tail -6
code=$(logs | grep -m1 -o "SEND_code is: .*" | sed 's/SEND_code is: //')
tmp=$(mktemp -d)
(cd "$tmp" && runpodctl receive "$code" >/dev/null && tar xzf gpu_results.tgz)
mkdir -p results
rm -rf results/gpu && mv "$tmp/results/gpu" results/gpu
echo "results in results/gpu/, pod deleted"
