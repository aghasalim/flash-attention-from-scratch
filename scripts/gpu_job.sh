#!/usr/bin/env bash
# Runs on a rented NVIDIA box. Fingerprints the GPU, runs the whole suite with
# Triton installed, and sends results/gpu/ back with runpodctl.
#
# Until fa/ops/attention.py exists the kernel tests report XFAIL, which is the
# expected result. Once it exists they either pass or fail for real.
set -uo pipefail
set -x
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
curl -sfL -o /usr/local/bin/runpodctl \
  https://github.com/runpod/runpodctl/releases/download/v2.14.0/runpodctl-linux-amd64 \
  && chmod +x /usr/local/bin/runpodctl && hash -r
cd /workspace
git clone -q "${REPO_URL:-https://github.com/aghasalim/flash-attention-from-scratch}" fa
cd fa
git checkout -q "${REF:-main}"
pip install -q -e ".[gpu]" 2>&1 | tail -2
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.cuda.is_available())"
set +x

out=results/gpu
mkdir -p "$out"
git rev-parse HEAD > "$out/commit.txt"

# The CUDA path of scripts/env.py measures HBM bandwidth and matmul peaks. It
# writes HARDWARE.md at the root, which describes the M4, so keep a copy here
# and leave the committed one alone.
python -m scripts.env > "$out/env.log" 2>&1
cp HARDWARE.md "$out/HARDWARE.md"; cp hardware.json "$out/hardware.json"
git checkout -q HARDWARE.md hardware.json
echo "ENV_DONE"

python -m pytest -q -rxX > "$out/pytest.txt" 2>&1
tail -3 "$out/pytest.txt"
echo "TESTS_DONE"

if [ -f fa/ops/attention.py ]; then
  echo "kernel present, the benches still time cpu and mps only; add cuda to bench/roofline.py" > "$out/bench.txt"
else
  echo "no fa/ops/attention.py, nothing to benchmark on the GPU yet" > "$out/bench.txt"
fi

tar czf /workspace/gpu_results.tgz "$out"
runpodctl send /workspace/gpu_results.tgz > /workspace/send.log 2>&1 &
for _ in $(seq 60); do
  c=$(tr '\r' '\n' < /workspace/send.log | grep -m1 -o "code is: .*")
  [ -n "$c" ] && { echo "SEND_$c"; break; }
  sleep 2
done
sleep infinity
