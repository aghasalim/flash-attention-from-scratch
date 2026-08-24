# Use the repo venv for everything. Override with `make PY=/some/other/python test`.
PY ?= $(CURDIR)/.venv/bin/python

.PHONY: setup env test bench profile lint clean

setup:
	$(PY) -m pip install -e .
	@echo "note: triton is in the [gpu] extra, not the base deps -- it has no macOS wheels."
	@echo "on an NVIDIA box run: $(PY) -m pip install -e '.[gpu]'"

env:
	$(PY) -m scripts.env

test:
	$(PY) -m pytest

bench:
	@if [ -z "$$(ls bench/*.py 2>/dev/null | grep -v __init__)" ]; then \
		echo "no benchmarks yet -- bench/ is filled in by tasks 01 and 07"; \
	else \
		for f in $$(ls bench/*.py | grep -v __init__); do \
			echo "== $$f"; $(PY) "$$f" || exit 1; \
		done; \
	fi

profile:
	@if command -v ncu >/dev/null 2>&1; then \
		ncu --set full --target-processes all $(PY) -m bench.profile_target; \
	else \
		echo "ncu not found -- no NVIDIA profiler on this platform."; \
		echo "Nsight Compute ships with the CUDA toolkit and is NVIDIA-only;"; \
		echo "this machine has no CUDA device (see HARDWARE.md)."; \
		echo "Profiling the Apple GPU means Xcode Instruments / Metal System Trace,"; \
		echo "which is not wired up here."; \
	fi

lint:
	@if $(PY) -m ruff --version >/dev/null 2>&1; then \
		$(PY) -m ruff check . && $(PY) -m ruff format --check . ; \
	else \
		echo "ruff not installed -- skipping lint. install with: $(PY) -m pip install -e '.[dev]'"; \
	fi

# -path pruning is load-bearing: without it `find . -name '*.so' -delete` walks into
# .venv and deletes torch's shared libraries.
PRUNE = -path ./.venv -prune -o -path ./.git -prune -o

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . $(PRUNE) -name __pycache__ -type d -prune -exec rm -rf {} +
	find . $(PRUNE) -name '*.so' -type f -print -delete
	@echo "kept results/ -- the CSVs are checked in on purpose."
