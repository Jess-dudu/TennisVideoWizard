# Default recipe
@_:
    just --list

# Build & serve mkdocs
mkdocs:
    mkdocs build
    mkdocs serve
    
# Run tests
test:
    uv run -m pytest -s

# Train ML for 3-class animal
train_cls3_animal:
    uv run ./src/frame_classifier/train.py --config cls3_animal.yaml

# Train ML for 2-class tennis
train_cls2_tennis:
    uv run ./src/frame_classifier/train.py --config cls2_tennis.yaml

# Setup project virtualenv (default to cpu-only)
[group('lifecycle')]
install:
    uv sync --extra cpu
    uv pip install -e .

# Setup project virtualenv cpu-only
[group('lifecycle')]
install-cpu:
    uv sync --extra cpu
    uv pip install -e .

# Setup project virtualenv cu126
[group('lifecycle')]
install-cu126:
    uv sync --extra cu126
    uv pip install -e .

# Setup project virtualenv cu129
[group('lifecycle')]
install-cu129:
    uv sync --extra cu129
    uv pip install -e .

# Remove temporary files
[group('lifecycle')]
clean:
    rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
    find . -type d -name "__pycache__" -exec rm -r {} +

# Recreate project virtualenv fresh
[group('lifecycle')]
fresh: clean install
