## install
```bash
# Clone the repository with submodules
git submodule update --init --recursive

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install mteb from submodule (editable mode)
uv pip install -e ./mteb

# Install colpali-engine
uv pip install .
```

## eval
```
mteb run -b "ViDoRe(v2)" -m "vidore/colqwen2.5-v0.2"
```
