# Development Guide

Local setup, dev workflow, and release process for `transition-compass-model`.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) — `pip install uv`
- git

## Standalone install

```bash
git clone https://github.com/2050Calculators/transition-compass-model.git
cd transition-compass-model
uv sync
```

This creates a `.venv` and installs all dependencies. You can then run scripts directly with `uv run python ...`.

## Running the model locally

```bash
# Full model run (reads lever config from transition_compass_model/config/lever_position.json)
uv run python -m transition_compass_model.model.interactions_localrun
```

Each sector module also has a `local_*_run()` function for standalone testing:

```bash
uv run python -m transition_compass_model.model.transport_module
uv run python -m transition_compass_model.model.buildings_module
# etc.
```

### IDE setup

The key requirement for all IDEs is to point the Python interpreter at `.venv/bin/python` inside this repo. Once that is done, you can open any `*_module.py` file and run or debug it directly.

#### VS Code

1. Open the repo folder in VS Code.
2. Open the Command Palette (`Ctrl+Shift+P`) → **Python: Select Interpreter**.
3. Choose the interpreter at `.venv/bin/python` (it usually appears automatically as a workspace suggestion).
4. Open any `*_module.py` file and click **Run Python File** (▷ top-right) or press `F5` to debug.

A `.vscode/settings.json` is already committed to this repo with the correct interpreter path, so steps 2–3 may be automatic.

#### Spyder

Spyder must use the same virtual environment as the project, otherwise imports will fail.

**Option A — launch Spyder from the activated venv (recommended):**

```bash
# Activate the venv first
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

# Install Spyder inside the venv (once)
pip install spyder

# Launch
spyder
```

**Option B — use an existing Spyder installation with a custom kernel:**

1. With the venv activated, install `spyder-kernels`:
   ```bash
   pip install spyder-kernels
   ```
2. In Spyder → **Preferences → Python interpreter → Use the following interpreter** → browse to `.venv/bin/python`.
3. Restart the kernel.

Once the interpreter is set, open any `*_module.py` and run it with `F5` or **Run → Run file**.

#### PyCharm

1. Go to **Settings → Project → Python Interpreter → Add Interpreter → Existing**.
2. Browse to `.venv/bin/python`.
3. Right-click any `*_module.py` → **Run**.

---

> **Note on pickle compatibility:** The data files in `_database/data/` were serialized before the package was renamed. The package registers backward-compatible `sys.modules` aliases at import time (`transition_compass_model/__init__.py`), so plain `pickle.load()` works transparently — no special handling needed.

---

## Working alongside the app (local editable mode)

For iterative model development, install the model as an editable package inside the app's virtual environment so changes are reflected immediately without reinstalling.

### Directory layout

The app's Makefile expects both repos to be siblings under the same parent directory:

```
parent-dir/
├── speed-to-zero/               ← app repo (leure-speed-to-zero)
│   └── backend/
│       └── Makefile             ← defines install-local, install, check-model
└── transition-compass-model/    ← this repo
```

### Setup

```bash
# 1. Clone both repos as siblings
git clone https://github.com/EPFL-ENAC/leure-speed-to-zero.git speed-to-zero
git clone https://github.com/2050Calculators/transition-compass-model.git transition-compass-model

# 2. Install the app using the local model folder (editable install)
cd speed-to-zero/backend
make install-local
```

`make install-local` runs `uv sync --frozen` (installs all locked dependencies) then overrides `transition-compass-model` with an editable install pointing at `../../transition-compass-model`. Any change you make to the model files is immediately active — no reinstall needed.

### Verify which model is active

```bash
# From speed-to-zero/backend/
make check-model
# prints: Model source: /path/to/transition_compass_model/__init__.py
```

- **Local mode** — path points inside `../../transition-compass-model/`
- **Remote mode** — path points inside `.venv/lib/.../site-packages/`

### Switch back to the remote (git-pinned) model

```bash
make install
# runs `uv sync --frozen` — restores the version pinned in uv.lock, drops editable install
```

Confirm with `make check-model` — the path should now point inside `.venv`.

### Run the full app

```bash
# From speed-to-zero/ root
make run
# starts backend (port 8000) + frontend (port 9000)
# also prints the active model source at startup
```

---

## Versioning and deployment

Releases follow [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`

| Part    | When to bump                                      |
| ------- | ------------------------------------------------- |
| `MAJOR` | Breaking changes to the module API or data format |
| `MINOR` | New sectors, modules, or significant new features |
| `PATCH` | Bug fixes, parameter tweaks, data updates         |

### Release flow

1. **Commit and push** all model changes to the `main` branch.

2. **Create and push a tag**:

   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

3. **Automation kicks in**: the `notify-speed-to-zero` GitHub Actions workflow fires, dispatching a `model-updated` event to `leure-speed-to-zero`.

4. **A bump PR is created automatically**: `chore/bump-model-v1.2.3 → dev` in `leure-speed-to-zero`, updating `pyproject.toml` and regenerating `uv.lock`.

5. **Review and merge** the PR — CI validates the Docker build, then `deploy.yml` deploys to production on merge.
