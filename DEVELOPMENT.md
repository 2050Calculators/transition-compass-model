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
