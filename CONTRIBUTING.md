# Contributing Guide

## Branching Strategy

```
main (protected)
  ↑
feature/*, fix/* (individual tasks)
```

- **`main`**: Stable, reviewed code. All changes go through a pull request.
- **`feature/*`, `fix/*`**: Short-lived branches for specific changes.

## Development Workflow

### 1. Start with Latest Code

```bash
git checkout main
git pull origin main

# Create a feature branch
git checkout -b feature/my-change
```

### 2. Set Up Your Environment

```bash
make install   # Install deps and register pre-commit hooks
```

### 3. Make Your Changes

Edit sector modules, data, parameters, configuration, etc.

### 4. Test Locally

```bash
make lint      # Check code quality (ruff + codespell)
make format    # Auto-fix formatting
make test      # Run test suite

# Run the full model
uv run python -m transition_compass_model.model.interactions_localrun

# Or run an individual sector
uv run python -m transition_compass_model.model.transport_module
```

### 5. Commit

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add <files>
git commit -m "fix(transport): correct emission factor for EV fleet"
```

**Commit types**: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`

### 6. Push and Open a Pull Request

```bash
git push origin feature/my-change
```

1. Go to GitHub: https://github.com/2050Calculators/transition-compass-model
2. Click **"New Pull Request"**
3. Set base branch: `main` ← compare: `feature/my-change`
4. Fill in the PR: title (conventional commit format), description, what you tested

### Pull Request Checklist

- [ ] `make lint` passes
- [ ] `make format` has been run
- [ ] `make test` passes
- [ ] Model runs end-to-end locally
- [ ] Conventional commit messages used
- [ ] Branch is up-to-date with `main`

## Code Quality

[Ruff](https://docs.astral.sh/ruff/) (lint + format) and [Codespell](https://github.com/codespell-project/codespell) run in CI on every push and PR, and locally via pre-commit hooks. The CI check must pass before a PR can be merged.

Pre-commit hooks run automatically on `git commit` after `make install`. To run them manually:

```bash
uv run pre-commit run --all-files
```

## Releases and App Integration

When a new version is ready, tag and push:

```bash
git tag v1.2.3
git push origin v1.2.3
```

GitHub Actions automatically dispatches a `model-updated` event to [speed-to-zero](https://github.com/EPFL-ENAC/leure-speed-to-zero), which creates a bump PR (`chore/bump-model-v1.2.3 → dev`) updating the dependency and regenerating the lock file. See [versioning in the README](README.md#versioning-and-releases) for semver conventions.

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/2050Calculators/transition-compass-model/issues)
- **App integration questions**: See the [speed-to-zero repo](https://github.com/EPFL-ENAC/leure-speed-to-zero)
