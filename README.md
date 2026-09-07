# tcaf-model

True Cost Accounting of Food (TCAF) build of the Transition Compass model.

It is a **separate build** of [transition-compass-model][model], not a newer
version of it. It keeps the food side (agriculture, crop, livestock, land use,
dietary habits, alcoholic beverages, population, true cost) and drops the other
sectors. Region: Switzerland.

Deployed at <https://transition-compass-tcaf.epfl.ch/>.

## Branches

| Branch | What it is |
|---|---|
| `tcaf` | the working branch, what gets released |
| `tcaf-import` | the raw import, do not touch |

`tcaf-import` is the research work exactly as it was written in
[2050Calculators/tcaf-calc][fork], branch `model`, replayed commit by commit
with the original authors and dates. That fork was taken from the web app
repository before the app and the model were split, so the model code sat in
`backend/model/`. Here it sits in `transition_compass_model/model/`, the same
place as in the main model repo, and nothing else changed.

`tcaf` adds what the code needs to be a real python package: the `__init__.py`
files, `pyproject.toml`, the lint setup and the release workflow. The science is
untouched.

See [issue #11][issue] for the background.

## Install

```bash
make install          # uv sync --all-groups + pre-commit
```

## Use it

```python
from transition_compass_model.model.interactions import runner
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
)

DM_input = filter_country_and_load_data_from_pickles(
    country_list=["Switzerland"], modules_list=["population", "agriculture"]
)
output, kpi = runner(lever_setting, years_setting, DM_input, ["agriculture"], logger)
```

Each module also runs on its own, on the pickles in `_database/`:

```bash
python -m transition_compass_model.model.crop_module
```

## Release

The web app installs the wheel from a GitHub release, there is no PyPI package
(the name `transition-compass-model` on PyPI belongs to the main model).

```bash
git tag tcaf-v1.0.0 && git push origin tcaf-v1.0.0
```

The workflow lints, builds the wheel with the pickles inside, and attaches it to
the release. Then point the app at the new URL in `backend/pyproject.toml` of
[tcaf-calc][fork] and run `uv lock`.

## Working here

- Data files (`*.pickle`, `*.pdf`) are in git LFS. Run `git lfs pull` if you get
  small text files instead of data.
- `make format` before committing, CI runs ruff and codespell.
- Keep the module names importable: no dash in a `.py` file name.

[model]: https://github.com/2050Calculators/transition-compass-model
[fork]: https://github.com/2050Calculators/tcaf-calc
[issue]: https://github.com/2050Calculators/transition-compass-model/issues/11
