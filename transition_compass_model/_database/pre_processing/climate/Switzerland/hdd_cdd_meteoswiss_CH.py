"""Population-weighted HDD(Tb) / CDD(Tb) tables per canton from MeteoSwiss TabsD.

Replaces the external (student-provided) HDD/CDD dataset with a reproducible
pipeline built only on official open data:

- MeteoSwiss TabsD: daily mean 2 m temperature, 1 km grid (LV95), one NetCDF per year
  https://data.geo.admin.ch/ch.meteoschweiz.ogd-surface-derived-grid/archive-ch/
  ogd-surface-derived-grid-archive.tabsd_ch01r.swiss.lv95_{Y}0101000000_{Y}1231000000.nc
- BFS STATPOP hectare grid (population weights, fixed year)
  https://data.geo.admin.ch/ch.bfs.statistik-bevoelkerung_haushalte/
  statistik-bevoelkerung_haushalte_2025/statistik-bevoelkerung_haushalte_2025_ha_2056.csv
- swisstopo swissBOUNDARIES3D (canton polygons)
  https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2026-01/
  swissboundaries3d_2026-01_2056_5728.gpkg.zip

Method
------
For every 1 km cell and day, degree-days are computed against a range of base
temperatures Tb, summed over the year, and only then averaged over cells with
population weights (HDD is non-linear in T, so cells are not averaged first).

    HDD(Tb) = sum_days max(0, Tb - T)      N_below(Tb) = #days with T <= Tb
    CDD(Tb) = sum_days max(0, T - Tb)      N_above(Tb) = #days with T >= Tb

The legacy two-threshold metrics used by the model follow exactly:

    HDD_18/15 = HDD(15) + 3 * N_below(15)
    CDD_21/24 = CDD(24) + 3 * N_above(24)

Output: data/meteoswiss_tabsd/hdd_cdd_annual_CH.csv (long format:
region, year, variable, Tb, value). Region = canton name (as in the legacy
files) or "Switzerland".

Raw inputs are large and live outside the repo (default ~/2050-Calculators/
climate_raw, override with CLIMATE_RAW_DIR). This script needs xarray, netCDF4 and
geopandas, which are not model dependencies; run it e.g. with

    uv run --no-project --python 3.12 --with xarray --with netCDF4 \
        --with geopandas --with pyogrio python hdd_cdd_meteoswiss_CH.py
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(
    os.environ.get("CLIMATE_RAW_DIR", Path.home() / "2050-Calculators" / "climate_raw")
)
TABSD_PATTERN = (
    "ogd-surface-derived-grid-archive.tabsd_ch01r.swiss.lv95_"
    "{y}0101000000_{y}1231000000.nc"
)
STATPOP_FILE = RAW_DIR / "statpop" / "statistik-bevoelkerung_haushalte_2025_ha_2056.csv"
BOUNDARIES_FILE = (
    "/vsizip/"
    + str(RAW_DIR / "boundaries" / "swissboundaries3d_2026-01_2056_5728.gpkg.zip")
    + "/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg"
)

YEARS = range(1990, 2026)
# Base temperatures, shared by all four variables so the table has one Tb dimension
TB = np.arange(-5, 31, 1.0)

OUT_DIR = Path(__file__).parent / "data" / "meteoswiss_tabsd"
OUT_FILE = OUT_DIR / "hdd_cdd_annual_CH.csv"


def population_weights(grid_E, grid_N):
    """Return (cell_index, region_names, W) with W[cell, region] summing to 1 per
    region. cell_index are flat indices into the (N, E) grid."""
    pop = pd.read_csv(STATPOP_FILE, sep=";", usecols=["E_KOORD", "N_KOORD", "BBTOT"])
    pop = pop[pop["BBTOT"] > 0]
    # STATPOP coordinates are the south-west corner of the hectare
    E_c = pop["E_KOORD"].to_numpy() + 50
    N_c = pop["N_KOORD"].to_numpy() + 50

    cantons = gpd.read_file(BOUNDARIES_FILE, layer="tlm_kantonsgebiet")
    cantons = cantons[cantons["icc"] == "CH"][["name", "geometry"]]
    pts = gpd.GeoDataFrame(
        {"pop": pop["BBTOT"].to_numpy()},
        geometry=gpd.points_from_xy(E_c, N_c),
        crs=cantons.crs,
    )
    joined = gpd.sjoin(pts, cantons, how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]
    missing = joined["name"].isna()
    if missing.any():
        # hectare centres falling just outside a polygon: take the nearest canton
        near = gpd.sjoin_nearest(pts[missing.to_numpy()], cantons, how="left")
        near = near[~near.index.duplicated()]
        joined.loc[missing, "name"] = near["name"]

    # Map hectare centre to the 1 km TabsD cell (grid coordinates are cell centres)
    dE = grid_E[1] - grid_E[0]
    iE = np.floor((E_c - (grid_E[0] - dE / 2)) / dE).astype(int)
    iN = np.floor((N_c - (grid_N[0] - dE / 2)) / dE).astype(int)
    flat = iN * len(grid_E) + iE

    df = pd.DataFrame(
        {"cell": flat, "region": joined["name"].to_numpy(), "pop": pts["pop"]}
    )
    by_canton = df.groupby(["cell", "region"])["pop"].sum().reset_index()
    by_ch = df.groupby("cell")["pop"].sum().reset_index().assign(region="Switzerland")
    table = pd.concat([by_canton, by_ch]).pivot_table(
        index="cell", columns="region", values="pop", fill_value=0
    )
    return table


def drop_nan_cells(table, T_first_day):
    """Remove cells where TabsD has no value and report the population share lost."""
    valid = ~np.isnan(T_first_day.ravel()[table.index.to_numpy()])
    lost = 1 - table[valid].sum() / table.sum()
    if (lost > 0).any():
        print("Population share in cells without TabsD (dropped):")
        print(lost[lost > 0].round(4).to_string())
    table = table[valid]
    return (
        table.index.to_numpy(),
        table.columns.tolist(),
        (table / table.sum()).to_numpy(),
    )


def degree_days_one_year(T, W):
    """T: (days, cells). Returns dict var -> array (nTb, regions)."""
    out = {}

    def annual_weighted(x):  # x: (days, cells) -> (regions,)
        return x.sum(axis=0, dtype=np.float64) @ W

    specs = {
        "HDD": lambda Tb: np.maximum(0, Tb - T),
        "N_below": lambda Tb: (T <= Tb).astype(np.float32),
        "CDD": lambda Tb: np.maximum(0, T - Tb),
        "N_above": lambda Tb: (T >= Tb).astype(np.float32),
    }
    for var, f in specs.items():
        out[var] = np.stack([annual_weighted(f(Tb)) for Tb in TB], axis=0)
    return out


def main():
    ds0 = xr.open_dataset(RAW_DIR / "tabsd" / TABSD_PATTERN.format(y=YEARS[0]))
    grid_E, grid_N = ds0["E"].to_numpy(), ds0["N"].to_numpy()
    table = population_weights(grid_E, grid_N)
    cells, regions, W = drop_nan_cells(table, ds0["TabsD"].isel(time=0).to_numpy())
    ds0.close()
    print(f"{len(cells)} populated 1 km cells, {len(regions)} regions")

    rows = []
    for y in YEARS:
        with xr.open_dataset(RAW_DIR / "tabsd" / TABSD_PATTERN.format(y=y)) as ds:
            T = ds["TabsD"].to_numpy().reshape(ds.sizes["time"], -1)[:, cells]
        if np.isnan(T).any():
            raise ValueError(f"NaN temperatures in populated cells for {y}")
        res = degree_days_one_year(T, W)
        for var, arr in res.items():
            t, r = np.meshgrid(range(len(TB)), range(len(regions)), indexing="ij")
            rows.append(
                pd.DataFrame(
                    {
                        "region": np.array(regions)[r.ravel()],
                        "year": y,
                        "variable": var,
                        "Tb": TB[t.ravel()],
                        "value": arr.ravel(),
                    }
                )
            )
        print(y, "done")

    df = pd.concat(rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False, float_format="%.3f")
    print("written", OUT_FILE, len(df), "rows")


if __name__ == "__main__":
    main()
