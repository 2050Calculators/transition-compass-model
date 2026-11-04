"""Helpers to turn cached FAOSTAT downloads into trade datamatrices.

Used by the pre-processing scripts. Two rules apply everywhere:

- The FAOSTAT Trade Matrix (TM) leaves out the partner-years with no trade.
  A missing partner-year is therefore a zero, never a gap to interpolate.
- Totals (import reference, exports) come from the Food Balance Sheets (FBS),
  the same source as the SSR and the demand. TM is only used for the origin
  shares, which are normalised, so its level and basket do not matter there.
"""

import numpy as np
import pandas as pd

from tcaf_model.model.common.data_matrix_class import DataMatrix


def _wide_to_dm(df_wide, variable, unit, categories):
    """Build a datamatrix from a (Country, Years) x category table."""
    df = df_wide.reset_index()
    df = df.rename(
        columns={c: f"{variable}_{c}[{unit}]" for c in categories},
    )
    return DataMatrix.create_from_df(df, num_cat=1)


def _future_mean(df_wide, years_ots, years_fts, n_years):
    """Append the future years, each holding the mean of the last n ots years.

    df_wide is indexed by (Country, Years).
    """
    last_years = years_ots[-n_years:]
    mean = (
        df_wide[df_wide.index.get_level_values("Years").isin(last_years)]
        .groupby(level="Country")
        .mean()
    )
    future = [mean.assign(Years=y).set_index("Years", append=True) for y in years_fts]
    return pd.concat([df_wide, *future]).sort_index()


def trade_matrix_partners(
    df_trade_agg,
    years_ots,
    years_fts,
    variable="agr_split-import",
    unit="t",
    n_years=5,
):
    """Imports by partner country, in tonnes, from a cached TM file.

    df_trade_agg has the columns "Partner Countries", "Item" (the model
    category), "Year" and "Value" [t]. Partner-years absent from the file are
    set to 0. The future years hold each partner's mean over the last n_years
    historical years, so the origin shares stay at their recent average.

    Returns a datamatrix (Country = partner, Categories1 = category) in t,
    labelled with `unit` (the pre-processing scripts label it "-" and convert
    it to kcal afterwards).
    """
    df = df_trade_agg[df_trade_agg["Year"].isin(years_ots)]
    df = df.groupby(["Partner Countries", "Year", "Item"])["Value"].sum()
    categories = sorted(df.index.get_level_values("Item").unique())
    partners = sorted(df.index.get_level_values("Partner Countries").unique())
    full = pd.MultiIndex.from_product(
        [partners, years_ots, categories],
        names=["Partner Countries", "Year", "Item"],
    )
    df = df.reindex(full, fill_value=0.0).clip(lower=0.0)
    df_wide = df.unstack("Item")
    df_wide.index = df_wide.index.set_names(["Country", "Years"])
    df_wide = _future_mean(df_wide, years_ots, years_fts, n_years)
    return _wide_to_dm(df_wide, variable, unit, categories)


def fbs_elements(df_fbs, item_to_category, country, years_ots):
    """FBS elements per model category, in kt, indexed by Years.

    df_fbs is a cached FAOSTAT FBS/FBSH download (columns Area, Item,
    Element, Year, Value). FBSH ends in 2009 and FBS starts in 2010, so no
    year is counted twice; if a year appears in both, the FBS value is kept.
    Missing elements are 0.

    "Stock Variation" is returned with the FBS sign convention in every year:
    Import = Food - Production + Export + Stock Variation + other uses, i.e.
    positive when stocks are built up. FBSH (up to 2009) publishes it with the
    opposite sign, so FBSH values are negated. (Checked on Switzerland,
    beverages 2000-2023: with this sign the balance closes within the 1 kt
    rounding in every year; with one sign for both vintages it misses by up to
    49 kt.)
    """
    df = df_fbs[(df_fbs["Area"] == country) & df_fbs["Item"].isin(item_to_category)]
    df = df.assign(category=df["Item"].map(item_to_category))
    if "Domain Code" in df.columns:
        is_fbsh_stock = df["Domain Code"].eq("FBSH") & df["Element"].eq(
            "Stock Variation"
        )
        df.loc[is_fbsh_stock, "Value"] = -df.loc[is_fbsh_stock, "Value"]
        # FBS (new methodology) wins over FBSH where both exist
        df = df.sort_values("Domain Code", key=lambda s: s.eq("FBS"))
        df = df.drop_duplicates(["category", "Element", "Year"], keep="last")
    df = df.pivot_table(
        index=["Year", "category"], columns="Element", values="Value", aggfunc="sum"
    )
    df = df.reindex(
        pd.MultiIndex.from_product(
            [years_ots, sorted(set(item_to_category.values()))],
            names=["Year", "category"],
        )
    )
    return df.fillna(0.0)


def fbs_imports_kcal(df_elements, cdm_kcal, country, variable):
    """FBS Import [kt] -> kcal, as a one-country datamatrix (ots years only)."""
    df_wide = df_elements["Import"].unstack("category") * 1000.0
    categories = list(df_wide.columns)
    idx = cdm_kcal.idx
    for c in categories:
        df_wide[c] = df_wide[c] * cdm_kcal.array[idx["cp_kcal-per-t"], idx[c]]
    df_wide["Country"] = country
    df_wide.index.name = "Years"
    df_wide = df_wide.set_index("Country", append=True).swaplevel()
    return _wide_to_dm(df_wide, variable, "kcal", categories)


def fbs_exports_kcal(
    df_elements, cdm_kcal, country, years_ots, years_fts, variable, n_years=10
):
    """FBS Export [kt] -> kcal, for ots and fts years.

    Exports are kept as a volume, not as a share of production: for small
    categories the share is meaningless (spirits: 1 kt produced, 1 kt
    exported, a share of 100% that would make every extra unit produced go
    abroad). The future years hold the mean of the last n_years historical
    years. The model caps exports at domestic production.
    """
    df_wide = df_elements["Export"].unstack("category") * 1000.0
    categories = list(df_wide.columns)
    idx = cdm_kcal.idx
    for c in categories:
        df_wide[c] = df_wide[c] * cdm_kcal.array[idx["cp_kcal-per-t"], idx[c]]
    df_wide["Country"] = country
    df_wide.index.name = "Years"
    df_wide = df_wide.set_index("Country", append=True).swaplevel()
    df_wide = _future_mean(df_wide, years_ots, years_fts, n_years)
    return _wide_to_dm(df_wide, variable, "kcal", categories)


def fbs_stock_and_other_uses_kcal(
    df_elements, cdm_kcal, country, years_ots, years_fts, variable
):
    """FBS stock variation + other (non-food) uses + residuals [kt] -> kcal.

    These are the terms of the FBS balance that the model does not simulate:
    Import = Food - Production + Export + this. Historical years only; the
    future years are 0 (no stock change, no non-food use projected).
    """
    elements = [
        e
        for e in ["Stock Variation", "Other uses (non-food)", "Residuals"]
        if e in df_elements.columns
    ]
    df_wide = df_elements[elements].sum(axis=1).unstack("category") * 1000.0
    categories = list(df_wide.columns)
    idx = cdm_kcal.idx
    for c in categories:
        df_wide[c] = df_wide[c] * cdm_kcal.array[idx["cp_kcal-per-t"], idx[c]]
    df_wide["Country"] = country
    df_wide.index.name = "Years"
    df_wide = df_wide.set_index("Country", append=True).swaplevel()
    future = [
        df_wide.xs(years_ots[-1], level="Years", drop_level=False)
        .rename(index={years_ots[-1]: y}, level="Years")
        .mul(0.0)
        for y in years_fts
    ]
    df_wide = pd.concat([df_wide, *future]).sort_index()
    return _wide_to_dm(df_wide, variable, "kcal", categories)


def finite_or_zero(array):
    """Replace inf and NaN by 0 (for yields of countries with no data)."""
    return np.where(np.isfinite(array), array, 0.0)
