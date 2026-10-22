"""
Swiss freight OTS: total tkm and modal share by mode (1990-2023).

Accounting basis: territorial
-----------------------------
All modes count freight moved on Swiss territory, whatever the vehicle's registration.
This matches the other targets the preprocessing aligns to (EP2050+ road vkm and
energy, FOEN inventory CO2), see processors/freight_road_ots.py.

Road HDV (HDVH, HDVM): BFS GTS all registrations (CH + foreign) and all traffic types
  (domestic, import, export, transit). Foreign trucks are ~44 % of HDVH tkm.
  The Swiss-registered fleet, which the industry module needs, comes from BFS fleet
  data in freight_road_ots, not from tkm.

Rail: BFS T7.2.1, total tkm on the Swiss rail network.

IWW (Rhine navigation): BFS Rhine import/export tonnes (all flags).

Aviation (freight): residual share of total; calibrated to BAZL Swiss airport cargo.

Sources
-------
Road HDVH (HAV) + HDVM (LORRY):
  BFS GTS survey (annual million tkm, 1993-2024, all registrations).
  LORRY = rigid trucks ≥ 3.5 t → HDVM; HAV = articulated trucks → HDVH.
  # https://www.bfs.admin.ch/bfs/fr/home/statistiques/mobilite-transports/
  #   transport-marchandises/route/vehicules-lourds.html
  # file: ts-x-11.05-GTS-E26.csv

Road HDVL (light commercial vehicles < 3.5 t):
  EP2050+ ZERO-Basis scenario LCV VKM (Mio VKM, 1990-2060), converted with
  _HDVL_PAYLOAD_T_PER_VKM (ASTRA average load factor ~20 % of ~1 t capacity).
  # https://www.bfe.admin.ch/bfe/en/home/policy/energy-perspectives-2050-plus.html
  # file: EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12.xlsx
  #   sheet '03 Fahrleistung', ZERO-Basis scenario


Rail freight:
  BFS public transport statistics, table T7.2.1 row 12 (Chemins de fer, million tkm).
  Years 1990, 1995, 2000-2023; gaps 1991-1994 and 1996-1999 interpolated linearly.
  #Transports publics (trafic marchandises rail inclus) - séries chronologiques détaillées - 1990, 1995, 2000-2024 | Données - Tableau
  # https://www.bfs.admin.ch/asset/fr/36202624
  # file: tra_public_transport.xlsx, sheet T7_2_1

IWW (Rhine navigation):
  BFS Rhine imports/exports (million tonnes, 1955-2024) × _IWW_HAUL_KM.
  _IWW_HAUL_KM = 34 km is the navigable Swiss Rhine stretch (Port of Basel to
  German border), which reproduces the ~0.17 bn-tkm IWW share in Swiss statistics.
  # https://www.bfs.admin.ch/bfs/en/home/statistics/mobility-transport/goods-transport/air-water-pipeline.html
  # file: gr-e-11.05.04b.csv

Marine:
  Hardcoded 0 (Switzerland landlocked).

Aviation:
  Residual after all other modes to maintain _AVIATION_SHARE = 4.4 % of total.
  Calibrated to BAZL Swiss airport cargo (~500 kt/yr at ~2600 km avg haul ≈ 1.3 bn-tkm).
  #TODO : maybe check that the 4.4% makes sense and use other database for getting the aviation bn-tkm

Canton share (any canton in country_list):
  BFS GTS canton survey 2016-2020 average: share = tkm originating from the canton
  (column "Total") / Swiss total, e.g. 767 635 / 11 013 466 thousand tkm = 6.97 % for
  Vaud. This survey only covers Swiss-registered trucks; its canton share is applied
  to the territorial totals of all modes. IWW is 0 for cantons without access to the
  navigable Rhine (all except _IWW_CANTONS = BS, BL). Aviation is recomputed as the
  _AVIATION_SHARE residual of the canton subtotal.
  # https://www.bfs.admin.ch/bfs/fr/home/statistiques/mobilite-transports/transport-marchandises/route/vehicules-lourds.assetdetail.18924747.html
  # file: su-f-11.05-GTS16-20-P19.xlsx, sheet FR
"""

import os

import numpy as np
import openpyxl
import pandas as pd

from transition_compass_model._database.pre_processing.constants import CANTONS_NAME
from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions.freight_data import (
    read_gts_road_tkm,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HDVL_PAYLOAD_T_PER_VKM = (
    0.2  # ASTRA LCV avg load ≈ 20 % of 1 t capacity; see module docstring
)
_IWW_HAUL_KM = (
    34.0  # navigable Swiss Rhine (Port of Basel to German border); see module docstring
)
_AVIATION_SHARE = (
    0.044  # residual share calibrated to BAZL airport cargo data; see module docstring
)
# Cantons with Rhine ports (Basel-Kleinhüningen, Birsfelden, Muttenz-Au), by acronym
# so the names follow CANTONS_NAME
_IWW_CANTONS = {CANTONS_NAME["accronym_to_name"][a] for a in ("BS", "BL")}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "../data")

_MODES = ["HDVH", "HDVL", "HDVM", "IWW", "aviation", "marine", "rail"]


# ---------------------------------------------------------------------------
# Private readers
# ---------------------------------------------------------------------------
def road_tkm(years_ots: list) -> dict:
    """Return {mode: np.ndarray} territorial road tkm in million tkm, on years_ots.

    HDVH/HDVM from BFS GTS; HDVL = EP2050 LCV vkm x _HDVL_PAYLOAD_T_PER_VKM.
    Single definition, also used by freight_road_ots for the load factor.
    """
    gts = read_gts_road_tkm()
    return {
        "HDVH": _fill_series(gts["HDVH"], years_ots),
        "HDVM": _fill_series(gts["HDVM"], years_ots),
        "HDVL": _fill_series(_read_ep2050_lcv_vkm(), years_ots),
    }


def _read_ep2050_lcv_vkm():
    """Return pd.Series(year→million tkm) for HDVL using EP2050 LCV VKM × payload."""
    path = os.path.join(
        _DATA_DIR,
        "EP2050_sectors/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/"
        "EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12.xlsx",
    )
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["03 Fahrleistung"]
    rows = list(ws.iter_rows(min_row=20, max_row=350, values_only=True))
    header = rows[0]
    years_ep = [int(v) for v in header[4:] if isinstance(v, (int, float))]

    lcv_vkm = np.zeros(len(years_ep))
    in_lcv = False
    done = False
    for r in rows[1:]:
        if r[1] == "LCV" and not done:
            lcv_vkm += np.array(
                [
                    v if isinstance(v, (int, float)) else 0.0
                    for v in r[4 : 4 + len(years_ep)]
                ]
            )
            in_lcv = True
        elif in_lcv and r[1] != "LCV":
            done = True

    series = pd.Series(
        {yr: lcv_vkm[i] * _HDVL_PAYLOAD_T_PER_VKM for i, yr in enumerate(years_ep)}
    )
    return series


def _read_rail_tkm():
    """Return pd.Series(year→million tkm) for rail from BFS T7.2.1."""
    path = os.path.join(_DATA_DIR, "tra_public_transport.xlsx")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["T7_2_1"]
    row4 = list(ws.iter_rows(min_row=4, max_row=4, values_only=True))[0]
    year_cols = [(i, int(v)) for i, v in enumerate(row4) if isinstance(v, (int, float))]
    row12 = list(ws.iter_rows(min_row=12, max_row=12, values_only=True))[0]

    data = {yr: row12[i] for i, yr in year_cols if isinstance(row12[i], (int, float))}
    return pd.Series(data, dtype=float)


def _read_iww_tkm():
    """Return pd.Series(year→million tkm) for IWW (Rhine tonnes × haul factor)."""
    path = os.path.join(_DATA_DIR, "Freight/gr-e-11.05.04b.csv")
    df = pd.read_csv(path, names=["Year", "total_Mt", "rhine_Mt"], skiprows=1)
    df["Year"] = df["Year"].astype(int)
    df = df.set_index("Year")
    return df["rhine_Mt"] * _IWW_HAUL_KM


def _read_canton_shares() -> dict:
    """Return {canton name: share of Swiss freight tkm} from BFS GTS 2016-2020.

    Sheet 'FR': column A = origin canton acronym, column B = total tkm from that
    canton (1000 tkm); the 'Total' row is the Swiss total.
    """
    path = os.path.join(_DATA_DIR, "Freight/su-f-11.05-GTS16-20-P19.xlsx")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["FR"]
    acronym_to_name = CANTONS_NAME["accronym_to_name"]
    total_ch = None
    tkm = {}
    for row in ws.iter_rows(min_row=5, max_col=2, values_only=True):
        label, value = row
        if label == "Total":
            total_ch = float(value)
        elif label in acronym_to_name and label != "CH":
            tkm[acronym_to_name[label]] = float(value)
    wb.close()
    if total_ch is None:
        raise ValueError(f"'Total' row not found in {path}")
    return {canton: v / total_ch for canton, v in tkm.items()}


def _scale_to_canton(ch: dict, share: float, has_iww: bool, n_yr: int):
    """Scale Swiss tkm by mode to a canton. Return ({mode: tkm}, total tkm)."""
    canton = {}
    for m in _MODES:
        if m == "marine" or (m == "IWW" and not has_iww):
            canton[m] = np.zeros(n_yr)
        else:
            canton[m] = ch[m] * share
    # Recompute aviation as residual with the same share
    subtotal = sum(canton[m] for m in canton if m != "aviation")
    canton["aviation"] = subtotal * (_AVIATION_SHARE / (1.0 - _AVIATION_SHARE))
    return canton, subtotal + canton["aviation"]


def _fill_series(raw: pd.Series, years_ots: list) -> np.ndarray:
    """Reindex to years_ots, interpolate interior gaps, bfill/ffill ends."""
    s = raw.reindex(years_ots, fill_value=np.nan).astype(float)
    s = s.interpolate(method="index")
    s = s.bfill().ffill()
    return s.values


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(years_ots: list, country_list: list) -> dict:
    """Build freight_tkm and freight_modal-share DataMatrix for OTS.

    Parameters
    ----------
    years_ots : list of int
        Annual years from start (e.g. 1990) to base year (e.g. 2023).
    country_list : list of str
        Switzerland plus any cantons (e.g. ['Switzerland', 'Vaud', 'Fribourg']).
        Each canton is computed from Switzerland with its GTS share.

    Returns
    -------
    dict with keys 'freight_tkm' and 'freight_modal-share', each a DataMatrix.
    """
    n_yr = len(years_ots)

    # --- Load raw series and fill to OTS years ----------------------------
    ch = road_tkm(years_ots)
    ch["rail"] = _fill_series(_read_rail_tkm(), years_ots)
    ch["IWW"] = _fill_series(_read_iww_tkm(), years_ots)
    ch["marine"] = np.zeros(n_yr)

    # Aviation = residual to hold _AVIATION_SHARE of total
    subtotal = sum(ch[m] for m in ch)
    ch["aviation"] = subtotal * (_AVIATION_SHARE / (1.0 - _AVIATION_SHARE))
    total_ch = subtotal + ch["aviation"]

    # --- Cantons: scale by GTS canton share -------------------------------
    results = {"Switzerland": (ch, total_ch)}
    cantons = [c for c in country_list if c != "Switzerland"]
    if cantons:
        shares = _read_canton_shares()
        missing = [c for c in cantons if c not in shares]
        if missing:
            raise KeyError(f"No GTS freight share for: {missing}")
        for canton in cantons:
            results[canton] = _scale_to_canton(
                ch, shares[canton], canton in _IWW_CANTONS, n_yr
            )
    all_countries = list(results)

    # --- DataMatrix: freight_modal-share and freight_tkm -----------------
    dm_modal = DataMatrix(
        col_labels={
            "Country": all_countries,
            "Years": years_ots,
            "Variables": ["tra_freight_modal-share"],
            "Categories1": _MODES,
        },
        units={"tra_freight_modal-share": "%"},
    )
    dm_tkm = DataMatrix(
        col_labels={
            "Country": all_countries,
            "Years": years_ots,
            "Variables": ["tra_freight_tkm-total-demand"],
        },
        units={"tra_freight_tkm-total-demand": "bn-tkm"},
    )
    idx = dm_modal.idx
    idx_t = dm_tkm.idx
    for country, (modes, total) in results.items():
        for m in _MODES:
            dm_modal.array[idx[country], :, 0, idx[m]] = modes[m] / total
        dm_tkm.array[idx_t[country], :, 0] = total / 1e3

    return {"freight_tkm": dm_tkm, "freight_modal-share": dm_modal}


if __name__ == "__main__":
    from transition_compass_model._database.pre_processing.params import (
        country_list,
        years_ots,
    )

    DM = run(years_ots, country_list)
