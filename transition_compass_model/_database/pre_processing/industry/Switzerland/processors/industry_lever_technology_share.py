"""Swiss-specific technology shares for the industry module.

EU27 preprocessing assigns a constant 9% wet-kiln / 91% dry-kiln cement split
to all countries. Switzerland has operated only dry-process kilns since ~2000
(Holcim, HeidelbergMaterials Swiss plants are all modern dry-process). This
module overrides cement-wet-kiln → 0 and cement-dry-kiln → 1.0 for Switzerland.
"""


def _apply_ch_overrides(dm):
    """Apply Swiss-specific technology-share overrides in-place."""
    # Switzerland: 100% dry-process cement; wet-kilns phased out by ~2000
    dm[..., "cement-wet-kiln"] = 0.0
    dm[..., "cement-dry-kiln"] = 1.0


def make_ch_technology_share_ots(dm_eu27_ots):
    """Return a Switzerland OTS technology-share DM with Swiss-specific overrides.

    Starts from the EU27 OTS technology-share, renames country to Switzerland,
    then applies Swiss-specific corrections.
    """
    dm = dm_eu27_ots.filter({"Country": ["EU27"]})
    dm.rename_col("EU27", "Switzerland", "Country")
    _apply_ch_overrides(dm)
    return dm


def make_ch_technology_share_fts(dm_eu27_fts_l1):
    """Return a Switzerland FTS technology-share DM with Swiss-specific overrides.

    Starts from the EU27 FTS level-1 technology-share, renames country to
    Switzerland, then applies Swiss-specific corrections.
    """
    dm = dm_eu27_fts_l1.filter({"Country": ["EU27"]})
    dm.rename_col("EU27", "Switzerland", "Country")
    _apply_ch_overrides(dm)
    return dm
