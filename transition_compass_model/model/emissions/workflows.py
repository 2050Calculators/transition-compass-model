def put_together_emissions(DM_emi):

    dm_emi = DM_emi["transport"].copy()
    modules = ["buildings", "industry", "agriculture", "ammonia"]
    for m in modules:
        dm_emi.append(DM_emi[m].copy(), "Variables")

    # for captured / negative emissions, make sure that they are negative
    for cat in ["industry-captured-emissions", "ammonia-captured-emissions"]:
        arr_temp = dm_emi[:, :, cat, :].copy()
        arr_temp[arr_temp > 0] = -arr_temp[arr_temp > 0]
        dm_emi[:, :, cat, :] = arr_temp.copy()

    # deepen
    for v in dm_emi.col_labels["Variables"]:
        dm_emi.rename_col(v, "emissions_" + v, "Variables")
    dm_emi.deepen(based_on="Variables")
    dm_emi.switch_categories_order("Categories1", "Categories2")

    return dm_emi


def make_co2_equivalent(dm_emi):

    dm_out = dm_emi.copy()

    GWP_N2O = 265
    GWP_CH4 = 28

    dm_out[..., "N2O"] = dm_out[..., "N2O"] * GWP_N2O
    dm_out[..., "CH4"] = dm_out[..., "CH4"] * GWP_CH4

    dm_out.group_all("Categories2")

    dm_out.change_unit(old_unit="Mt", new_unit="MtCO2eq", factor=1, var="emissions")

    return dm_out
