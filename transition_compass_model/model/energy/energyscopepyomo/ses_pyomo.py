
# ses_pyomo.py — full translation of ses_main.mod to Pyomo (HiGHS / APPSI)
# Notes:
# - Derived sets are built inside Pyomo (do not duplicate them in JSON).
# - Uses APPSI HiGHS; no commercial solvers.
# - Defaults to minimizing TotalGWP; you can switch to "cost".

import json
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

# -----------------
# I/O helpers
# -----------------
def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def make_highs(time_limit=None, show_log=True):
    opt = Highs()
    opt.config.stream_solver = bool(show_log)
    if time_limit is not None:
        opt.config.time_limit = float(time_limit)
    if hasattr(opt, "highs_options"):
        opt.highs_options.update({"presolve": "on", "threads": 0})
    return opt

def attach(opt, m):
    opt.set_instance(m)

def solve(opt, m, warmstart=True):
    if warmstart and hasattr(opt, "warm_start"):
        opt.warm_start(m)
    return opt.solve(m)

def extract_results(m):
    out = {
        "obj": pyo.value(m.obj),
        "TotalGWP": pyo.value(m.TotalGWP),
        "TotalCost": pyo.value(m.TotalCost),
        "F_Mult": {i: pyo.value(m.F_Mult[i]) for i in m.TECHNOLOGIES},
    }
    return out

def _get2(P, name, i, j, default=0.0):
    """Lookup PARAMS[name] for a 2D key (i,j) supporting both:
       - flat dict with "i,j" keys
       - nested dict {i: {j: ...}}
       Also tolerates int/str for j (e.g., periods)."""
    D = P.get(name, {})
    if isinstance(D, dict):
        v = D.get(i)
        if isinstance(v, dict):
            return v.get(j, v.get(str(j), default))
    return D.get(f"{i},{j}", default)

def _get1(P, name, i, default=0.0):
    """Lookup PARAMS[name][i] (1D) with fallback to str(i)."""
    D = P.get(name, {})
    if isinstance(D, dict):
        return D.get(i, D.get(str(i), default))
    return default

# -----------------
# Model builder
# -----------------
def build_model(data, objective="gwp"):
    S = data.get("SETS", {})
    P = data.get("PARAMS", {})

    m = pyo.ConcreteModel()

    # ---------- Sets (base) ----------
    m.PERIODS = pyo.Set(initialize=S["PERIODS"], ordered=True)
    m.SECTORS = pyo.Set(initialize=S["SECTORS"])
    m.END_USES_INPUT = pyo.Set(initialize=S["END_USES_INPUT"])
    m.END_USES_CATEGORIES = pyo.Set(initialize=S["END_USES_CATEGORIES"])
    m.RESOURCES = pyo.Set(initialize=S["RESOURCES"])
    m.BIOFUELS = pyo.Set(within=m.RESOURCES, initialize=S["BIOFUELS"])
    m.EXPORT = pyo.Set(within=m.RESOURCES, initialize=S["EXPORT"])
    m.STORAGE_TECH = pyo.Set(initialize=S["STORAGE_TECH"])
    m.INFRASTRUCTURE = pyo.Set(initialize=S["INFRASTRUCTURE"])

    # END_USES_TYPES_OF_CATEGORY (indexed)
    eutoc_map = S["END_USES_TYPES_OF_CATEGORY"]
    m.END_USES_TYPES_OF_CATEGORY = pyo.Set(
        m.END_USES_CATEGORIES,
        initialize=lambda m, cat: eutoc_map.get(cat, [])
    )

    # END_USES_TYPES = union over categories
    def _init_eut(m):
        acc = set()
        for cat in m.END_USES_CATEGORIES:
            acc |= set(m.END_USES_TYPES_OF_CATEGORY[cat])
        return sorted(acc)
    m.END_USES_TYPES = pyo.Set(initialize=_init_eut)

    # TECHNOLOGIES_OF_END_USES_TYPE (indexed)
    teu_map = S["TECHNOLOGIES_OF_END_USES_TYPE"]
    m.TECHNOLOGIES_OF_END_USES_TYPE = pyo.Set(
        m.END_USES_TYPES,
        initialize=lambda m, ty: teu_map.get(ty, [])
    )

    # LAYERS := (RESOURCES diff BIOFUELS diff EXPORT) union END_USES_TYPES
    def _init_layers(m):
        base = set(m.RESOURCES) - set(m.BIOFUELS) - set(m.EXPORT)
        return sorted(base | set(m.END_USES_TYPES))
    m.LAYERS = pyo.Set(initialize=_init_layers)

    # TECHNOLOGIES := union of techs over end-use types plus storage plus infra
    def _init_techs(m):
        acc = set()
        for ty in m.END_USES_TYPES:
            acc |= set(m.TECHNOLOGIES_OF_END_USES_TYPE[ty])
        acc |= set(m.STORAGE_TECH) | set(m.INFRASTRUCTURE)
        return sorted(acc)
    m.TECHNOLOGIES = pyo.Set(initialize=_init_techs)

    # TECHNOLOGIES_OF_END_USES_CATEGORY {cat} within TECHNOLOGIES
    def _init_tecs_by_cat(m, cat):
        acc = set()
        for ty in m.END_USES_TYPES_OF_CATEGORY[cat]:
            acc |= set(m.TECHNOLOGIES_OF_END_USES_TYPE[ty])
        return sorted(acc & set(m.TECHNOLOGIES))
    m.TECHNOLOGIES_OF_END_USES_CATEGORY = pyo.Set(
        m.END_USES_CATEGORIES, within=m.TECHNOLOGIES, initialize=_init_tecs_by_cat
    )

    # Optional reporting subsets
    if "COGEN" in S:
        m.COGEN = pyo.Set(within=m.TECHNOLOGIES, initialize=S["COGEN"])
    if "BOILERS" in S:
        m.BOILERS = pyo.Set(within=m.TECHNOLOGIES, initialize=S["BOILERS"])

    # Combined indices
    m.RES_OR_TECH = pyo.Set(initialize=sorted(set(m.RESOURCES) | set(m.TECHNOLOGIES)))
    m.NON_STORAGE_X = pyo.Set(initialize=sorted((set(m.RESOURCES) | set(m.TECHNOLOGIES)) - set(m.STORAGE_TECH)))
    m.X_TECH_RES = pyo.Set(initialize=sorted((set(m.RESOURCES) | set(m.TECHNOLOGIES)) - set(m.STORAGE_TECH)))

    # ---------- Parameters ----------
    m.end_uses_demand_year = pyo.Param(
        m.END_USES_INPUT, m.SECTORS, mutable=True, default=0.0,
        initialize=lambda m, i, s: _get2(P, "end_uses_demand_year", i, s, 0.0)
    )
    if "end_uses_input" in P:
        m.end_uses_input = pyo.Param(m.END_USES_INPUT, mutable=True, default=0.0,
                                     initialize=P.get("end_uses_input", {}))
    else:
        def _eui_init(m, i):
            return sum(m.end_uses_demand_year[i, s] for s in m.SECTORS)
        m.end_uses_input = pyo.Param(m.END_USES_INPUT, initialize=_eui_init, mutable=True)

    m.i_rate = pyo.Param(mutable=True, initialize=P.get("i_rate", 0.04))
    m.share_mobility_public_min = pyo.Param(mutable=True, initialize=P.get("share_mobility_public_min", 0.0))
    m.share_mobility_public_max = pyo.Param(mutable=True, initialize=P.get("share_mobility_public_max", 1.0))
    m.share_freight_train_min = pyo.Param(mutable=True, initialize=P.get("share_freight_train_min", 0.0))
    m.share_freight_train_max = pyo.Param(mutable=True, initialize=P.get("share_freight_train_max", 1.0))
    m.share_heat_dhn_min = pyo.Param(mutable=True, initialize=P.get("share_heat_dhn_min", 0.0))
    m.share_heat_dhn_max = pyo.Param(mutable=True, initialize=P.get("share_heat_dhn_max", 1.0))

    m.t_op = pyo.Param(
        m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, t: _get1(P, "t_op", t, 0.0)
    )
    m.lighting_month = pyo.Param(
        m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, t: _get1(P, "lighting_month", t, 0.0)
    )
    m.heating_month = pyo.Param(
        m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, t: _get1(P, "heating_month", t, 0.0)
    )

    if "total_time" in P:
        m.total_time = pyo.Param(mutable=True, default=P.get("total_time", 8760.0))
    else:
        # compute from t_op
        def _tt_init(m):
            return sum(m.t_op[t] for t in m.PERIODS)
        m.total_time = pyo.Param(initialize=_tt_init, mutable=True)

    # Conversion map f: (RESOURCES ∪ TECHNOLOGIES \ STORAGE_TECH) × LAYERS
    m.layers_in_out = pyo.Param(
        m.NON_STORAGE_X, m.LAYERS, mutable=True, default=0.0,
        initialize=lambda m, x, l: _get2(P, "layers_in_out", x, l, 0.0)
    )
    # Build-time positive contributors into each end-use layer (to avoid symbolic boolean tests)
    def _pos_providers_init(m, i):
        providers = []
        for j in m.NON_STORAGE_X:
            # Check the actual Pyomo parameter, not the raw data
            if pyo.value(m.layers_in_out[j, i]) > 0.0:
                providers.append(j)
        return providers

    m.POS_PROVIDERS = pyo.Set(m.END_USES_TYPES, initialize=_pos_providers_init)

    # Technology attributes
    m.ref_size  = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("ref_size", {}))
    m.c_inv     = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("c_inv", {}))
    m.c_maint   = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("c_maint", {}))
    m.lifetime  = pyo.Param(m.TECHNOLOGIES, mutable=True, default=1.0, initialize=P.get("lifetime", {}))
    m.f_max     = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("f_max", {}))
    m.f_min     = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("f_min", {}))
    m.fmax_perc = pyo.Param(m.TECHNOLOGIES, mutable=True, default=1.0, initialize=P.get("fmax_perc", {}))
    m.fmin_perc = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("fmin_perc", {}))
    # capacity factors by (tech, period)
    m.c_p_t = pyo.Param(
        m.TECHNOLOGIES, m.PERIODS, mutable=True, default=1.0,
        initialize=lambda m, i, t: _get2(P, "c_p_t", i, t, 1.0)
    )
    m.c_p       = pyo.Param(m.TECHNOLOGIES, mutable=True, default=1.0, initialize=P.get("c_p", {}))
    m.gwp_constr_param = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("gwp_constr", {}))

    # Resource attributes
    # resource operating costs by (resource, period)
    m.c_op = pyo.Param(
        m.RESOURCES, m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, r, t: _get2(P, "c_op", r, t, 0.0)
    )
    m.avail = pyo.Param(m.RESOURCES, mutable=True, default=0.0, initialize=P.get("avail", {}))
    m.gwp_op_param = pyo.Param(m.RESOURCES, mutable=True, default=0.0, initialize=P.get("gwp_op", {}))

    # Storage attributes
    m.storage_eff_in = pyo.Param(
        m.STORAGE_TECH, m.LAYERS, mutable=True, default=0.0,
        initialize=lambda m, i, l: _get2(P, "storage_eff_in", i, l, 0.0)
    )
    m.storage_eff_out = pyo.Param(
        m.STORAGE_TECH, m.LAYERS, mutable=True, default=0.0,
        initialize=lambda m, i, l: _get2(P, "storage_eff_out", i, l, 0.0)
    )

    # Losses & peaks
    m.loss_coeff = pyo.Param(m.END_USES_TYPES, mutable=True, default=0.0, initialize=P.get("loss_coeff", {}))
    m.peak_dhn_factor = pyo.Param(mutable=True, default=P.get("peak_dhn_factor", 0.0))

    # Annuity factor
    m.tau = pyo.Expression(
        m.TECHNOLOGIES,
        rule=lambda m,i: (m.i_rate * (1 + m.i_rate)**m.lifetime[i]) / ((1 + m.i_rate)**m.lifetime[i] - 1.0)
    )

    # ---------- Variables ----------
    m.End_Uses = pyo.Var(m.LAYERS, m.PERIODS, domain=pyo.NonNegativeReals)
    m.Number_Of_Units = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeIntegers)
    m.F_Mult = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.F_Mult_t = pyo.Var(m.RES_OR_TECH, m.PERIODS, domain=pyo.NonNegativeReals)

    m.C_inv = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.C_maint = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.C_op = pyo.Var(m.RESOURCES, domain=pyo.NonNegativeReals)

    m.Storage_In = pyo.Var(m.STORAGE_TECH, m.LAYERS, m.PERIODS, domain=pyo.NonNegativeReals)
    m.Storage_Out = pyo.Var(m.STORAGE_TECH, m.LAYERS, m.PERIODS, domain=pyo.NonNegativeReals)

    m.Share_Mobility_Public = pyo.Var(bounds=(pyo.value(m.share_mobility_public_min), pyo.value(m.share_mobility_public_max)))
    m.Share_Freight_Train   = pyo.Var(bounds=(pyo.value(m.share_freight_train_min), pyo.value(m.share_freight_train_max)))
    m.Share_Heat_Dhn        = pyo.Var(bounds=(pyo.value(m.share_heat_dhn_min), pyo.value(m.share_heat_dhn_max)))

    m.Y_Solar_Backup = pyo.Var(m.TECHNOLOGIES, domain=pyo.Binary)
    m.Losses = pyo.Var(m.END_USES_TYPES, m.PERIODS, domain=pyo.NonNegativeReals)

    m.GWP_constr = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.GWP_op = pyo.Var(m.RESOURCES, domain=pyo.NonNegativeReals)
    m.TotalGWP = pyo.Var(domain=pyo.NonNegativeReals)
    m.TotalCost = pyo.Var(domain=pyo.NonNegativeReals)

    # Aux for solar backup linearization (includes DEC_SOLAR; constraints will Skip there)
    m.X_Solar_Backup_Aux = pyo.Var(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"],
        m.PERIODS, domain=pyo.NonNegativeReals
    )

    # immediately fix DEC_SOLAR entries to 0 (so they behave as if they don't exist)
    if "DEC_SOLAR" in m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"]:
        for t in m.PERIODS:
            m.X_Solar_Backup_Aux["DEC_SOLAR", t].fix(0)

    # Binaries for storage no-transfer (for Eq. 1.17)
    m.Y_Sto_In  = pyo.Var(m.STORAGE_TECH, m.PERIODS, domain=pyo.Binary)
    m.Y_Sto_Out = pyo.Var(m.STORAGE_TECH, m.PERIODS, domain=pyo.Binary)

    # Max_Heat_Demand_DHN (scalar var)
    m.Max_Heat_Demand_DHN = pyo.Var(domain=pyo.NonNegativeReals)

    # ---------- Constraints ----------
    # [Figure 1.4] End-uses demand rules
    def end_uses_rule(m, l, t):
        if l == "ELECTRICITY":
            return m.End_Uses[l, t] == ((m.end_uses_input[l] / m.total_time) +
                                        (m.end_uses_input["LIGHTING"] * m.lighting_month[t] / m.t_op[t])) + m.Losses[l, t]
        elif l == "HEAT_LOW_T_DHN":
            return m.End_Uses[l, t] == ((m.end_uses_input["HEAT_LOW_T_HW"] / m.total_time) +
                                        (m.end_uses_input["HEAT_LOW_T_SH"] * m.heating_month[t] / m.t_op[t])) * m.Share_Heat_Dhn + m.Losses[l, t]
        elif l == "HEAT_LOW_T_DECEN":
            return m.End_Uses[l, t] == ((m.end_uses_input["HEAT_LOW_T_HW"] / m.total_time) +
                                        (m.end_uses_input["HEAT_LOW_T_SH"] * m.heating_month[t] / m.t_op[t])) * (1 - m.Share_Heat_Dhn)
        elif l == "MOB_PUBLIC":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_PASSENGER"] / m.total_time) * m.Share_Mobility_Public
        elif l == "MOB_PRIVATE":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_PASSENGER"] / m.total_time) * (1 - m.Share_Mobility_Public)
        elif l == "MOB_FREIGHT_RAIL":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_FREIGHT"] / m.total_time) * m.Share_Freight_Train
        elif l == "MOB_FREIGHT_ROAD":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_FREIGHT"] / m.total_time) * (1 - m.Share_Freight_Train)
        elif l == "HEAT_HIGH_T":
            return m.End_Uses[l, t] == m.end_uses_input[l] / m.total_time
        else:
            return m.End_Uses[l, t] == 0
    m.end_uses_t = pyo.Constraint(m.LAYERS, m.PERIODS, rule=end_uses_rule)

    # [Eq. 1.7] Number of units (TECHNOLOGIES \ INFRASTRUCTURE)
    m.number_of_units = pyo.Constraint(
        m.TECHNOLOGIES - m.INFRASTRUCTURE, rule=lambda m,i: m.Number_Of_Units[i] == m.F_Mult[i] / m.ref_size[i]
    )

    # [Eq. 1.6] Size limits
    m.size_limit = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m,i: pyo.inequality(m.f_min[i], m.F_Mult[i], m.f_max[i])
    )

    # [Eq. 1.8] Period capacity factor
    m.capacity_factor_t = pyo.Constraint(
        m.TECHNOLOGIES, m.PERIODS, rule=lambda m,i,t: m.F_Mult_t[i, t] <= m.F_Mult[i] * m.c_p_t[i, t]
    )

    # [Eq. 1.9] Annual capacity factor
    m.capacity_factor = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m,i: sum(m.F_Mult_t[i, t] * m.t_op[t] for t in m.PERIODS) <= m.F_Mult[i] * m.c_p[i] * m.total_time
    )

    # Linearization of Eq. 1.19
    # Decentralized heat operating strategy (linearized)
    m.op_strategy_decen_1_linear = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m,i,t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
            m.F_Mult_t[i, t] + m.X_Solar_Backup_Aux[i, t] >=
            sum(m.F_Mult_t[i, t2] * m.t_op[t2] for t2 in m.PERIODS) *
            ((m.end_uses_input["HEAT_LOW_T_HW"] / m.total_time + m.end_uses_input["HEAT_LOW_T_SH"] * m.heating_month[t] / m.t_op[t]) /
             (m.end_uses_input["HEAT_LOW_T_HW"] + m.end_uses_input["HEAT_LOW_T_SH"]))
        )
    )
    m.op_strategy_decen_1_linear_1 = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m,i,t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
            m.X_Solar_Backup_Aux[i, t] <= m.f_max["DEC_SOLAR"] * m.Y_Solar_Backup[i]
        )
    )
    m.op_strategy_decen_1_linear_2 = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m,i,t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
            m.X_Solar_Backup_Aux[i, t] <= m.F_Mult_t["DEC_SOLAR", t]
        )
    )
    m.op_strategy_decen_1_linear_3 = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m,i,t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
            m.X_Solar_Backup_Aux[i, t] >= m.F_Mult_t["DEC_SOLAR", t] - (1 - m.Y_Solar_Backup[i]) * m.f_max["DEC_SOLAR"]
        )
    )

    # [Eq. 1.20] ]Only one backup
    m.op_strategy_decen_2 = pyo.Constraint(expr = sum(m.Y_Solar_Backup[i] for i in m.TECHNOLOGIES) <= 1)

    ## Layers

    # [Eq. 1.13] Layer balance with storage
    m.layer_balance = pyo.Constraint(
        m.LAYERS, m.PERIODS,
        rule=lambda m,l,t: 0 ==
            (sum(m.layers_in_out[x, l] * m.F_Mult_t[x, t] for x in m.NON_STORAGE_X)) +
            (sum(m.Storage_Out[j, l, t] - m.Storage_In[j, l, t] for j in m.STORAGE_TECH)) -
            m.End_Uses[l, t]
    )

    # [Eq. 1.12] Resources availability
    m.resource_availability = pyo.Constraint(
        m.RESOURCES, rule=lambda m,i: sum(m.F_Mult_t[i, t] * m.t_op[t] for t in m.PERIODS) <= m.avail[i]
    )

    # [Eq. 1.15-1.16] Each storage technology can have input/output only to certain layers.
    # If incompatible then the variable is set to 0 ceil (x) operator rounds a number to the highest nearest integer.
    # Storage compatibility: if eff==0, var == 0 (ceil trick equivalent)
    # If eff = 0 ⇒ var == 0; if eff = 1 ⇒ 0 == 0 (vacuous)
    m.storage_layer_in = pyo.Constraint(
        m.STORAGE_TECH, m.LAYERS, m.PERIODS,
        rule=lambda m, i, l, t: m.Storage_In[i, l, t] * (1 - m.storage_eff_in[i, l]) == 0
    )
    m.storage_layer_out = pyo.Constraint(
        m.STORAGE_TECH, m.LAYERS, m.PERIODS,
        rule=lambda m, i, l, t: m.Storage_Out[i, l, t] * (1 - m.storage_eff_out[i, l]) == 0
    )

    # Linearization of [Eq. 1.17]
    # Storage no-transfer linearization
    m.storage_no_transfer_1 = pyo.Constraint(
        m.STORAGE_TECH, m.PERIODS,
        rule=lambda m,i,t: (sum(m.Storage_In[i, l, t] * m.storage_eff_in[i, l] for l in m.LAYERS if pyo.value(m.storage_eff_in[i, l]) > 0)
                            * m.t_op[t] / m.f_max[i]) <= m.Y_Sto_In[i, t]
    )
    m.storage_no_transfer_2 = pyo.Constraint(
        m.STORAGE_TECH, m.PERIODS,
        rule=lambda m,i,t: (sum(m.Storage_Out[i, l, t] / m.storage_eff_out[i, l] for l in m.LAYERS if pyo.value(m.storage_eff_out[i, l]) > 0)
                            * m.t_op[t] / m.f_max[i]) <= m.Y_Sto_Out[i, t]
    )
    m.storage_no_transfer_3 = pyo.Constraint(
        m.STORAGE_TECH, m.PERIODS,
        rule=lambda m,i,t: m.Y_Sto_In[i, t] + m.Y_Sto_Out[i, t] <= 1
    )

    # [Eq. 1.14] Storage level with wrap-around
    def storage_level_rule(m, i, t):
        prev_t = m.PERIODS.last() if t == m.PERIODS.first() else m.PERIODS.prev(t)
        inflow = sum(
            m.Storage_In[i, l, t] * m.storage_eff_in[i, l]
            for l in m.LAYERS
            if pyo.value(m.storage_eff_in[i, l]) > 0
        )
        outflow = sum(
            m.Storage_Out[i, l, t] / m.storage_eff_out[i, l]
            for l in m.LAYERS
            if pyo.value(m.storage_eff_out[i, l]) > 0
        )
        return m.F_Mult_t[i, t] == m.F_Mult_t[i, prev_t] + (inflow - outflow) * m.t_op[t]

    m.storage_level = pyo.Constraint(m.STORAGE_TECH, m.PERIODS, rule=storage_level_rule)

    # [Eq. 1.18] Network losses (pre-filtered contributors)
    m.network_losses = pyo.Constraint(
        m.END_USES_TYPES, m.PERIODS,
        rule=lambda m,i,t: m.Losses[i, t] == (
            sum(m.layers_in_out[j, i] * m.F_Mult_t[j, t] for j in m.POS_PROVIDERS[i])
        ) * m.loss_coeff[i]
    )

    # [Eq 1.22] f_max_perc / f_min_perc per end-use type
    m.f_max_perc = pyo.Constraint(
        m.END_USES_TYPES, m.TECHNOLOGIES,
        rule=lambda m,i,j: pyo.Constraint.Skip
            if (j not in set(m.TECHNOLOGIES_OF_END_USES_TYPE[i]))
            else (sum(m.F_Mult_t[j, t] * m.t_op[t] for t in m.PERIODS)
                  <= m.fmax_perc[j] * sum(m.F_Mult_t[j2, t2] * m.t_op[t2]
                                           for j2 in m.TECHNOLOGIES_OF_END_USES_TYPE[i] for t2 in m.PERIODS))
    )
    m.f_min_perc = pyo.Constraint(
        m.END_USES_TYPES, m.TECHNOLOGIES,
        rule=lambda m,i,j: pyo.Constraint.Skip
            if (j not in set(m.TECHNOLOGIES_OF_END_USES_TYPE[i]))
            else (sum(m.F_Mult_t[j, t] * m.t_op[t] for t in m.PERIODS)
                  >= m.fmin_perc[j] * sum(m.F_Mult_t[j2, t2] * m.t_op[t2]
                                           for j2 in m.TECHNOLOGIES_OF_END_USES_TYPE[i] for t2 in m.PERIODS))
    )

    # [Eq. 1.24] Seasonal storage in hydro dams — rule-based with Skip
    def _hydro_cap_rule(m):
        if ("PUMPED_HYDRO" in m.TECHNOLOGIES) and ("NEW_HYDRO_DAM" in m.TECHNOLOGIES):
            return m.F_Mult["PUMPED_HYDRO"] <= m.f_max["PUMPED_HYDRO"] * (
                (m.F_Mult["NEW_HYDRO_DAM"] - m.f_min["NEW_HYDRO_DAM"]) /
                (m.f_max["NEW_HYDRO_DAM"] - m.f_min["NEW_HYDRO_DAM"])
            )
        return pyo.Constraint.Skip
    m.storage_level_hydro_dams = pyo.Constraint(rule=_hydro_cap_rule)

    # [Eq. 1.25] Hydro dams can only shift production — rule-based with Skip ------   !  HERE  !
    m.hydro_dams_shift = pyo.Constraint(
        m.PERIODS,
        rule=lambda m, t: m.Storage_In["PUMPED_HYDRO", "ELECTRICITY", t]
                          <= m.F_Mult_t["HYDRO_DAM", t] + m.F_Mult_t["NEW_HYDRO_DAM", t]
    )

    # [Eq. 1.26] DHN: assigning a cost to the network
    # Note that in Moret (2017), page 26, there is a ">=" sign instead of an "=". The two formulations are equivalent as long as the problem minimises cost and the DHN has a cost > 0
    m.extra_dhn = pyo.Constraint(
        rule=lambda m: m.F_Mult["DHN"] ==
                       sum(m.F_Mult[j] for j in m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DHN"])
    )

    # [Eq. 1.27] Calculation of max heat demand in DHN
    m.max_dhn_heat_demand = pyo.Constraint(
        m.PERIODS,
        rule=lambda m, t: m.Max_Heat_Demand_DHN >= m.End_Uses["HEAT_LOW_T_DHN", t]
    )

    # peak_dhn
    m.peak_dhn = pyo.Constraint(
        rule=lambda m: sum(m.F_Mult[j] for j in m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DHN"])
                       >= m.peak_dhn_factor * m.Max_Heat_Demand_DHN
    )

    # [Eq. 1.28] 9.4 BCHF is the extra investment needed if there is a big deployment of stochastic renewables
    m.extra_grid = pyo.Constraint(
        rule=lambda m: m.F_Mult["GRID"] ==
                       1 + (9400 / m.c_inv["GRID"]) * (m.F_Mult["WIND"] + m.F_Mult["PV"]) /
                       (m.f_max["WIND"] + m.f_max["PV"])
    )

    # [Eq. 1.29] Power2Gas investment cost is calculated on the max size of the two units
    m.extra_power2gas_1 = pyo.Constraint(rule=lambda m: m.F_Mult["POWER2GAS_3"] >= m.F_Mult["POWER2GAS_1"])
    m.extra_power2gas_2 = pyo.Constraint(rule=lambda m: m.F_Mult["POWER2GAS_3"] >= m.F_Mult["POWER2GAS_2"])

    # [Eq. 1.23] Operating strategy in private mobility (to make model more realistic)
    m.op_strategy_mob_private = pyo.Constraint(
        m.TECHNOLOGIES, m.PERIODS,
        rule=lambda m, i, t: pyo.Constraint.Skip
        if i not in (set(m.TECHNOLOGIES_OF_END_USES_CATEGORY["MOBILITY_PASSENGER"])
                     | set(m.TECHNOLOGIES_OF_END_USES_CATEGORY["MOBILITY_FREIGHT"]))
        else m.F_Mult_t[i, t] >= sum(m.F_Mult_t[i, t2] * m.t_op[t2] / m.total_time for t2 in m.PERIODS)
    )

    # Energy efficiency is a fixed cost
    m.extra_efficiency = pyo.Constraint(
        rule=lambda m: m.F_Mult["EFFICIENCY"] == 1 / (1 + m.i_rate)
    )

    # ----- Cost -----
    # [Eq. 1.3] Investment cost of each technology
    m.investment_cost_calc = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m,i: m.C_inv[i] == m.c_inv[i] * m.F_Mult[i]
    )
    # [Eq. 1.4] O&M cost of each technology
    m.main_cost_calc = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m,i: m.C_maint[i] == m.c_maint[i] * m.F_Mult[i]
    )
    # [Eq. 1.10] Total cost of each resource
    m.op_cost_calc = pyo.Constraint(
        m.RESOURCES, rule=lambda m,i: m.C_op[i] == sum(m.c_op[i, t] * m.F_Mult_t[i, t] * m.t_op[t] for t in m.PERIODS)
    )
    # [Eq. 1.1]
    m.totalcost_cal = pyo.Constraint(
        expr = m.TotalCost == sum(m.tau[i] * m.C_inv[i] + m.C_maint[i] for i in m.TECHNOLOGIES) + sum(m.C_op[j] for j in m.RESOURCES)
    )

    # ----- Emissions -----
    m.gwp_constr_calc = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m,i: m.GWP_constr[i] == m.gwp_constr_param[i] * m.F_Mult[i]
    )
    m.gwp_op_calc = pyo.Constraint(
        m.RESOURCES, rule=lambda m,i: m.GWP_op[i] == m.gwp_op_param[i] * sum(m.t_op[t] * m.F_Mult_t[i, t] for t in m.PERIODS)
    )
    m.totalGWP_calc = pyo.Constraint(
        expr = m.TotalGWP == sum(m.GWP_constr[i] / m.lifetime[i] for i in m.TECHNOLOGIES) + sum(m.GWP_op[j] for j in m.RESOURCES)
    )

    # ---------- Objective ----------
    if objective == "gwp":
        m.obj = pyo.Objective(expr=m.TotalGWP, sense=pyo.minimize)
    elif objective == "cost":
        m.obj = pyo.Objective(expr=m.TotalCost, sense=pyo.minimize)
    else:
        raise ValueError("objective must be 'gwp' or 'cost'")

    # helper to switch objective post-build
    def set_objective(mdl, which="gwp"):
        if hasattr(mdl, "obj"):
            mdl.del_component(mdl.obj)
        if which == "gwp":
            mdl.obj = pyo.Objective(expr=mdl.TotalGWP, sense=pyo.minimize)
        elif which == "cost":
            mdl.obj = pyo.Objective(expr=mdl.TotalCost, sense=pyo.minimize)
        else:
            raise ValueError("which must be 'gwp' or 'cost'")
    m.set_objective = set_objective

    return m

def build_model_structure(data):
    S = data.get("SETS", {})
    P = data.get("PARAMS", {})

    m = pyo.ConcreteModel()

    # ---------- Sets (base) ----------
    m.PERIODS = pyo.Set(initialize=S["PERIODS"], ordered=True)
    m.SECTORS = pyo.Set(initialize=S["SECTORS"])
    m.END_USES_INPUT = pyo.Set(initialize=S["END_USES_INPUT"])
    m.END_USES_CATEGORIES = pyo.Set(initialize=S["END_USES_CATEGORIES"])
    m.RESOURCES = pyo.Set(initialize=S["RESOURCES"])
    m.BIOFUELS = pyo.Set(within=m.RESOURCES, initialize=S["BIOFUELS"])
    m.EXPORT = pyo.Set(within=m.RESOURCES, initialize=S["EXPORT"])
    m.STORAGE_TECH = pyo.Set(initialize=S["STORAGE_TECH"])
    m.INFRASTRUCTURE = pyo.Set(initialize=S["INFRASTRUCTURE"])

    # END_USES_TYPES_OF_CATEGORY (indexed)
    eutoc_map = S["END_USES_TYPES_OF_CATEGORY"]
    m.END_USES_TYPES_OF_CATEGORY = pyo.Set(
        m.END_USES_CATEGORIES,
        initialize=lambda m, cat: eutoc_map.get(cat, [])
    )

    # END_USES_TYPES = union over categories
    def _init_eut(m):
        acc = set()
        for cat in m.END_USES_CATEGORIES:
            acc |= set(m.END_USES_TYPES_OF_CATEGORY[cat])
        return sorted(acc)
    m.END_USES_TYPES = pyo.Set(initialize=_init_eut)

    # TECHNOLOGIES_OF_END_USES_TYPE (indexed)
    teu_map = S["TECHNOLOGIES_OF_END_USES_TYPE"]
    m.TECHNOLOGIES_OF_END_USES_TYPE = pyo.Set(
        m.END_USES_TYPES,
        initialize=lambda m, ty: teu_map.get(ty, [])
    )

    # LAYERS := (RESOURCES diff BIOFUELS diff EXPORT) union END_USES_TYPES
    def _init_layers(m):
        base = set(m.RESOURCES) - set(m.BIOFUELS) - set(m.EXPORT)
        return sorted(base | set(m.END_USES_TYPES))
    m.LAYERS = pyo.Set(initialize=_init_layers)

    # TECHNOLOGIES := union of techs over end-use types plus storage plus infra
    def _init_techs(m):
        acc = set()
        for ty in m.END_USES_TYPES:
            acc |= set(m.TECHNOLOGIES_OF_END_USES_TYPE[ty])
        acc |= set(m.STORAGE_TECH) | set(m.INFRASTRUCTURE)
        return sorted(acc)
    m.TECHNOLOGIES = pyo.Set(initialize=_init_techs)

    # TECHNOLOGIES_OF_END_USES_CATEGORY {cat} within TECHNOLOGIES
    def _init_tecs_by_cat(m, cat):
        acc = set()
        for ty in m.END_USES_TYPES_OF_CATEGORY[cat]:
            acc |= set(m.TECHNOLOGIES_OF_END_USES_TYPE[ty])
        return sorted(acc & set(m.TECHNOLOGIES))
    m.TECHNOLOGIES_OF_END_USES_CATEGORY = pyo.Set(
        m.END_USES_CATEGORIES, within=m.TECHNOLOGIES, initialize=_init_tecs_by_cat
    )

    # Optional reporting subsets
    if "COGEN" in S:
        m.COGEN = pyo.Set(within=m.TECHNOLOGIES, initialize=S["COGEN"])
    if "BOILERS" in S:
        m.BOILERS = pyo.Set(within=m.TECHNOLOGIES, initialize=S["BOILERS"])

    # Combined indices
    m.RES_OR_TECH = pyo.Set(initialize=sorted(set(m.RESOURCES) | set(m.TECHNOLOGIES)))
    m.NON_STORAGE_X = pyo.Set(initialize=sorted((set(m.RESOURCES) | set(m.TECHNOLOGIES)) - set(m.STORAGE_TECH)))
    m.X_TECH_RES = pyo.Set(initialize=sorted((set(m.RESOURCES) | set(m.TECHNOLOGIES)) - set(m.STORAGE_TECH)))

    # ---------- Parameters ----------
    m.end_uses_demand_year = pyo.Param(
        m.END_USES_INPUT, m.SECTORS, mutable=True, default=0.0,
        initialize=lambda m, i, s: _get2(P, "end_uses_demand_year", i, s, 0.0)
    )

    m.i_rate = pyo.Param(mutable=True, initialize=P.get("i_rate", 0.04))
    m.share_mobility_public_min = pyo.Param(mutable=True, initialize=P.get("share_mobility_public_min", 0.0))
    m.share_mobility_public_max = pyo.Param(mutable=True, initialize=P.get("share_mobility_public_max", 1.0))
    m.share_freight_train_min = pyo.Param(mutable=True, initialize=P.get("share_freight_train_min", 0.0))
    m.share_freight_train_max = pyo.Param(mutable=True, initialize=P.get("share_freight_train_max", 1.0))
    m.share_heat_dhn_min = pyo.Param(mutable=True, initialize=P.get("share_heat_dhn_min", 0.0))
    m.share_heat_dhn_max = pyo.Param(mutable=True, initialize=P.get("share_heat_dhn_max", 1.0))

    m.t_op = pyo.Param(
        m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, t: _get1(P, "t_op", t, 0.0)
    )
    m.lighting_month = pyo.Param(
        m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, t: _get1(P, "lighting_month", t, 0.0)
    )
    m.heating_month = pyo.Param(
        m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, t: _get1(P, "heating_month", t, 0.0)
    )

    # Conversion map f: (RESOURCES ∪ TECHNOLOGIES \ STORAGE_TECH) × LAYERS
    m.layers_in_out = pyo.Param(
        m.NON_STORAGE_X, m.LAYERS, mutable=True, default=0.0,
        initialize=lambda m, x, l: _get2(P, "layers_in_out", x, l, 0.0)
    )
    # Build-time positive contributors into each end-use layer (to avoid symbolic boolean tests)
    def _pos_providers_init(m, i):
        providers = []
        for j in m.NON_STORAGE_X:
            # Check the actual Pyomo parameter, not the raw data
            if pyo.value(m.layers_in_out[j, i]) > 0.0:
                providers.append(j)
        return providers

    # Technology attributes
    m.ref_size  = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("ref_size", {}))
    m.c_inv     = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("c_inv", {}))
    m.c_maint   = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("c_maint", {}))
    m.lifetime  = pyo.Param(m.TECHNOLOGIES, mutable=True, default=1.0, initialize=P.get("lifetime", {}))
    m.f_max     = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("f_max", {}))
    m.f_min     = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("f_min", {}))
    m.fmax_perc = pyo.Param(m.TECHNOLOGIES, mutable=True, default=1.0, initialize=P.get("fmax_perc", {}))
    m.fmin_perc = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("fmin_perc", {}))
    # capacity factors by (tech, period)
    m.c_p_t = pyo.Param(
        m.TECHNOLOGIES, m.PERIODS, mutable=True, default=1.0,
        initialize=lambda m, i, t: _get2(P, "c_p_t", i, t, 1.0)
    )
    m.c_p       = pyo.Param(m.TECHNOLOGIES, mutable=True, default=1.0, initialize=P.get("c_p", {}))
    m.gwp_constr_param = pyo.Param(m.TECHNOLOGIES, mutable=True, default=0.0, initialize=P.get("gwp_constr", {}))

    # Resource attributes
    # resource operating costs by (resource, period)
    m.c_op = pyo.Param(
        m.RESOURCES, m.PERIODS, mutable=True, default=0.0,
        initialize=lambda m, r, t: _get2(P, "c_op", r, t, 0.0)
    )
    m.avail = pyo.Param(m.RESOURCES, mutable=True, default=0.0, initialize=P.get("avail", {}))
    m.gwp_op_param = pyo.Param(m.RESOURCES, mutable=True, default=0.0, initialize=P.get("gwp_op", {}))

    # Storage attributes
    m.storage_eff_in = pyo.Param(
        m.STORAGE_TECH, m.LAYERS, mutable=True, default=0.0,
        initialize=lambda m, i, l: _get2(P, "storage_eff_in", i, l, 0.0)
    )
    m.storage_eff_out = pyo.Param(
        m.STORAGE_TECH, m.LAYERS, mutable=True, default=0.0,
        initialize=lambda m, i, l: _get2(P, "storage_eff_out", i, l, 0.0)
    )

    # Losses & peaks
    m.loss_coeff = pyo.Param(m.END_USES_TYPES, mutable=True, default=0.0, initialize=P.get("loss_coeff", {}))
    m.peak_dhn_factor = pyo.Param(mutable=True, default=P.get("peak_dhn_factor", 0.0))

    # Annuity factor
    m.tau = pyo.Expression(
        m.TECHNOLOGIES,
        rule=lambda m,i: (m.i_rate * (1 + m.i_rate)**m.lifetime[i]) / ((1 + m.i_rate)**m.lifetime[i] - 1.0)
    )

    # ---------- Variables ----------
    m.End_Uses = pyo.Var(m.LAYERS, m.PERIODS, domain=pyo.NonNegativeReals)
    m.Number_Of_Units = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeIntegers)
    m.F_Mult = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.F_Mult_t = pyo.Var(m.RES_OR_TECH, m.PERIODS, domain=pyo.NonNegativeReals)

    m.C_inv = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.C_maint = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.C_op = pyo.Var(m.RESOURCES, domain=pyo.NonNegativeReals)

    m.Storage_In = pyo.Var(m.STORAGE_TECH, m.LAYERS, m.PERIODS, domain=pyo.NonNegativeReals)
    m.Storage_Out = pyo.Var(m.STORAGE_TECH, m.LAYERS, m.PERIODS, domain=pyo.NonNegativeReals)


    m.Y_Solar_Backup = pyo.Var(m.TECHNOLOGIES, domain=pyo.Binary)
    m.Losses = pyo.Var(m.END_USES_TYPES, m.PERIODS, domain=pyo.NonNegativeReals)

    m.GWP_constr = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)
    m.GWP_op = pyo.Var(m.RESOURCES, domain=pyo.NonNegativeReals)
    m.TotalGWP = pyo.Var(domain=pyo.NonNegativeReals)
    m.TotalCost = pyo.Var(domain=pyo.NonNegativeReals)

    # Aux for solar backup linearization (includes DEC_SOLAR; constraints will Skip there)
    m.X_Solar_Backup_Aux = pyo.Var(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"],
        m.PERIODS, domain=pyo.NonNegativeReals
    )

    # immediately fix DEC_SOLAR entries to 0 (so they behave as if they don't exist)
    if "DEC_SOLAR" in m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"]:
        for t in m.PERIODS:
            m.X_Solar_Backup_Aux["DEC_SOLAR", t].fix(0)

    # Binaries for storage no-transfer (for Eq. 1.17)
    m.Y_Sto_In  = pyo.Var(m.STORAGE_TECH, m.PERIODS, domain=pyo.Binary)
    m.Y_Sto_Out = pyo.Var(m.STORAGE_TECH, m.PERIODS, domain=pyo.Binary)

    # Max_Heat_Demand_DHN (scalar var)
    m.Max_Heat_Demand_DHN = pyo.Var(domain=pyo.NonNegativeReals)

    return m

def set_constraints(m, objective='gwp'):

    # Initialising derived constraints
    # 1. end_uses_input
    if not hasattr(m, 'end_uses_input'):
        def _eui_init(m, i):
            return sum(m.end_uses_demand_year[i, s] for s in m.SECTORS)
        m.end_uses_input = pyo.Param(m.END_USES_INPUT, initialize=_eui_init, mutable=True)

    # 2. total_time
    if not hasattr(m, 'total_time'):
        def _tt_init(m):
            return sum(m.t_op[t] for t in m.PERIODS)
        m.total_time = pyo.Param(initialize=_tt_init, mutable=True)

    # 3. POS_PROVIDERS
    if not hasattr(m, 'POS_PROVIDERS'):
        def _pos_providers_init(m, i):
            providers = []
            for j in m.NON_STORAGE_X:
                if pyo.value(m.layers_in_out[j, i]) > 0.0:
                    providers.append(j)
            return providers
        m.POS_PROVIDERS = pyo.Set(m.END_USES_TYPES, initialize=_pos_providers_init)

    m.Share_Mobility_Public = pyo.Var(bounds=(pyo.value(m.share_mobility_public_min), pyo.value(m.share_mobility_public_max)))
    m.Share_Freight_Train   = pyo.Var(bounds=(pyo.value(m.share_freight_train_min), pyo.value(m.share_freight_train_max)))
    m.Share_Heat_Dhn        = pyo.Var(bounds=(pyo.value(m.share_heat_dhn_min), pyo.value(m.share_heat_dhn_max)))

    # ---------- Constraints ----------
    # [Figure 1.4] End-uses demand rules
    def end_uses_rule(m, l, t):
        if l == "ELECTRICITY":
            return m.End_Uses[l, t] == ((m.end_uses_input[l] / m.total_time) +
                                        (m.end_uses_input["LIGHTING"] * m.lighting_month[t] / m.t_op[t])) + m.Losses[
                l, t]
        elif l == "HEAT_LOW_T_DHN":
            return m.End_Uses[l, t] == ((m.end_uses_input["HEAT_LOW_T_HW"] / m.total_time) +
                                        (m.end_uses_input["HEAT_LOW_T_SH"] * m.heating_month[t] / m.t_op[
                                            t])) * m.Share_Heat_Dhn + m.Losses[l, t]
        elif l == "HEAT_LOW_T_DECEN":
            return m.End_Uses[l, t] == ((m.end_uses_input["HEAT_LOW_T_HW"] / m.total_time) +
                                        (m.end_uses_input["HEAT_LOW_T_SH"] * m.heating_month[t] / m.t_op[t])) * (
                        1 - m.Share_Heat_Dhn)
        elif l == "MOB_PUBLIC":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_PASSENGER"] / m.total_time) * m.Share_Mobility_Public
        elif l == "MOB_PRIVATE":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_PASSENGER"] / m.total_time) * (
                        1 - m.Share_Mobility_Public)
        elif l == "MOB_FREIGHT_RAIL":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_FREIGHT"] / m.total_time) * m.Share_Freight_Train
        elif l == "MOB_FREIGHT_ROAD":
            return m.End_Uses[l, t] == (m.end_uses_input["MOBILITY_FREIGHT"] / m.total_time) * (
                        1 - m.Share_Freight_Train)
        elif l == "HEAT_HIGH_T":
            return m.End_Uses[l, t] == m.end_uses_input[l] / m.total_time
        else:
            return m.End_Uses[l, t] == 0

    m.end_uses_t = pyo.Constraint(m.LAYERS, m.PERIODS, rule=end_uses_rule)

    # [Eq. 1.7] Number of units (TECHNOLOGIES \ INFRASTRUCTURE)
    m.number_of_units = pyo.Constraint(
        m.TECHNOLOGIES - m.INFRASTRUCTURE, rule=lambda m, i: m.Number_Of_Units[i] == m.F_Mult[i] / m.ref_size[i]
    )

    # [Eq. 1.6] Size limits
    m.size_limit = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m, i: pyo.inequality(m.f_min[i], m.F_Mult[i], m.f_max[i])
    )

    # [Eq. 1.8] Period capacity factor
    m.capacity_factor_t = pyo.Constraint(
        m.TECHNOLOGIES, m.PERIODS, rule=lambda m, i, t: m.F_Mult_t[i, t] <= m.F_Mult[i] * m.c_p_t[i, t]
    )

    # [Eq. 1.9] Annual capacity factor
    m.capacity_factor = pyo.Constraint(
        m.TECHNOLOGIES,
        rule=lambda m, i: sum(m.F_Mult_t[i, t] * m.t_op[t] for t in m.PERIODS) <= m.F_Mult[i] * m.c_p[i] * m.total_time
    )

    # Linearization of Eq. 1.19
    # Decentralized heat operating strategy (linearized)
    m.op_strategy_decen_1_linear = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m, i, t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
                m.F_Mult_t[i, t] + m.X_Solar_Backup_Aux[i, t] >=
                sum(m.F_Mult_t[i, t2] * m.t_op[t2] for t2 in m.PERIODS) *
                ((m.end_uses_input["HEAT_LOW_T_HW"] / m.total_time + m.end_uses_input["HEAT_LOW_T_SH"] *
                  m.heating_month[t] / m.t_op[t]) /
                 (m.end_uses_input["HEAT_LOW_T_HW"] + m.end_uses_input["HEAT_LOW_T_SH"]))
        )
    )
    m.op_strategy_decen_1_linear_1 = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m, i, t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
                m.X_Solar_Backup_Aux[i, t] <= m.f_max["DEC_SOLAR"] * m.Y_Solar_Backup[i]
        )
    )
    m.op_strategy_decen_1_linear_2 = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m, i, t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
                m.X_Solar_Backup_Aux[i, t] <= m.F_Mult_t["DEC_SOLAR", t]
        )
    )
    m.op_strategy_decen_1_linear_3 = pyo.Constraint(
        m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"], m.PERIODS,
        rule=lambda m, i, t: pyo.Constraint.Skip if i == "DEC_SOLAR" else (
                m.X_Solar_Backup_Aux[i, t] >= m.F_Mult_t["DEC_SOLAR", t] - (1 - m.Y_Solar_Backup[i]) * m.f_max[
            "DEC_SOLAR"]
        )
    )

    # [Eq. 1.20] ]Only one backup
    m.op_strategy_decen_2 = pyo.Constraint(expr=sum(m.Y_Solar_Backup[i] for i in m.TECHNOLOGIES) <= 1)

    ## Layers

    # [Eq. 1.13] Layer balance with storage
    m.layer_balance = pyo.Constraint(
        m.LAYERS, m.PERIODS,
        rule=lambda m, l, t: 0 ==
                             (sum(m.layers_in_out[x, l] * m.F_Mult_t[x, t] for x in m.NON_STORAGE_X)) +
                             (sum(m.Storage_Out[j, l, t] - m.Storage_In[j, l, t] for j in m.STORAGE_TECH)) -
                             m.End_Uses[l, t]
    )

    # [Eq. 1.12] Resources availability
    m.resource_availability = pyo.Constraint(
        m.RESOURCES, rule=lambda m, i: sum(m.F_Mult_t[i, t] * m.t_op[t] for t in m.PERIODS) <= m.avail[i]
    )

    # [Eq. 1.15-1.16] Each storage technology can have input/output only to certain layers.
    # If incompatible then the variable is set to 0 ceil (x) operator rounds a number to the highest nearest integer.
    # Storage compatibility: if eff==0, var == 0 (ceil trick equivalent)
    # If eff = 0 ⇒ var == 0; if eff = 1 ⇒ 0 == 0 (vacuous)
    m.storage_layer_in = pyo.Constraint(
        m.STORAGE_TECH, m.LAYERS, m.PERIODS,
        rule=lambda m, i, l, t: m.Storage_In[i, l, t] * (1 - m.storage_eff_in[i, l]) == 0
    )
    m.storage_layer_out = pyo.Constraint(
        m.STORAGE_TECH, m.LAYERS, m.PERIODS,
        rule=lambda m, i, l, t: m.Storage_Out[i, l, t] * (1 - m.storage_eff_out[i, l]) == 0
    )

    # Linearization of [Eq. 1.17]
    # Storage no-transfer linearization
    m.storage_no_transfer_1 = pyo.Constraint(
        m.STORAGE_TECH, m.PERIODS,
        rule=lambda m, i, t: (sum(
            m.Storage_In[i, l, t] * m.storage_eff_in[i, l] for l in m.LAYERS if pyo.value(m.storage_eff_in[i, l]) > 0)
                              * m.t_op[t] / m.f_max[i]) <= m.Y_Sto_In[i, t]
    )
    m.storage_no_transfer_2 = pyo.Constraint(
        m.STORAGE_TECH, m.PERIODS,
        rule=lambda m, i, t: (sum(m.Storage_Out[i, l, t] / m.storage_eff_out[i, l] for l in m.LAYERS if
                                  pyo.value(m.storage_eff_out[i, l]) > 0)
                              * m.t_op[t] / m.f_max[i]) <= m.Y_Sto_Out[i, t]
    )
    m.storage_no_transfer_3 = pyo.Constraint(
        m.STORAGE_TECH, m.PERIODS,
        rule=lambda m, i, t: m.Y_Sto_In[i, t] + m.Y_Sto_Out[i, t] <= 1
    )

    # [Eq. 1.14] Storage level with wrap-around
    def storage_level_rule(m, i, t):
        prev_t = m.PERIODS.last() if t == m.PERIODS.first() else m.PERIODS.prev(t)
        inflow = sum(
            m.Storage_In[i, l, t] * m.storage_eff_in[i, l]
            for l in m.LAYERS
            if pyo.value(m.storage_eff_in[i, l]) > 0
        )
        outflow = sum(
            m.Storage_Out[i, l, t] / m.storage_eff_out[i, l]
            for l in m.LAYERS
            if pyo.value(m.storage_eff_out[i, l]) > 0
        )
        return m.F_Mult_t[i, t] == m.F_Mult_t[i, prev_t] + (inflow - outflow) * m.t_op[t]

    m.storage_level = pyo.Constraint(m.STORAGE_TECH, m.PERIODS, rule=storage_level_rule)

    # [Eq. 1.18] Network losses (pre-filtered contributors)
    m.network_losses = pyo.Constraint(
        m.END_USES_TYPES, m.PERIODS,
        rule=lambda m, i, t: m.Losses[i, t] == (
            sum(m.layers_in_out[j, i] * m.F_Mult_t[j, t] for j in m.POS_PROVIDERS[i])
        ) * m.loss_coeff[i]
    )

    # [Eq 1.22] f_max_perc / f_min_perc per end-use type
    m.f_max_perc = pyo.Constraint(
        m.END_USES_TYPES, m.TECHNOLOGIES,
        rule=lambda m, i, j: pyo.Constraint.Skip
        if (j not in set(m.TECHNOLOGIES_OF_END_USES_TYPE[i]))
        else (sum(m.F_Mult_t[j, t] * m.t_op[t] for t in m.PERIODS)
              <= m.fmax_perc[j] * sum(m.F_Mult_t[j2, t2] * m.t_op[t2]
                                      for j2 in m.TECHNOLOGIES_OF_END_USES_TYPE[i] for t2 in m.PERIODS))
    )
    m.f_min_perc = pyo.Constraint(
        m.END_USES_TYPES, m.TECHNOLOGIES,
        rule=lambda m, i, j: pyo.Constraint.Skip
        if (j not in set(m.TECHNOLOGIES_OF_END_USES_TYPE[i]))
        else (sum(m.F_Mult_t[j, t] * m.t_op[t] for t in m.PERIODS)
              >= m.fmin_perc[j] * sum(m.F_Mult_t[j2, t2] * m.t_op[t2]
                                      for j2 in m.TECHNOLOGIES_OF_END_USES_TYPE[i] for t2 in m.PERIODS))
    )

    # [Eq. 1.24] Seasonal storage in hydro dams — rule-based with Skip
    def _hydro_cap_rule(m):
        if ("PUMPED_HYDRO" in m.TECHNOLOGIES) and ("NEW_HYDRO_DAM" in m.TECHNOLOGIES):
            return m.F_Mult["PUMPED_HYDRO"] <= m.f_max["PUMPED_HYDRO"] * (
                    (m.F_Mult["NEW_HYDRO_DAM"] - m.f_min["NEW_HYDRO_DAM"]) /
                    (m.f_max["NEW_HYDRO_DAM"] - m.f_min["NEW_HYDRO_DAM"])
            )
        return pyo.Constraint.Skip

    m.storage_level_hydro_dams = pyo.Constraint(rule=_hydro_cap_rule)

    # [Eq. 1.25] Hydro dams can only shift production — rule-based with Skip ------   !  HERE  !
    m.hydro_dams_shift = pyo.Constraint(
        m.PERIODS,
        rule=lambda m, t: m.Storage_In["PUMPED_HYDRO", "ELECTRICITY", t]
                          <= m.F_Mult_t["HYDRO_DAM", t] + m.F_Mult_t["NEW_HYDRO_DAM", t]
    )

    # [Eq. 1.26] DHN: assigning a cost to the network
    # Note that in Moret (2017), page 26, there is a ">=" sign instead of an "=". The two formulations are equivalent as long as the problem minimises cost and the DHN has a cost > 0
    m.extra_dhn = pyo.Constraint(
        rule=lambda m: m.F_Mult["DHN"] ==
                       sum(m.F_Mult[j] for j in m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DHN"])
    )

    # [Eq. 1.27] Calculation of max heat demand in DHN
    m.max_dhn_heat_demand = pyo.Constraint(
        m.PERIODS,
        rule=lambda m, t: m.Max_Heat_Demand_DHN >= m.End_Uses["HEAT_LOW_T_DHN", t]
    )

    # peak_dhn
    m.peak_dhn = pyo.Constraint(
        rule=lambda m: sum(m.F_Mult[j] for j in m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DHN"])
                       >= m.peak_dhn_factor * m.Max_Heat_Demand_DHN
    )

    # [Eq. 1.28] 9.4 BCHF is the extra investment needed if there is a big deployment of stochastic renewables
    m.extra_grid = pyo.Constraint(
        rule=lambda m: m.F_Mult["GRID"] ==
                       1 + (9400 / m.c_inv["GRID"]) * (m.F_Mult["WIND"] + m.F_Mult["PV"]) /
                       (m.f_max["WIND"] + m.f_max["PV"])
    )

    # [Eq. 1.29] Power2Gas investment cost is calculated on the max size of the two units
    m.extra_power2gas_1 = pyo.Constraint(rule=lambda m: m.F_Mult["POWER2GAS_3"] >= m.F_Mult["POWER2GAS_1"])
    m.extra_power2gas_2 = pyo.Constraint(rule=lambda m: m.F_Mult["POWER2GAS_3"] >= m.F_Mult["POWER2GAS_2"])

    # [Eq. 1.23] Operating strategy in private mobility (to make model more realistic)
    m.op_strategy_mob_private = pyo.Constraint(
        m.TECHNOLOGIES, m.PERIODS,
        rule=lambda m, i, t: pyo.Constraint.Skip
        if i not in (set(m.TECHNOLOGIES_OF_END_USES_CATEGORY["MOBILITY_PASSENGER"])
                     | set(m.TECHNOLOGIES_OF_END_USES_CATEGORY["MOBILITY_FREIGHT"]))
        else m.F_Mult_t[i, t] >= sum(m.F_Mult_t[i, t2] * m.t_op[t2] / m.total_time for t2 in m.PERIODS)
    )

    # Energy efficiency is a fixed cost
    m.extra_efficiency = pyo.Constraint(
        rule=lambda m: m.F_Mult["EFFICIENCY"] == 1 / (1 + m.i_rate)
    )

    # ----- Cost -----
    # [Eq. 1.3] Investment cost of each technology
    m.investment_cost_calc = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m, i: m.C_inv[i] == m.c_inv[i] * m.F_Mult[i]
    )
    # [Eq. 1.4] O&M cost of each technology
    m.main_cost_calc = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m, i: m.C_maint[i] == m.c_maint[i] * m.F_Mult[i]
    )
    # [Eq. 1.10] Total cost of each resource
    m.op_cost_calc = pyo.Constraint(
        m.RESOURCES, rule=lambda m, i: m.C_op[i] == sum(m.c_op[i, t] * m.F_Mult_t[i, t] * m.t_op[t] for t in m.PERIODS)
    )
    # [Eq. 1.1]
    m.totalcost_cal = pyo.Constraint(
        expr=m.TotalCost == sum(m.tau[i] * m.C_inv[i] + m.C_maint[i] for i in m.TECHNOLOGIES) + sum(
            m.C_op[j] for j in m.RESOURCES)
    )

    # ----- Emissions -----
    m.gwp_constr_calc = pyo.Constraint(
        m.TECHNOLOGIES, rule=lambda m, i: m.GWP_constr[i] == m.gwp_constr_param[i] * m.F_Mult[i]
    )
    m.gwp_op_calc = pyo.Constraint(
        m.RESOURCES,
        rule=lambda m, i: m.GWP_op[i] == m.gwp_op_param[i] * sum(m.t_op[t] * m.F_Mult_t[i, t] for t in m.PERIODS)
    )
    m.totalGWP_calc = pyo.Constraint(
        expr=m.TotalGWP == sum(m.GWP_constr[i] / m.lifetime[i] for i in m.TECHNOLOGIES) + sum(
            m.GWP_op[j] for j in m.RESOURCES)
    )

    # ---------- Objective ----------
    if objective == "gwp":
        m.obj = pyo.Objective(expr=m.TotalGWP, sense=pyo.minimize)
    elif objective == "cost":
        m.obj = pyo.Objective(expr=m.TotalCost, sense=pyo.minimize)
    else:
        raise ValueError("objective must be 'gwp' or 'cost'")

    # helper to switch objective post-build
    def set_objective(mdl, which="gwp"):
        if hasattr(mdl, "obj"):
            mdl.del_component(mdl.obj)
        if which == "gwp":
            mdl.obj = pyo.Objective(expr=mdl.TotalGWP, sense=pyo.minimize)
        elif which == "cost":
            mdl.obj = pyo.Objective(expr=mdl.TotalCost, sense=pyo.minimize)
        else:
            raise ValueError("which must be 'gwp' or 'cost'")

    m.set_objective = set_objective
    return


# CLI usage
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to ses_main.json")
    ap.add_argument("--objective", default="gwp", choices=["gwp","cost"])
    args = ap.parse_args()

    data = load_data(args.data)
    m = build_model(data, objective=args.objective)
    opt = make_highs()
    attach(opt, m)
    res = solve(opt, m, warmstart=True)
    print(res.termination_condition)
    print("Objective:", pyo.value(m.obj))
    print(extract_results(m))
