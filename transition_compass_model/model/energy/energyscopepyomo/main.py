from transition_compass_model.model.energy.energyscopepyomo.ses_pyomo import (  # was .ses_pyomo before
    attach,
    build_model,
    extract_results,
    load_data,
    make_highs,
    solve,
)

data = load_data("ses_main.json")
m = build_model(data, objective="gwp")
opt = make_highs()
attach(opt, m)
res = solve(opt, m, warmstart=True)

print(extract_results(m))

# Scenario 2:
# apply_scenario(m, {"f_max": {"PV": 1200.0, "WIND": 1500.0}})
# solve(opt, m, warmstart=True)
# print(extract_results(m))


print("Hello")
