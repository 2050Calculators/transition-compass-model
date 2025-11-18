from ses_pyomo import load_data, build_model, make_highs, attach, solve, extract_results
import pyomo.environ as pyo

data = load_data("ses_main.json")
m = build_model(data, objective="gwp")
opt = make_highs()
attach(opt, m)
res = solve(opt, m, warmstart=True)

print(extract_results(m))

# Scenario 2:
#apply_scenario(m, {"f_max": {"PV": 1200.0, "WIND": 1500.0}})
#solve(opt, m, warmstart=True)
#print(extract_results(m))



print('Hello')