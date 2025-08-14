######################################################
#
# Swiss-EnergyScope (SES) MILP modeling framework
# Model file
# Author: Stefano Moret
# Date: 27.10.2017
# Model documentation: Moret S. (2017). "Strategic Energy Planning under Uncertainty". PhD Thesis n. 7961, EPFL, Switzerland (Chapter 1). (http://dx.doi.org/10.5075/epfl-thesis-7961)
# For terms of use and how to cite this work please check the ReadMe file. 
#
######################################################


### Printing output

## OUTPUT IN TXT FILES ##

## Print total yearly output to txt file
printf "%s\t%s\n", "Name", "Yearly output" > "energy/energyscope-MILP/output/total_output.txt";
for {i in TECHNOLOGIES union RESOURCES diff STORAGE_TECH}{
	printf "%s\t%.3f\n", i, sum{t in PERIODS} (F_Mult_t [i, t] * t_op [t]) >> "energy/energyscope-MILP/output/total_output.txt";
}

## Print cost breakdown to txt file.
printf "%s\t%s\t%s\t%s\n", "Name", "C_inv", "C_maint", "C_op" > "energy/energyscope-MILP/output/cost_breakdown.txt"; 
for {i in TECHNOLOGIES union RESOURCES}{
	printf "%s\t%.6f\t%.6f\t%.6f\n", i, if i in TECHNOLOGIES then (tau [i] * C_inv [i]) else 0, if i in TECHNOLOGIES then C_maint [i] else 0, if i in RESOURCES then C_op [i] else 0 >> "energy/energyscope-MILP/output/cost_breakdown.txt";
}

## Print GWP breakdown
printf "%s\t%s\t%s\n", "Name", "GWP_constr", "GWP_op" > "energy/energyscope-MILP/output/gwp_breakdown.txt"; 
for {i in TECHNOLOGIES union RESOURCES}{
	printf "%s\t%.6f\t%.6f\n", i, if i in TECHNOLOGIES then GWP_constr [i] / lifetime [i] else 0, if i in RESOURCES then GWP_op [i] else 0 >> "energy/energyscope-MILP/output/gwp_breakdown.txt";
}

## Print F_Mult to txt file
printf "%s\t%s\n", "Name", "Mult" > "energy/energyscope-MILP/output/f_mult.txt"; 
for {i in TECHNOLOGIES}{
	printf "%s\t%.6f\n", i, F_Mult[i] >> "energy/energyscope-MILP/output/f_mult.txt";
}

## Print F_Mult_t to txt file.
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "Name", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" > "energy/energyscope-MILP/output/f_mult_t.txt"; 
for {i in TECHNOLOGIES union RESOURCES}{
	printf "%s\t", i  >> "energy/energyscope-MILP/output/f_mult_t.txt";
	for {t in PERIODS}{
		printf "\t%.6f", F_Mult_t[i,t] >> "energy/energyscope-MILP/output/f_mult_t.txt";
	}
	printf "\n" >> "energy/energyscope-MILP/output/f_mult_t.txt";
}

## Print End_Uses to txt file (with negative sign to close balance)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "Name", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" > "energy/energyscope-MILP/output/End_Uses.txt"; 
for {i in LAYERS}{
	printf "%s\t", i >> "energy/energyscope-MILP/output/End_Uses.txt";
	for {t in PERIODS}{
		printf "%.6f\t", -End_Uses[i,t] >> "energy/energyscope-MILP/output/End_Uses.txt";
	}
	printf "\n" >> "energy/energyscope-MILP/output/End_Uses.txt";
}

## Print storage balance from/to layers (Storage_Out - Storage_In) to txt file.
for {i in STORAGE_TECH}{
	printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "Not accounting for efficiency", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" > ("energy/energyscope-MILP/output/" & i & ".txt"); 
	for {l in LAYERS}{
		printf "%s\t", l >> ("energy/energyscope-MILP/output/" & i & ".txt");
		for {t in PERIODS}{
			printf "%.6f\t", (Storage_Out[i,l,t] - Storage_In[i,l,t]) >> ("energy/energyscope-MILP/output/" & i & ".txt");
		}
		printf "\n" >> ("energy/energyscope-MILP/output/" & i & ".txt");
	}
}

## Print losses to txt file
printf "%s\t%s\n", "End use", "Losses" > "energy/energyscope-MILP/output/losses.txt";
for {i in END_USES_TYPES}{
		printf "%s\t%.3f\n", i, sum{t in PERIODS}(Losses [i,t] * t_op [t]) >> "energy/energyscope-MILP/output/losses.txt";
}


## SANKEY DIAGRAM ##
# The code to plot the Sankey diagrams is originally taken from: http://bl.ocks.org/d3noob/c9b90689c1438f57d649
# Adapted by the IPESE team

## Generate CSV file to be used as input to Sankey diagram
# Notes:
# - workaround to write if-then-else statements in GLPK: https://en.wikibooks.org/wiki/GLPK/GMPL_Workarounds#If–then–else_conditional
# - to visualize the Sankey, open the html file in any browser. If it does not work, try this: https://github.com/mrdoob/three.js/wiki/How-to-run-things-locally
 
printf "%s,%s,%s,%s,%s,%s\n", "source" , "target", "realValue", "layerID", "layerColor", "layerUnit" > "energy/energyscope-MILP/output/sankey/input2sankey.csv";
## Gasoline
for{{0}: sum{t in PERIODS}(F_Mult_t ["GASOLINE", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Gasoline" , "Mob priv", sum{t in PERIODS}(layers_in_out["GASOLINE","GASOLINE"] * F_Mult_t ["GASOLINE", t] * t_op [t]) / 1000 , "Gasoline", "#808080", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
## Diesel
for{{0}: sum{t in PERIODS}(F_Mult_t ["CAR_DIESEL", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Diesel" , "Mob priv", sum{t in PERIODS}(-layers_in_out["CAR_DIESEL","DIESEL"] * F_Mult_t ["CAR_DIESEL", t] * t_op [t]) / 1000 , "Diesel", "#D3D3D3", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS} ((F_Mult_t ["BUS_COACH_DIESEL", t] + F_Mult_t["BUS_COACH_HYDIESEL", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Diesel" , "Mob public", sum{t in PERIODS}(-layers_in_out["BUS_COACH_DIESEL","DIESEL"] * F_Mult_t ["BUS_COACH_DIESEL", t] * t_op [t] - layers_in_out["BUS_COACH_HYDIESEL","DIESEL"] * F_Mult_t ["BUS_COACH_HYDIESEL", t] * t_op [t] ) / 1000 , "Diesel", "#D3D3D3", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS} ((F_Mult_t ["TRUCK", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Diesel" , "Freight", sum{t in PERIODS}(-layers_in_out["TRUCK","DIESEL"] * F_Mult_t ["TRUCK", t] * t_op [t]) / 1000 , "Diesel", "#D3D3D3", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
## Natural Gas
for{{0}: sum{t in PERIODS}(F_Mult_t ["CAR_NG", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "Mob priv", sum{t in PERIODS}(-layers_in_out["CAR_NG","NG"] * F_Mult_t ["CAR_NG", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["BUS_COACH_CNG_STOICH", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "Mob public", sum{t in PERIODS}(-layers_in_out["BUS_COACH_CNG_STOICH","NG"] * F_Mult_t ["BUS_COACH_CNG_STOICH", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["H2_NG", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "H2 prod", sum{t in PERIODS}(-layers_in_out["H2_NG","NG"] * F_Mult_t ["H2_NG", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["CCGT", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "Elec", sum{t in PERIODS}(-layers_in_out["CCGT","NG"] * F_Mult_t ["CCGT", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["CCGT_CCS", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG CCS" , "Elec", sum{t in PERIODS}(-layers_in_out["CCGT_CCS","NG_CCS"] * F_Mult_t ["CCGT_CCS", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_COGEN_GAS", t] + F_Mult_t ["DHN_COGEN_GAS", t] + F_Mult_t ["DEC_COGEN_GAS", t] + F_Mult_t ["DEC_ADVCOGEN_GAS", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "CHP", sum{t in PERIODS}(-layers_in_out["IND_COGEN_GAS","NG"] * F_Mult_t ["IND_COGEN_GAS", t] * t_op [t] - layers_in_out["DHN_COGEN_GAS","NG"] * F_Mult_t ["DHN_COGEN_GAS", t] * t_op [t] - layers_in_out["DEC_COGEN_GAS","NG"] * F_Mult_t ["DEC_COGEN_GAS", t] * t_op [t] - layers_in_out["DEC_ADVCOGEN_GAS","NG"] * F_Mult_t ["DEC_ADVCOGEN_GAS", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["DEC_THHP_GAS", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "HPs", sum{t in PERIODS}(-layers_in_out["DEC_THHP_GAS","NG"] * F_Mult_t ["DEC_THHP_GAS", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_BOILER_GAS", t] + F_Mult_t ["DHN_BOILER_GAS", t] + F_Mult_t ["DEC_BOILER_GAS", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "Boilers", sum{t in PERIODS}(-layers_in_out["IND_BOILER_GAS","NG"] * F_Mult_t ["IND_BOILER_GAS", t] * t_op [t] - layers_in_out["DHN_BOILER_GAS","NG"] * F_Mult_t ["DHN_BOILER_GAS", t] * t_op [t] - layers_in_out["DEC_BOILER_GAS","NG"] * F_Mult_t ["DEC_BOILER_GAS", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
## Electricity production
for{{0}: sum{t in PERIODS}(F_Mult_t ["ELECTRICITY", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Electricity" , "Elec", sum{t in PERIODS}(layers_in_out["ELECTRICITY","ELECTRICITY"] * F_Mult_t ["ELECTRICITY", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["NUCLEAR", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Nuclear" , "Elec", sum{t in PERIODS}(layers_in_out["NUCLEAR","ELECTRICITY"] * F_Mult_t ["NUCLEAR", t] * t_op [t]) / 1000 , "Nuclear", "#FFC0CB", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["WIND", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Wind" , "Elec", sum{t in PERIODS}(layers_in_out["WIND","ELECTRICITY"] * F_Mult_t ["WIND", t] * t_op [t]) / 1000 , "Wind", "#FFA500", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["HYDRO_DAM", t] + F_Mult_t ["NEW_HYDRO_DAM", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Hydro Dams" , "Elec", sum{t in PERIODS}(layers_in_out["HYDRO_DAM","ELECTRICITY"] * F_Mult_t ["HYDRO_DAM", t] * t_op [t] + layers_in_out["NEW_HYDRO_DAM","ELECTRICITY"] * F_Mult_t ["NEW_HYDRO_DAM", t] * t_op [t]) / 1000 , "Hydro Dam", "#00CED1", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["HYDRO_RIVER", t] + F_Mult_t ["NEW_HYDRO_RIVER", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Hydro River" , "Elec", sum{t in PERIODS}(layers_in_out["HYDRO_RIVER","ELECTRICITY"] * F_Mult_t ["HYDRO_RIVER", t] * t_op [t] + layers_in_out["NEW_HYDRO_RIVER","ELECTRICITY"] * F_Mult_t ["NEW_HYDRO_RIVER", t] * t_op [t]) / 1000 , "Hydro River", "#0000FF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Coal
for{{0}: sum{t in PERIODS}((F_Mult_t ["COAL_US", t] + F_Mult_t ["COAL_IGCC", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Coal" , "Elec", sum{t in PERIODS}(-layers_in_out["COAL_US","COAL"] * F_Mult_t ["COAL_US", t] * t_op [t] - layers_in_out["COAL_IGCC","COAL"] * F_Mult_t ["COAL_IGCC", t] * t_op [t]) / 1000 , "Coal", "#A0522D", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["COAL_US_CCS", t] + F_Mult_t ["COAL_IGCC_CCS", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Coal CCS" , "Elec", sum{t in PERIODS}(-layers_in_out["COAL_US_CCS","COAL_CCS"] * F_Mult_t ["COAL_US_CCS", t] * t_op [t] - layers_in_out["COAL_IGCC_CCS","COAL_CCS"] * F_Mult_t ["COAL_IGCC_CCS", t] * t_op [t]) / 1000 , "Coal", "#A0522D", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["IND_BOILER_COAL", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Coal" , "Boilers", sum{t in PERIODS}(-layers_in_out["IND_BOILER_COAL","COAL"] * F_Mult_t ["IND_BOILER_COAL", t] * t_op [t]) / 1000 , "Coal", "#A0522D", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Solar
for{{0}: sum{t in PERIODS}(F_Mult_t ["PV", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Solar" , "Elec", sum{t in PERIODS}(layers_in_out["PV","ELECTRICITY"] * F_Mult_t ["PV", t] * t_op [t]) / 1000 , "Solar", "#FFFF00", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["DEC_SOLAR", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Solar" , "Heat LT Dec", sum{t in PERIODS}(layers_in_out["DEC_SOLAR","HEAT_LOW_T_DECEN"] * F_Mult_t ["DEC_SOLAR", t] * t_op [t]) / 1000 , "Solar", "#FFFF00", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Geothermal
for{{0}: sum{t in PERIODS}(F_Mult_t ["GEOTHERMAL", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Geothermal" , "Elec", sum{t in PERIODS}(layers_in_out["GEOTHERMAL","ELECTRICITY"] * F_Mult_t ["GEOTHERMAL", t] * t_op [t]) / 1000 , "Geothermal", "#FF0000", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["DHN_DEEP_GEO", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Geothermal" , "DHN", sum{t in PERIODS}(layers_in_out["DHN_DEEP_GEO","HEAT_LOW_T_DHN"] * F_Mult_t ["DHN_DEEP_GEO", t] * t_op [t]) / 1000 , "Geothermal", "#FF0000", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Waste
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_COGEN_WASTE", t] + F_Mult_t ["DHN_COGEN_WASTE", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Waste" , "CHP", sum{t in PERIODS}(-layers_in_out["IND_COGEN_WASTE","WASTE"] * F_Mult_t ["IND_COGEN_WASTE", t] * t_op [t] -layers_in_out["DHN_COGEN_WASTE","WASTE"] * F_Mult_t ["DHN_COGEN_WASTE", t] * t_op [t]) / 1000 , "Waste", "#808000", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["IND_BOILER_WASTE", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Waste" , "Boilers", sum{t in PERIODS}(-layers_in_out["IND_BOILER_WASTE","WASTE"] * F_Mult_t ["IND_BOILER_WASTE", t] * t_op [t]) / 1000 , "Waste", "#808000", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Oil
for{{0}: sum{t in PERIODS}((F_Mult_t ["DEC_COGEN_OIL", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Oil" , "CHP", sum{t in PERIODS}(-layers_in_out["DEC_COGEN_OIL","LFO"] * F_Mult_t ["DEC_COGEN_OIL", t] * t_op [t]) / 1000 , "Oil", "#8B008B", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_BOILER_OIL", t] + F_Mult_t ["DHN_BOILER_OIL", t] + F_Mult_t ["DEC_BOILER_OIL", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Oil" , "Boilers", sum{t in PERIODS}(-layers_in_out["IND_BOILER_OIL","LFO"] * F_Mult_t ["IND_BOILER_OIL", t] * t_op [t] - layers_in_out["DHN_BOILER_OIL","LFO"] * F_Mult_t ["DHN_BOILER_OIL", t] * t_op [t] - layers_in_out["DEC_BOILER_OIL","LFO"] * F_Mult_t ["DEC_BOILER_OIL", t] * t_op [t]) / 1000 , "Oil", "#8B008B", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Wood
for{{0}: sum{t in PERIODS}(F_Mult_t ["H2_BIOMASS", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Wood" , "H2 prod", sum{t in PERIODS}(-layers_in_out["H2_BIOMASS","WOOD"] * F_Mult_t ["H2_BIOMASS", t] * t_op [t]) / 1000 , "Wood", "#CD853F", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["GASIFICATION_SNG", t] + F_Mult_t ["PYROLYSIS", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Wood" , "Biofuels", sum{t in PERIODS}(-layers_in_out["GASIFICATION_SNG","WOOD"] * F_Mult_t ["GASIFICATION_SNG", t] * t_op [t] - layers_in_out["PYROLYSIS","WOOD"] * F_Mult_t ["PYROLYSIS", t] * t_op [t]) / 1000 , "Wood", "#CD853F", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_COGEN_WOOD", t] + F_Mult_t ["DHN_COGEN_WOOD", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Wood" , "CHP", sum{t in PERIODS}(-layers_in_out["IND_COGEN_WOOD","WOOD"] * F_Mult_t ["IND_COGEN_WOOD", t] * t_op [t] - layers_in_out["DHN_COGEN_WOOD","WOOD"] * F_Mult_t ["DHN_COGEN_WOOD", t] * t_op [t]) / 1000 , "Wood", "#CD853F", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_BOILER_WOOD", t] + F_Mult_t ["DHN_BOILER_WOOD", t] + F_Mult_t ["DEC_BOILER_WOOD", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Wood" , "Boilers", sum{t in PERIODS}(-layers_in_out["IND_BOILER_WOOD","WOOD"] * F_Mult_t ["IND_BOILER_WOOD", t] * t_op [t] - layers_in_out["DHN_BOILER_WOOD","WOOD"] * F_Mult_t ["DHN_BOILER_WOOD", t] * t_op [t] - layers_in_out["DEC_BOILER_WOOD","WOOD"] * F_Mult_t ["DEC_BOILER_WOOD", t] * t_op [t]) / 1000 , "Wood", "#CD853F", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Electricity use
for{{0}: sum{t in PERIODS}((F_Mult_t ["CAR_PHEV", t] + F_Mult_t ["CAR_BEV", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "Mob priv", sum{t in PERIODS}(-layers_in_out["CAR_PHEV","ELECTRICITY"] * F_Mult_t ["CAR_PHEV", t] * t_op [t] - layers_in_out["CAR_BEV","ELECTRICITY"] * F_Mult_t ["CAR_BEV", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["TRAIN_PUB", t] + F_Mult_t ["TRAMWAY_TROLLEY", t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "Mob public", sum{t in PERIODS}(-layers_in_out["TRAIN_PUB","ELECTRICITY"] * F_Mult_t ["TRAIN_PUB", t] * t_op [t] - layers_in_out["TRAMWAY_TROLLEY","ELECTRICITY"] * F_Mult_t ["TRAMWAY_TROLLEY", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["TRAIN_FREIGHT", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "Freight", sum{t in PERIODS}(-layers_in_out["TRAIN_FREIGHT","ELECTRICITY"] * F_Mult_t ["TRAIN_FREIGHT", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(Losses ["ELECTRICITY", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "Exp & Loss", sum{t in PERIODS}(Losses ["ELECTRICITY",t] * t_op [t] - layers_in_out["ELEC_EXPORT","ELECTRICITY"] * F_Mult_t ["ELEC_EXPORT", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(End_Uses ["ELECTRICITY",t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "Elec demand", sum{t in PERIODS}(End_Uses ["ELECTRICITY", t] * t_op [t] - Losses ["ELECTRICITY",t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["DHN_HP_ELEC",t] + F_Mult_t ["DEC_HP_ELEC",t])* t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "HPs", sum{t in PERIODS}(-layers_in_out["DHN_HP_ELEC","ELECTRICITY"] * F_Mult_t ["DHN_HP_ELEC", t] * t_op [t] - layers_in_out["DEC_HP_ELEC","ELECTRICITY"] * F_Mult_t ["DEC_HP_ELEC", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["H2_ELECTROLYSIS",t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "H2 prod", sum{t in PERIODS}(-layers_in_out["H2_ELECTROLYSIS","ELECTRICITY"] * F_Mult_t ["H2_ELECTROLYSIS", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["DEC_DIRECT_ELEC",t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "Heat LT Dec", sum{t in PERIODS}(layers_in_out["DEC_DIRECT_ELEC","HEAT_LOW_T_DECEN"] * F_Mult_t ["DEC_DIRECT_ELEC", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["IND_DIRECT_ELEC",t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Elec" , "Heat HT", sum{t in PERIODS}(layers_in_out["IND_DIRECT_ELEC","HEAT_HIGH_T"] * F_Mult_t ["IND_DIRECT_ELEC", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# H2 use
for{{0}: sum{t in PERIODS}(F_Mult_t ["DEC_ADVCOGEN_H2",t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "H2 prod" , "CHP", sum{t in PERIODS}(-layers_in_out["DEC_ADVCOGEN_H2","H2"] * F_Mult_t ["DEC_ADVCOGEN_H2", t] * t_op [t]) / 1000 , "H2", "#FF00FF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["CAR_FUEL_CELL",t] * t_op [t]) > 100}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "H2 prod" , "Mob priv", sum{t in PERIODS}(-layers_in_out["CAR_FUEL_CELL","H2"] * F_Mult_t ["CAR_FUEL_CELL", t] * t_op [t]) / 1000 , "H2", "#FF00FF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["BUS_COACH_FC_HYBRIDH2",t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "H2 prod" , "Mob public", sum{t in PERIODS}(-layers_in_out["BUS_COACH_FC_HYBRIDH2","H2"] * F_Mult_t ["BUS_COACH_FC_HYBRIDH2", t] * t_op [t]) / 1000 , "H2", "#FF00FF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# CHP
for{{0}: sum{i in COGEN, t in PERIODS}(F_Mult_t [i,t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "CHP" , "Elec", sum{i in COGEN, t in PERIODS}(layers_in_out[i,"ELECTRICITY"] * F_Mult_t [i, t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["DEC_COGEN_GAS",t] + F_Mult_t ["DEC_COGEN_OIL",t] + F_Mult_t ["DEC_ADVCOGEN_GAS",t] + F_Mult_t ["DEC_ADVCOGEN_H2",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "CHP" , "Heat LT Dec", sum{i in COGEN, t in PERIODS}(layers_in_out[i,"HEAT_LOW_T_DECEN"] * F_Mult_t [i, t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["DHN_COGEN_GAS",t] + F_Mult_t ["DHN_COGEN_WOOD",t] + F_Mult_t ["DHN_COGEN_WASTE",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "CHP" , "DHN", sum{i in COGEN, t in PERIODS}(layers_in_out[i,"HEAT_LOW_T_DHN"] * F_Mult_t [i, t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_COGEN_GAS",t] + F_Mult_t ["IND_COGEN_WOOD",t] + F_Mult_t ["IND_COGEN_WASTE",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "CHP" , "Heat HT", sum{i in COGEN, t in PERIODS}(layers_in_out[i,"HEAT_HIGH_T"] * F_Mult_t [i, t] * t_op [t]) / 1000 , "Heat HT", "#DC143C", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# HPs
for{{0}: sum{t in PERIODS}((F_Mult_t ["DEC_HP_ELEC",t] + F_Mult_t ["DEC_THHP_GAS",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "HPs" , "Heat LT Dec", sum{t in PERIODS}(layers_in_out["DEC_HP_ELEC","HEAT_LOW_T_DECEN"] * F_Mult_t ["DEC_HP_ELEC", t] * t_op [t] + layers_in_out["DEC_THHP_GAS","HEAT_LOW_T_DECEN"] * F_Mult_t ["DEC_THHP_GAS", t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(F_Mult_t ["DHN_HP_ELEC",t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "HPs" , "DHN", sum{t in PERIODS}(layers_in_out["DHN_HP_ELEC","HEAT_LOW_T_DHN"] * F_Mult_t ["DHN_HP_ELEC", t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Biofuels
for{{0}: sum{t in PERIODS}((F_Mult_t ["GASIFICATION_SNG",t] + F_Mult_t ["PYROLYSIS",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Biofuels" , "Elec", sum{t in PERIODS}(layers_in_out["GASIFICATION_SNG","ELECTRICITY"] * F_Mult_t ["GASIFICATION_SNG", t] * t_op [t] + layers_in_out["PYROLYSIS","ELECTRICITY"] * F_Mult_t ["PYROLYSIS", t] * t_op [t]) / 1000 , "Electricity", "#00BFFF", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["GASIFICATION_SNG",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Biofuels" , "NG", sum{t in PERIODS}(layers_in_out["GASIFICATION_SNG","NG"] * F_Mult_t ["GASIFICATION_SNG", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["PYROLYSIS",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Biofuels" , "Oil", sum{t in PERIODS}(layers_in_out["PYROLYSIS","LFO"] * F_Mult_t ["PYROLYSIS", t] * t_op [t]) / 1000 , "Oil", "#8B008B", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["GASIFICATION_SNG",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Biofuels" , "DHN", sum{t in PERIODS}(layers_in_out["GASIFICATION_SNG","HEAT_LOW_T_DHN"] * F_Mult_t ["GASIFICATION_SNG", t] * t_op [t]) / 1000, "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# Boilers
for{{0}: sum{t in PERIODS}((F_Mult_t ["DEC_BOILER_GAS",t] + F_Mult_t ["DEC_BOILER_WOOD",t] + F_Mult_t ["DEC_BOILER_OIL",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Boilers" , "Heat LT Dec", sum{i in BOILERS, t in PERIODS}(layers_in_out[i,"HEAT_LOW_T_DECEN"] * F_Mult_t [i, t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["DHN_BOILER_GAS",t] + F_Mult_t ["DHN_BOILER_WOOD",t] + F_Mult_t ["DHN_BOILER_OIL",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Boilers" , "DHN", sum{i in BOILERS, t in PERIODS}(layers_in_out[i,"HEAT_LOW_T_DHN"] * F_Mult_t [i, t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}((F_Mult_t ["IND_BOILER_GAS",t] + F_Mult_t ["IND_BOILER_WOOD",t] + F_Mult_t ["IND_BOILER_OIL",t] + F_Mult_t ["IND_BOILER_COAL",t] + F_Mult_t ["IND_BOILER_WASTE",t]) * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "Boilers" , "Heat HT", sum{i in BOILERS, t in PERIODS}(layers_in_out[i,"HEAT_HIGH_T"] * F_Mult_t [i, t] * t_op [t]) / 1000 , "Heat HT", "#DC143C", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
# DHN 
for{{0}: sum{t in PERIODS}(End_Uses ["HEAT_LOW_T_DHN", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "DHN" , "Heat LT DHN", sum{t in PERIODS}(sum {i in TECHNOLOGIES diff STORAGE_TECH} (layers_in_out[i, "HEAT_LOW_T_DHN"] * F_Mult_t [i, t] * t_op [t]) - Losses ["HEAT_LOW_T_DHN",t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
for{{0}: sum{t in PERIODS}(Losses ["HEAT_LOW_T_DHN", t] * t_op [t]) > 10}{
	printf "%s,%s,%.2f,%s,%s,%s\n", "DHN" , "Loss DHN", sum{t in PERIODS}(Losses ["HEAT_LOW_T_DHN",t] * t_op [t]) / 1000 , "Heat LT", "#FA8072", "TWh" >> "energy/energyscope-MILP/output/sankey/input2sankey.csv";
}
