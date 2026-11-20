Needed for: wwp/timber split in energy intensity preprocessing (data_fxa_energy_demand.py).

What to download:
1. Go to https://www.fao.org/faostat/en/#data/FO (Forestry Production and Trade)
2. Filters: Countries = "European Union (27)" + all 27 EU member states;
   Elements = "Production"; Items = "Sawnwood + (Total)"; Years = 1990-2023
3. Download as CSV, save two files here:
   - FAOSTAT_sawnwood_EU27_aggregate.csv  (European Union (27) only)
   - FAOSTAT_sawnwood_EU27_countries.csv  (individual member states)

Why: PRODCOM EU27 aggregate for sawn timber (C16.10.11.xx) covers only ~10% of actual
EU27 production. FAOSTAT FO has reliable annual sawnwood data from national forest
statistics (~90 Mm3/yr x 0.6 t/m3 = ~54 Mt). Combined with PRODCOM chips+boards
(~43.6 Mt), total C16 ~97.6 Mt is used as the energy intensity denominator.
