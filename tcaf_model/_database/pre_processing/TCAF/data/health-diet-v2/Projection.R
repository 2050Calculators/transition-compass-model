# =============================================================================
# DALYs attribuables à l'alimentation : projections 2025-2050
# =============================================================================
# =============================================================================

# -----------------------------------------------------------------------------
# 0) Environnement
# -----------------------------------------------------------------------------

rm(list = ls())

library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(ggplot2)


# -----------------------------------------------------------------------------
# 1)  Chargement des données (changer avec vos chemins d'accès)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

##### Choix des facteurs de risques
Risk_factor <- c("Fruits", "Vegetables", "Whole_Grains", "Nuts",
                 "Milk", "Legumes", "Red_Meat", "Processed_Meat")

##### Cibles des recommandations suisses (g/jour)
Target <- data.frame(
  Fruits         = 240,
  Vegetables     = 360,
  Whole_Grains   = 180,
  Nuts           = 30,
  Milk           = 300,
  Legumes        = 150,
  Red_Meat       = 16,
  Processed_Meat = 16
)

##### Parts de la population qui adhèrent au régime cible
alpha <- c(1, 0.75, 0.5, 0.25)

Mean_Intake <- read_csv(
  "//file3.intranet.chuv/data3/UNISANTE_DESS/S_ECOSAN/3_RECHERCHE_ET_MANDATS/TRUE COST OF FOOD - SINERGIA/WP2/WP2_Health_Externalities/part3_counterfactual/Counterfactual_calculator/Mean_Intake.csv",
  show_col_types = FALSE
) %>%
  filter(Diet %in% Risk_factor)

PAF <- read_csv(
  "//file3.intranet.chuv/data3/UNISANTE_DESS/S_ECOSAN/3_RECHERCHE_ET_MANDATS/TRUE COST OF FOOD - SINERGIA/WP2/WP2_Health_Externalities/part3_counterfactual/Counterfactual_calculator/PAF_grid.csv",
  show_col_types = FALSE
) %>%
  filter(Risk_Factor %in% Risk_factor)

Projected_DALYs <- read_csv(
  "//file3.intranet.chuv/data3/UNISANTE_DESS/S_ECOSAN/3_RECHERCHE_ET_MANDATS/TRUE COST OF FOOD - SINERGIA/WP2/WP2_Health_Externalities/part3_counterfactual/Counterfactual_calculator/Projected_DALYs.csv",
  show_col_types = FALSE
)


# -----------------------------------------------------------------------------
# 2) Scénarios d'apport
# -----------------------------------------------------------------------------
# Deux trajectoires, empilées dans une seule table :
#   Reference: l'apport reste au niveau de 2025 jusqu'en 2050
#   Scenario1: l'apport va linéairement de 2025 (niveau actuel) à 2050 (cible)

Target_long <- Target %>%
  pivot_longer(everything(), names_to = "Diet", values_to = "target")

Scenarios <- Mean_Intake %>%
  left_join(Target_long, by = "Diet") %>%
  crossing(Year = 2025:2050) %>%
  mutate(frac = (Year - 2025) / (2050 - 2025)) %>%   # 0 en 2025, 1 en 2050
  transmute(
    Diet, Year,
    Reference = mean,
    Scenario1 = mean + frac * (target - mean)
  ) %>%
  pivot_longer(c(Reference, Scenario1),
               names_to = "scenario", values_to = "intake")


# -----------------------------------------------------------------------------
# 3) PAF pour chaque intake
# -----------------------------------------------------------------------------
# On lit la valeur de la PAF sur la courbe


## 3.1 ) Une courbe par maladie × aliment, rangée dans une colonne-liste.
Courbes <- PAF %>%
  group_by(Disease, Diet = Risk_Factor) %>%
  summarise(courbe = list(pick(x, PAF_mean, PAF_lower, PAF_upper)),
            .groups = "drop")

## 3.2 ) Création de la fonction qui approxime la courbe PAF 

lire_courbe <- function(x_grille, y_grille, x) {
  approx(x_grille, y_grille, xout = x, rule = 2)$y
}

## 3.3 ) Pour chaque ligne, on évalue la PAF correspondante à l'apport selon l'aliment x maladie de la courbe qui correspond à l'aliment x maladie 

PAF_eval <- Scenarios %>%
  left_join(Courbes, by = "Diet", relationship = "many-to-many") %>%
  mutate(
    PAF_mean  = map2_dbl(courbe, intake, ~ lire_courbe(.x$x, .x$PAF_mean,  .y)),
    PAF_lower = map2_dbl(courbe, intake, ~ lire_courbe(.x$x, .x$PAF_lower, .y)),
    PAF_upper = map2_dbl(courbe, intake, ~ lire_courbe(.x$x, .x$PAF_upper, .y))
  ) %>%
  select(-courbe)


# -----------------------------------------------------------------------------
# 4) Pour toutes les valeurs du scénario contrefactuel, on estime la PIF.
# PIF : Permet d'estimer les DALYs evitable en comporant le scénario de référence (BAU) au scénario contrefactuel 
#  étant donné que la PAF est un example précis de la PIF, on peut estimer la PAF avec une transformation simple  
# PIF = (PAF_reference - PAF_contrefactuel) / (1 - PAF_scenario)

pif <- function(paf_ref, paf_scen) {
  pmax((paf_ref - paf_scen) / (1 - paf_scen), 0)   
}

PAF_ref_rf <- PAF_eval %>%
  filter(scenario == "Reference") %>%
  select(Diet, Disease, Year,
         PAF_ref_mean  = PAF_mean,
         PAF_ref_lower = PAF_lower,
         PAF_ref_upper = PAF_upper)

PIF_rf <- PAF_eval %>%
  left_join(PAF_ref_rf, by = c("Diet", "Disease", "Year")) %>%
  mutate(
    PIF_mean  = pif(PAF_ref_mean,  PAF_mean),
    PIF_lower = pif(PAF_ref_lower, PAF_lower),
    PIF_upper = pif(PAF_ref_upper, PAF_upper)
  )


# -----------------------------------------------------------------------------
# 5) Combinaison des aliments, par maladie
# -----------------------------------------------------------------------------
# Une maladie a plusieurs facteurs de risque alimentaires. On les combine de
# façon multiplicative : 1 - (1-p1)(1-p2)... — c'est la formule GBD standard.
# Elle contrôle la coexposition : chaque cas n'est attribué qu'à un seul facteur.

Par_maladie <- PIF_rf %>%
  group_by(scenario, Year, Disease) %>%
  summarise(
    PAF_ref_comb = 1 - prod(1 - PAF_ref_mean),   
    PAF_ref_lo   = 1 - prod(1 - PAF_ref_lower),
    PAF_ref_hi   = 1 - prod(1 - PAF_ref_upper),
    PIF_comb     = 1 - prod(1 - PIF_mean),       
    PIF_lo       = 1 - prod(1 - PIF_lower),
    PIF_hi       = 1 - prod(1 - PIF_upper),
    n_aliments   = n(),
    .groups = "drop"
  )


# -----------------------------------------------------------------------------
# 6) Séries : part de la population qui adhère
# -----------------------------------------------------------------------------
# On suppose qu'une fraction alpha de la population adopte intégralement le
# régime cible, et que le reste ne change rien. Le risque moyen est alors un
# mélange linéaire entre les deux sous-populations, donc :
#
#   PIF(alpha) = alpha × PIF(adhésion totale)
#

serie_ref <- Par_maladie %>%
  filter(scenario == "Reference") %>%
  mutate(adherence = 0, serie = "Référence")

serie_fbdg <- Par_maladie %>%
  filter(scenario == "Scenario1") %>%
  crossing(adherence = alpha) %>%
  mutate(
    PIF_comb = adherence * PIF_comb,
    PIF_lo   = adherence * PIF_lo,
    PIF_hi   = adherence * PIF_hi,
    serie    = paste0("FBDG ", adherence * 100, "%")
  )

Series <- bind_rows(serie_ref, serie_fbdg)


# -----------------------------------------------------------------------------
# 7) DALYs attribuables et évitables
# -----------------------------------------------------------------------------
# attribuable = ce que coûte le régime actuel
# évité       = ce que le scénario permet d'éviter
# résiduel    = ce qui reste attribuable une fois le scénario appliqué

Resultats <- Series %>%
  left_join(Projected_DALYs, by = c("Year", "Disease")) %>%
  mutate(
    attribuable    = PAF_ref_comb * Forecast_DALY,
    attribuable_lo = PAF_ref_lo   * Forecast_DALY,
    attribuable_hi = PAF_ref_hi   * Forecast_DALY,
    
    evite    = PIF_comb * Forecast_DALY,
    evite_lo = PIF_lo   * Forecast_DALY,
    evite_hi = PIF_hi   * Forecast_DALY,
    
    residuel    = attribuable    - evite,
    residuel_lo = attribuable_lo - evite_lo,
    residuel_hi = attribuable_hi - evite_hi
  )


# -----------------------------------------------------------------------------
# 8) Agrégation pour les graphiques
# -----------------------------------------------------------------------------

par_maladie <- Resultats %>%
  select(serie, adherence, Year, Disease, total_dalys = Forecast_DALY,
         attribuable, attribuable_lo, attribuable_hi,
         evite, evite_lo, evite_hi,
         residuel, residuel_lo, residuel_hi)

par_total <- Resultats %>%
  group_by(serie, adherence, Year) %>%
  summarise(
    total_dalys    = sum(Forecast_DALY),
    attribuable    = sum(attribuable),
    attribuable_lo = sum(attribuable_lo),
    attribuable_hi = sum(attribuable_hi),
    evite          = sum(evite),
    evite_lo       = sum(evite_lo),
    evite_hi       = sum(evite_hi),
    residuel       = sum(residuel),
    residuel_lo    = sum(residuel_lo),
    residuel_hi    = sum(residuel_hi),
    .groups = "drop"
  ) %>%
  mutate(serie = factor(serie, levels = c("Référence", paste0("FBDG ", alpha * 100, "%"))))


# -----------------------------------------------------------------------------
# 9) Graphique de projection
# -----------------------------------------------------------------------------

ggplot(par_total, aes(Year, residuel, colour = serie)) +
  geom_line(linewidth = 1) +
  labs(title = "DALYs attribuables à l'alimentation",
       subtitle = "Suisse, 2025-2050, selon la part de la population qui adhère aux recommandations",
       y = "DALYs par an", x = NULL, colour = NULL) +
  theme_minimal()


