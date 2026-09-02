# =============================================================
# Projection des DALYs par évolution démographique (taux figé)
# =============================================================

# 1) Nettoyer l'environnement
rm(list = ls())

# 2) Packages
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(readxl)
library(openxlsx)
library(purrr)

# -------------------------------------------------------------
# Paramètres (à adapter)
# -------------------------------------------------------------
pop_path  <- "//file3.intranet.chuv/data3/UNISANTE_DESS/S_ECOSAN/3_RECHERCHE_ET_MANDATS/TRUE COST OF FOOD - SINERGIA/WP2/WP2_Health_Externalities/data/raw/Calculator/__py_debug_temp_var_1844187303.xlsx"
daly_path <- "//file3.intranet.chuv/data3/UNISANTE_DESS/S_ECOSAN/3_RECHERCHE_ET_MANDATS/TRUE COST OF FOOD - SINERGIA/WP2/3. Articles/7. Calculator/1. Data/GBD/DALYs_age_sex_2023.csv"

daly_year <- 2023   # année des DALYs GBD
  # année de population servant de dénominateur au taux (idéalement = daly_year)

# =============================================================
# 3) Importer la projection de population et la mettre au format long
# =============================================================
sheets <- c("level 1", "level 2", "level 3", "level 4")

Population <- lapply(seq_along(sheets), function(i) {
  read_excel(pop_path, sheet = sheets[i]) %>% mutate(Scenario = i)
}) %>% bind_rows()

Population <- Population %>%
  rename(Year = Years) %>%                                   # adapter si la colonne s'appelle déjà "Year"
  pivot_longer(cols = starts_with("lfs_demography_"),        # sélection par motif (robuste à l'ordre)
               names_to = "Age", values_to = "Population") %>%
  mutate(Age = Age %>%
           str_remove("^lfs_demography_") %>%
           str_remove("\\[inhabitants\\]$")) %>%
  separate(Age, into = c("Sex", "Age"), sep = "-", extra = "merge") %>%
  mutate(
    Sex = recode(Sex, female = "Female", male = "Male"),
    Age = recode(Age,
                 "below19"  = "0-19",
                 "age20-29" = "20-29",
                 "age30-54" = "30-54",
                 "age55-64" = "55-64",
                 "above65"  = "65+"),
    Age = factor(Age, levels = c("0-19", "20-29", "30-54", "55-64", "65+"))
  ) %>%
  dplyr::select(Year, Scenario, Sex, Age, Population)        # ne garder que l'essentiel

# =============================================================
# 4) Population de base (dénominateur du taux), indépendante du scénario
#    -> moyenne sur les scénarios (identiques en année de base en principe)
# =============================================================
base_pop <- Population %>%
  filter(Year == 2025) %>%
  group_by(Age, Sex) %>%
  summarise(Population_base = mean(Population), .groups = "drop")

# =============================================================
# 5) DALYs par âge, sexe et cause -> taux par habitant (figé)
# =============================================================
DALYs_raw <- read_csv(daly_path) %>%
  mutate(cause = ifelse(cause =="Tracheal, bronchus, and lung cancer","Tracheal bronchus and lung cancer",cause))

# --- IMPORTANT : garder UNE seule mesure / métrique / année / lieu ---
# Adapter les noms de colonnes au CSV GBD réel. Les blocs ci-dessous ne
# s'appliquent que si la colonne existe (évite une erreur si absente).
DALY_rate <- DALYs_raw
if ("measure"  %in% names(DALY_rate)) DALY_rate <- DALY_rate %>% filter(str_detect(measure, "DALY"))
if ("metric"   %in% names(DALY_rate)) DALY_rate <- DALY_rate %>% filter(metric == "Number")
if ("year"     %in% names(DALY_rate)) DALY_rate <- DALY_rate %>% filter(year == daly_year)
# if ("location" %in% names(DALY_rate)) DALY_rate <- DALY_rate %>% filter(location == "Switzerland")

DALY_rate <- DALY_rate %>%
  dplyr::select(Age = age, Sex = sex, cause, DALYs = val) %>%
  filter(Sex %in% c("Male", "Female")) %>%                   # exclure "Both"
  mutate(Age = str_replace_all(Age, " years", "")) %>%
  filter(!Age %in% c("All ages", "All Ages", "Age-standardized")) %>%
  mutate(Age = case_when(
    Age %in% c("<5", "5-9", "10-14", "15-19")               ~ "0-19",
    Age %in% c("20-24", "25-29")                            ~ "20-29",
    Age %in% c("30-34", "35-39", "40-44", "45-49", "50-54") ~ "30-54",
    Age %in% c("55-59", "60-64")                            ~ "55-64",
    TRUE                                                    ~ "65+"   # 65-69 ... 95+
  )) %>%
  group_by(Age, Sex, cause) %>%
  summarise(DALYs = sum(DALYs, na.rm = TRUE), .groups = "drop") %>%
  left_join(base_pop, by = c("Age", "Sex")) %>%
  mutate(DALYs_cap = DALYs / Population_base) %>%             # taux par habitant FIGÉ
  dplyr::select(Age, Sex, cause, DALYs_cap)

# =============================================================
# 6) Projection : taux figé x population projetée
# =============================================================
DALYs_proj <- Population %>%
  inner_join(DALY_rate, by = c("Age", "Sex"),
             relationship = "many-to-many") %>%
  mutate(DALYs = DALYs_cap * Population) %>%
  dplyr::select(Year, Scenario, Age, Sex, cause, Population, DALYs)

# =============================================================
# 7) Agrégations
# =============================================================
# Totaux de population PROPRES (sans les causes -> pas de double comptage)
Pop_total <- Population %>%
  group_by(Year, Scenario) %>%
  summarise(Population = sum(Population), .groups = "drop")

# DALYs par cause
Sum_Data <- DALYs_proj %>%
  group_by(Year, Scenario, cause) %>%
  summarise(DALYs = sum(DALYs), .groups = "drop") %>%
  rename(Disease = cause)%>%
  rename(Forecast_DALY = DALYs)

# DALYs totaux (toutes causes) + population propre
Total <- DALYs_proj %>%
  group_by(Year, Scenario) %>%
  summarise(DALYs = sum(DALYs), .groups = "drop") %>%
  left_join(Pop_total, by = c("Year", "Scenario"))%>%
  rename(Disease = cause)%>%
  rename(Forecast_DALY = DALYs)


write_csv(Sum_Data, file = "//file3.intranet.chuv/data3/UNISANTE_DESS/S_ECOSAN/3_RECHERCHE_ET_MANDATS/TRUE COST OF FOOD - SINERGIA/WP2/WP2_Health_Externalities/part3_counterfactual/Counterfactual_calculator/Projection/Projected_DALYs.csv")
