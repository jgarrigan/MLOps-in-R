# PROJECT OBJECTIVE: CLASSIFICATION OF SPECIES USING PALMER PENGUINS DATA

# LOAD PACKAGES -----------------------------------------------------------

if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  tidyverse,
  skimr,
  tidymodels,
  palmerpenguins,
  gt,
  ranger,
  brulee,
  pins,
  vetiver,
  plumber,
  conflicted,
  usethis,
  themis,
  googleCloudStorageR,
  googleCloudRunner,
  googleAuthR,
  gargle,
  tune,
  finetune,
  doMC
)


# tidymodels_conflicts()
conflict_prefer("penguins", "palmerpenguins")
options(tidymodels.dark = TRUE)


# EDA ---------------------------------------------------------------------

skimr::skim(penguins)

# https://rpubs.com/StatsGary/tidmodels_and_vetiver_from_scratch

# COUNT OF SPECIES BY GENDER
penguins %>%
  drop_na() %>%
  count(sex, species) %>%
  ggplot() +
  geom_col(aes(x = species, y = n, fill = species)) +
  geom_label(aes(x = species, y = n, label = n)) +
  scale_fill_manual(values = c("darkorange", "purple", "cyan4")) +
  facet_wrap(~sex) +
  theme_minimal() +
  labs(title = "Penguins Species ~ Gender")

# SCATTER PLOT OF BODY MASS TO FLIPPER LENGTH
ggplot(
  data = penguins,
  aes(
    x = flipper_length_mm,
    y = body_mass_g
  )
) +
  geom_point(
    aes(
      color = species,
      shape = species
    ),
    size = 3,
    alpha = 0.8
  ) +
  # theme_minimal() +
  scale_color_manual(values = c("darkorange", "purple", "cyan4")) +
  labs(
    title = "Penguin size, Palmer Station LTER",
    subtitle = "Flipper length and body mass for Adelie, Chinstrap and Gentoo Penguins",
    x = "Flipper length (mm)",
    y = "Body mass (g)",
    color = "Penguin species",
    shape = "Penguin species"
  ) +
  theme_minimal()

# SCATTER PLOT OF PENGUIN SIZE WITH RESPECT TO ISLAND
ggplot(
  data = penguins,
  aes(
    x = flipper_length_mm,
    y = body_mass_g
  )
) +
  geom_point(
    aes(
      color = island,
      shape = species
    ),
    size = 3,
    alpha = 0.8
  ) +
  # theme_minimal() +
  scale_color_manual(values = c("darkorange", "purple", "cyan4")) +
  labs(
    title = "Penguin size, Palmer Station LTER",
    subtitle = "Flipper length and body mass for each island",
    x = "Flipper length (mm)",
    y = "Body mass (g)",
    color = "Penguin island",
    shape = "Penguin species"
  ) +
  theme_minimal()

# HOW DOES THE BODY MASS BETWEEN SEXES COMPARE
ggplot(data = penguins) +
  geom_point(mapping = aes(x = sex, y = body_mass_g, color = sex)) +
  labs(title = "Comparing The Sex And Body Mass Of The Palmer Penguins") +
  theme(text = element_text(size = 18))

# WHAT SPECIES ARE ON EACH ISLAND
ggplot(data = penguins) +
  geom_bar(mapping = aes(x = island, fill = species)) +
  labs(title = "Population of Penguin species on each Island", y = "count of species") +
  theme(text = element_text(size = 18))

# IS THERE A CLASS IMBALANCE IN THE DEPENDENT VARIABLE?
penguins %>%
  dplyr::filter(!is.na(sex)) %>%
  ggplot(aes(x = species)) +
  geom_bar(aes(fill = species)) +
  theme(legend.position = "none")

# CAN FLIPPER LENGTH AND BILL LENGTH BE USED TO SEPARATE THE CLASSES?
penguins %>%
  dplyr::filter(!is.na(sex)) %>%
  ggplot(aes(
    x = flipper_length_mm,
    y = bill_length_mm,
    color = sex,
    size = body_mass_g
  )) +
  geom_point(alpha = 0.5) +
  facet_wrap(~species)

# PREPARE & SPLIT DATA ----------------------------------------------------

# REMOVE ROWS WITH MISSING SEX, EXCLUDE YEAR AND ISLAND
penguins_df <-
  penguins %>%
  drop_na(sex) %>%
  select(-year, -island)

set.seed(123)

# SPLIT THE DATA INTO TRAIN AND TEST SETS STRATIFIED BY SEX
penguin_split <- initial_split(penguins_df, strata = sex, prop = 3 / 4)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)

# CREATE FOLDS FOR CROSS VALIDATION
penguin_folds <- vfold_cv(penguin_train)


# CREATE PREPROCESSING RECIPE ---------------------------------------------

penguin_rec <-
  recipe(sex ~ ., data = penguin_train) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  themis::step_upsample(species) %>%
  step_dummy(species) %>%
  step_normalize(all_numeric_predictors())


# MODEL SPECIFICATION -----------------------------------------------------

# LOGISTIC REGRESSION
glm_spec <-
  # L1 REGULARISATION
  logistic_reg(penalty = 1) %>%
  set_engine("glm")

# RANDOM FOREST
tree_spec <-
  rand_forest(min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# NEURAL NETWORK WITH TORCH
mlp_brulee_spec <-
  mlp(
    hidden_units = tune(),
    epochs = tune(),
    penalty = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("brulee") %>%
  set_mode("classification")


# MODEL FITTING AND HYPER PARAMETER TUNING --------------------------------

# # REGISTER PARALLEL CORES
# registerDoMC(cores = 2)

# BAYESIAN OPTIMIZATION FOR HYPER PARAMETER TUNING
bayes_control <- control_bayes(
  no_improve = 10L,
  time_limit = 20,
  save_pred = TRUE,
  verbose = TRUE
)

# FIT ALL THREE MODELS WITH HYPER PARAMETER TUNING
workflow_set <-
  workflow_set(
    preproc = list(penguin_rec),
    models = list(
      glm = glm_spec,
      tree = tree_spec,
      torch = mlp_brulee_spec
    )
  ) %>%
  workflow_map("tune_bayes",
    iter = 50L,
    resamples = penguin_folds,
    control = bayes_control
  )


# COMPARE MODEL RESULTS ---------------------------------------------------

rank_results(workflow_set,
  rank_metric = "roc_auc",
  select_best = TRUE
) %>%
  gt()

# PLOT MODEL PERFORMANCE
workflow_set %>%
  autoplot()


# FINALIZE MODEL FIT ------------------------------------------------------

# SELECT THE LOGISTIC MODEL GIVEN THAT ITS A SIMPLER MODEL AND PERFORMANCE
# IS SIMILAR TO THE NUERAL NET MODEL
best_model_id <- "recipe_glm"

# SELECT BEST MODEL
best_fit <-
  workflow_set %>%
  extract_workflow_set_result(best_model_id) %>%
  select_best(metric = "accuracy")

# CREATE WORKFLOW FOR BEST MODEL
final_workflow <-
  workflow_set %>%
  extract_workflow(best_model_id) %>%
  finalize_workflow(best_fit)

final_fit <-
  final_workflow %>%
  last_fit(penguin_split)

# FINAL FIT METRICS
final_fit %>%
  collect_metrics() %>%
  gt()

final_fit %>%
  collect_predictions() %>%
  roc_curve(sex, .pred_female) %>%
  autoplot()

final_fit_to_deploy <- final_fit %>%
  extract_workflow()

# VERSION WITH VETIVER ----------------------------------------------------

# INITIALISE VETIVER MODEL OBJECT
v <- vetiver_model(final_fit_to_deploy,
  model_name = "logistic_regression_model"
)

v

gcs_auth(json_file = "single-azimuth-397219-a13fa7c978de.json")

gcs_list_buckets(projectId = "single-azimuth-397219")

model_board <- board_gcs(bucket = "ml_model_bucket_jg")

model_board %>% vetiver_pin_write(board = model_board, vetiver_model = v)

pr() %>%
  vetiver_api(v, debug = TRUE) # %>%
# pr_run()

# PREPARE DOCKERFILE FOR DEPLOYMENT FOR GOOGLE CLOUD RUN
vetiver_prepare_docker(board = model_board, name = "logistic_regression_model")

# INSTALL DOCKER FOR WINDOWS AND RUN THE COMMAND BELOW FROM THE TERMINAL
docker build -t penguins .

docker run --env-file C:/Users/John/Documents/.Renviron --rm -p 8000:8000 penguins

endpoint <- vetiver_endpoint("http://0.0.0.0:8000/predict")

endpoint
