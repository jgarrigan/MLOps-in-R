
# LOAD PACKAGES -----------------------------------------------------------


pacman::p_load(
  tidyverse,
  tidymodels,
  palmerpenguins,
  gt,
  ranger,
  brulee,         
  pins,
  vetiver,
  plumber,
  conflicted
)

tidymodels_conflicts()
conflict_prefer("penguins","palmerpenguins")
options(tidymodels.dark = TRUE)


# EDA ---------------------------------------------------------------------


penguins %>% 
  dplyr::filter(!is.na(sex)) %>% 
  ggplot(aes(x = flipper_length_mm,
             y = bill_length_mm,
             color = sex,
             size = body_mass_g)) + 
  geom_point(alpha = 0.5) + 
  facet_wrap(~species)

# PREPARE & SPLIT DATA ----------------------------------------------------

# REMOVE ROWS WITH MISSING SEX, EXCLUDE YEAR AND ISLAND
penguins_df <- 
  penguins %>% 
  drop_na(sex) %>% 
  select(-year,-island)

set.seed(123)

# SPLIT THE DATA INTO TRAIN AND TEST SETS STRATIFIED BY SEX
penguin_split <- initial_split(penguins_df, strata = sex)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)

# CREATE FOLDS FOR CROSS VALIDATION
penguin_folds <- vfold_cv(penguin_train)


# CREATE PREPROCESSING RECIPE ---------------------------------------------


penguin_rec <-
  recipe(sex ~ ., data = penguin_train) %>% 
  step_YeoJohnson(all_numeric_predictors()) %>% 
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
    models = list(glm = glm_spec,
                  tree = tree_spec,
                  torch = mlp_brulee_spec)
  ) %>% 
  workflow_map("tune_bayes",
               iter = 50L,
               resamples = penguin_folds,
               control = bayes_control)


# COMPARE MODEL RESULTS ---------------------------------------------------


rank_results(workflow_set,
             rank_metric = "roc_auc",
             select_best = TRUE) %>% 
  gt()

# PLOT MODEL PERFORMANCE
workflow_set %>% 
  autoplot()


# FINALIZE MODEL FIT ------------------------------------------------------

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


# CREATE VETIVER MODEL & API ----------------------------------------------


final_fit_to_deploy <- final_fit %>% 
  extract_workflow()

v <- vetiver_model(final_fit_to_deploy,
                   model_name = "penguin_model")

model_board <- board_folder()