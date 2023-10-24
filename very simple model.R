library(tidymodels)
library(googleCloudStorageR)
library(gargle)
library(vetiver)
library(pins)
library(plumber)

# Authentication
options(gargle_oauth_email = "my_email_address@gmail.com")
scope <- c("https://www.googleapis.com/auth/cloud-platform")
token <- token_fetch(scopes = scope, credentials = "C:/Users/me/Documents/googlecloudrunner-auth-key.json")
gcs_auth(token = token)

data(mtcars)
mtcars_split <- initial_split(mtcars, prop = 0.8)
train_data <- training(mtcars_split)
test_data <- testing(mtcars_split)

linear_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

linear_wf <- workflow() %>%
  add_formula(mpg ~ .) %>%
  add_model(linear_spec)

linear_fit <- fit(linear_wf, data = train_data)

linear_vetiver <- vetiver_model(linear_fit, "linear_model_mtcars")

board <- board_gcs(bucket = Sys.getenv("GCS_DEFAULT_BUCKET"), versioned = FALSE)

vetiver_pin_write(board, linear_vetiver)

plumber::pr() %>%
  vetiver_api(linear_vetiver, debug = TRUE) %>% 
  pr_run(port = 8080)

vetiver_write_plumber(board, "linear_model_mtcars", rsconnect = FALSE)

vetiver_write_docker(linear_vetiver)

#Set up a Docker repository in Artifact Registry in your GCP project

#gcloud auth configure-docker 

#docker build -t mtcars-model .

#docker run -v C:/Users/me/Documents/googlecloudrunner-auth-key.json:/app/credentials/my-service-account-key.json linear_vetiver

#docker tag mtcars-model gcr.io/ml-ops-with-r/mtcars-model:v1

#docker push gcr.io/ml-ops-with-r/mtcars-model:v1

#gcloud beta auth configure-docker europe-west1-docker.pkg.dev 

endpoint <- vetiver_endpoint("https://linear-model-uqf3dpyyza-ew.a.run.app/predict")

endpoint

new_car <- tibble(cyl = 4,  disp = 200, 
                  hp = 100, drat = 3,
                  wt = 3,   qsec = 17, 
                  vs = 0,   am = 1,
                  gear = 4, carb = 2)

predict(endpoint, new_car)

