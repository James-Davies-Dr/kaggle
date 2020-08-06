library(tidymodels)
library(tidyverse)
library(gridExtra)
library(moments)
library(outliers)
library(mice)
library(janitor)

test <- read.csv("~/R/tidymodels/kaggle/test.csv")
train <- read.csv("~/R/tidymodels/kaggle/train.csv")

clean_names(train) -> train
str(train)
train <- train %>% 
  select(-c(name,ticket,cabin,embarked)) %>% 
  mutate(pclass = as.factor(pclass)) %>% 
  mutate(survived = as.factor(survived))


titanic_prep <- recipe(survived ~., data = train) %>%
  step_meanimpute(fare) %>%
  step_medianimpute(age, sib_sp, parch) %>%
  step_modeimpute(sex) %>%
  step_knnimpute(pclass) %>%
  step_dummy(pclass, sex) %>%
  update_role(passenger_id, new_role = "ID") %>% 
  prep(strings_as_factors = FALSE) 

titanic_prep

gb_model <- boost_tree(
  trees = 1000,
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(), 
  sample_size = tune(), 
  mtry = tune(), 
  learn_rate = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")
)

gb_model

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), train),
  learn_rate(),
  size = 30
)

xgb_grid


titanic_wf <- workflow() %>% 
  add_model(gb_model) %>% 
  add_recipe(titanic_prep)

titanic_wf

set.seed(123)
tt_folds <- vfold_cv(train, strata = survived)

tt_folds

set.seed(234)

xgb_res <- tune_grid(
  titanic_wf,
  resamples = tt_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)

collect_metrics(xgb_res) %>% 
  arrange(-mean)


xgb_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, mtry:sample_size) %>%
  pivot_longer(mtry:sample_size,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")


show_best(xgb_res, "roc_auc")
best_auc <- select_best(xgb_res, "roc_auc")
best_auc

final_xgb <- finalize_workflow(
  xgb_wf,
  best_auc)

final_xgb
final_xgb %>%
  fit(data = train) %>%
  pull_workflow_fit() %>%
  vip(geom = "point")
final_res <- last_fit(final_xgb, test)


library(vip)
install.packages("vip")


predict.xgb
