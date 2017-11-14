library(testthat)
context("test_revere_pipeline.R -- Revere pipeline functionality")

library(sl3)
library(origami)
library(SuperLearner)
library(data.table)

data(cpp_imputed)
covars <- c("apgar1", "apgar5", "parity", "gagebrth", "mage", "meducyrs", "sexn")
outcome <- "haz"
task <- sl3_Task$new(cpp_imputed, covariates = covars, outcome = outcome)

set.seed(1234)
screen_glmnet <- Lrnr_pkg_SuperLearner_screener$new("screen.glmnet")
glm_learner <- Lrnr_glm$new()
cv_glm_learner <- make_learner(Lrnr_cv, glm_learner)
revere <- make_learner(Pipeline_revere, cv_glm_learner, glm_learner, cv_glm_learner, glm_learner)
debugonce(revere$.__enclos_env__$private$.train_sublearners)
revere_fit <- revere$.__enclos_env__$private$.train_sublearners(task)
