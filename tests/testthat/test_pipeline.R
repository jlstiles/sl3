library(testthat)
context("test_pipeline.R -- Basic pipeline functionality")

library(sl3)
library(origami)
library(SuperLearner)
library(data.table)

data(cpp_imputed)
covars <- c("apgar1", "apgar5", "parity", "gagebrth", "mage", "meducyrs", "sexn")
outcome <- "haz"
task <- sl3_Task$new(cpp_imputed, covariates = covars, outcome = outcome)

# standard pipe construction
set.seed(1234)
screen_glmnet <- Lrnr_pkg_SuperLearner_screener$new("screen.glmnet")
glm_learner <- Lrnr_glm$new()
pipe <- make_learner(Pipeline, screen_glmnet, glm_learner)
pipe_fit <- pipe$train(task)
pipe_preds <- pipe_fit$predict(task)

# operator pipe construction
set.seed(1234)
pipe_op <- screen_glmnet %|% glm_learner
pipe_op_fit <- pipe_op$train(task)
pipe_op_preds <- pipe_op_fit$predict()
test_that("The pipe operator %|% works", expect_equal(pipe_preds, pipe_op_preds))

# extendable pipelines
set.seed(1234)
pipe_op_extended <- pipe_op %|% glm_learner
test_that("An extension of an non-trained learner remains non-trained", expect_true(!pipe_op_extended$is_trained))
pipe_op_extended_fit <- pipe_op_extended$train(task)
pipe_op_extended_preds <- pipe_op_extended_fit$predict()

set.seed(1234)
pipe_op_fit_extended <- pipe_op_fit %|% glm_learner
test_that("An extension of an trained learner remains trained", expect_true(pipe_op_fit_extended$is_trained))
pipe_op_fit_extended_preds <- pipe_op_fit_extended$predict()
test_that("The pipe operator %|% can extend pipes", expect_equal(pipe_op_extended_preds, pipe_op_fit_extended_preds))

# custom chaining
chain_y <- function(learner, task){
  preds <- learner$predict(task) 
  pred_dt <- data.table(preds)
  if(ncol(pred_dt)>1){
    stop("chain_y should only be used with learners that return a single prediction per observation")
  }
  
  setnames(pred_dt, names(pred_dt), learner$name)
  
  # add predictions as new column
  new_col_names <- task$add_columns(learner$fit_uuid, pred_dt)
  # prediction becomes outcome
  return(task$next_in_chain(outcome = names(pred_dt), column_names = new_col_names))
}

glm_outcome <- customize_chain(glm_learner, chain_y)

pipe2 = make_learner(Pipeline, glm_outcome, glm_learner)
fit2 <- pipe2$train(task)


# just carry through two tasks -> the fold-specific one, and the validation one.
# if you see a Lrnr_cv, pass the fs one, otherwise, pass the validation one.
# need a fs_predict method for cv
# rewrite pipeline code to have private functions for "fit next learner" and "chain next learner"
# back to the "delayed fold problem" -- we need folds to define the Lrnr_cv subtasks, but sometimes task is delayed and has no folds
# we also don't necessarily know the folds at the time we define the Lrnr, especially in the case of fold_specific
# we do know the folds when we call train, but train doesn't take folds as an argument
# best option: provide folds to train directly.