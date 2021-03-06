# context("test_bartMachine.R: bartMachine")

# if (FALSE) {
#  setwd("..")
#  getwd()
#  library("devtools")
#  document()
#  load_all("./") # load all R files in /R and datasets in /data. Ignores NAMESPACE:
#  # devtools::check() # runs full check
#  setwd("..")
#  install("sl3", build_vignettes = FALSE, dependencies = FALSE) # INSTALL W/ devtools:
# }

# library(testthat)
# library(sl3)

## generate Friedman data
# set.seed(11)
# n  = 200
# p = 5
# X = data.frame(matrix(runif(n * p), ncol = p))
# y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n)
# data<-cbind.data.frame(y,X)

# covars=names(data)[2:6]
# outcome=names(data)[1]

# task <- sl3_Task$new(data, covariates = covars, outcome = outcome)

# test_that("Lrnr_bartMachine gives the same thing as bartMachine", {

#  bart_learner<-Lrnr_bartMachine$new()
#  bart_fit<-bart_learner$train(task)
#  mean_pred_sl3<-mean(bart_fit$predict(task))

## build BART regression model
# bart_machine = bartMachine(X, y)
# mean_pred<-mean(predict(bart_machine, X))

# expect_true(mean_pred_sl3 - mean_pred < 0.5)
# expect_true(bart_fit$.__enclos_env__$private$.fit_object$PseudoRsq < 10)
# })
