#' Cross-validation Pipeline (chain) of learners. 
#'
#' An extension of \code{\link{Pipeline}} that chains cross-claidate tasks in a computationally efficient, but statistically valid way.
#' The pipeline will be a mix of cross-validated steps (defined by \code{\link{Lrnr_cv}}) and non-cross-validated steps (defined by any other learner).
#' For the cross-validated steps, separate learner fits will be maintained for each fold, fit on separate training tasks. 
#' For other steps, there is one learner fit that is shared across folds, fit on a pooled validation task.
#' @docType class
#' @importFrom R6 R6Class
#' @export
#' @keywords data
#' @return Learner object with methods for training and prediction. See \code{\link{Lrnr_base}} for documentation on learners.
#' @format \code{\link{R6Class}} object.
#' @family Learners
#' 
#' @section Parameters:
#' \describe{
#'   \item{\code{...}}{Parameters should be individual \code{Learner}s, in the order they should be applied.}
#' }
#' @template common_parameters
#' @importFrom assertthat assert_that is.count is.flag
Pipeline_revere <- R6Class(classname = "Pipeline_revere",
                    inherit= Pipeline,
                    portable = TRUE,
                    class = TRUE,
                    public = list(

                      extend = function(new_learner){
                        if(is.list(new_learner) && length(new_learner)==1){
                          new_learner <- new_learner[[1]]
                        }
                        if(!inherits(new_learner, "Lrnr_base") || inherits(new_learner, "Pipeline")){
                          stop("currently, pipelines can only be extended by one non-pipeline learner")
                        }
                        
                        # extend the learner list in params
                        private$.params$learners <- c(private$.params$learners, new_learner)
                        
                        if(self$is_trained){
                          # if we're already fit, extend the fit list too
                          if(!new_learner$is_trained){
                            # if the new learner isn't trained, train it.
                            last_fit <- tail(self$fit_object$learner_fits, 1)[[1]]
                            last_task <- last_fit$chain()
                            new_learner <- new_learner$train(last_task)
                          }
                          
                          private$.fit_object <- c(private$.fit_object, new_learner)
                        }
                        
                        return(self)
                      }),
                    active = list(
                      name = function(){
                        learners=self$params$learners
                        learner_names=sapply(learners,function(learner)learner$name)
                        name = paste(learner_names, collapse="___")
                        
                        return(name)
                      }
                    ),
                    private = list(
                      .train_sublearners = function(task){
                        fs_task <- task$fold_specific_task()
                        learners <- self$params$learners
                        learner_names <- sapply(learners,function(learner)learner$name)
                        learner_fits <- as.list(rep(NA,length(learners)))
                        names(learner_fits) <- learner_names
                        
                        current_task <- task
                        current_fs_task <- fs_task
                        for(i in seq_along(learners)){
                          current_learner <- learners[[i]]
                          if(inherits(current_learner, "Lrnr_cv")){
                            # fit with split-specific task
                            fit <- delayed_learner_train(current_learner, current_fs_task)
                            next_fs_task <- delayed_learner_fit_chain(fit, current_fs_task, component="both")
                          } else {
                            # fit with validation task
                            fit <- delayed_learner_train(current_learner, current_task)  
                            next_fs_task <- delayed_learner_fit_chain(fit, current_fs_task)
                          }
                            
                          next_task <- delayed_learner_fit_chain(fit, current_task)
                          learner_fits[[i]] <- fit
                          
                          current_task <- next_task
                          current_fs_task <- next_fs_task
                        }
                        
                        return(bundle_delayed(learner_fits))
                        
                      },
                      .train = function(task, trained_sublearners) {
                        
                        fit_object <- list(learner_fits = trained_sublearners)
                        
                        return(fit_object)
                        
                      },
                      .predict = function(task){
                        
                        #prediction is just chaining until you get to the last fit, and then calling predict
                        learner_fits = private$.fit_object$learner_fits
                        next_task=task
                        for(i in seq_along(learner_fits)){
                          current_task=next_task
                          current_fit=learner_fits[[i]]
                          next_task=current_fit$base_chain(current_task)
                        }
                        
                        # current_task is now the task for the last fit, so we can just do this
                        predictions=current_fit$base_predict(current_task)
                        
                        return(predictions)
                      }
                      
                    )
                    
)
