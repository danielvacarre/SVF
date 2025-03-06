#' Creates a new instance of the SVF class
#'
#' This function defines the base SVF model, with common properties and functions.
#' It acts as an abstract class and should not be instantiated directly.
#'
#' @param method SVF method to be used.
#' @param inputs Inputs to evaluate in the dataset.
#' @param outputs Outputs to evaluate in the dataset.
#' @param data Dataset to evaluate.
#' @param C Values of the model's hyperparameter C.
#' @param eps Values of the model's epsilon hyperparameter.
#' @param d Value of the model's hyperparameter d.
#'
#' @return An object of class 'SVF'.
#'
#' @export
SVF <- function(method, inputs, outputs, data, C, eps, d) {
  if (!is.character(method) || length(method) != 1) stop("The 'method' parameter must be a single character string.")
  if (!is.vector(inputs) || !is.character(inputs)) stop("The 'inputs' parameter must be a character vector.")
  if (!is.vector(outputs) || !is.character(outputs)) stop("The 'outputs' parameter must be a character vector.")
  if (!is.data.frame(data)) stop("The 'data' parameter must be a data frame.")
  if (!is.numeric(C) || any(C <= 0)) stop("The 'C' parameter must be a numeric vector with positive values.")
  if (!is.numeric(eps) || length(eps) != 1 || eps < 0) stop("The 'eps' parameter must be a positive number.")
  if (!is.numeric(d) || length(d) != 1 || d <= 0) stop("The 'd' parameter must be a positive number.")

  structure(list(
    method = method, inputs = inputs, outputs = outputs, data = data,
    C = C, eps = eps, d = d, grid = NULL, model = NULL, model_d = NULL, solution = NULL, name = NULL
  ), class = "SVF")
}

#' Estimation of a DMU
#'
#' This function calculates the output estimations for a specific DMU using the SVF model.
#'
#' @param svf SVF object.
#' @param dmu A numeric vector with the characteristics of the DMU (Must have the same number of elements as the inputs).
#'
#' @return A numeric vector with the estimations for each output.

#' @export
get_estimation_svf <- function(svf, dmu) {

  if (!inherits(svf, "SVF")) stop("The 'svf' parameter must be an object of class 'SVF'.")

  if (length(dmu) != length(svf$inputs)) {
    stop("The number of inputs for the DMU does not match the number of inputs for the problem.")
  }

  dmu <- as.numeric(dmu)

  dmu_cell <- search_dmu_grid(svf$grid, dmu)

  id_cells <- svf$grid$df_grid$id_cells
  cell_index <- which(apply(id_cells, 1, function(row) all(row == dmu_cell)))

  if (length(cell_index) == 0) {
    stop("The DMU cell was not found in the grid.")
  }

  phi <- unlist(svf$grid$df_grid$phi[[cell_index]])

  solution <- svf$model$xopt
  n_var <- length(svf$grid$data_grid$phi[[1]][[1]])
  n_out <- length(svf$outputs)
  n_w_vars <- n_out * n_var
  w_solution <- solution[1:n_w_vars]

  mat_w <- vector("list", n_out)
  for (out in seq_len(n_out)) {
    start_index <- (out - 1) * n_var + 1
    end_index <- out * n_var
    mat_w[[out]] <- round(w_solution[start_index:end_index], 6)
  }

  prediction_list <- numeric(n_out)
  for (out in seq_along(svf$outputs)) {
    w <- unlist(mat_w[[out]])
    prediction_list[out] <- round(sum(w * phi), 3)
  }

  return(prediction_list)
}

#' Prints the optimization model in a human-readable format
#'
#' This function prints the optimization problem of the SSVF model in a format similar to the one you provided.
#' It is fully modular, meaning it works for any SSVF optimization problem.
#'
#' @param svf SSVF object with the trained model.
#' @param bounds A list of bounds for each variable, e.g., list(c(0, 10), c(1, 5)).
#'
#' @export
print_ssvf_model <- function(svf, bounds = NULL) {
  # Ensure the model has been trained
  if (is.null(svf$model)) {
    stop("The model has not been trained. Please run `train_ssvf` first.")
  }

  # Retrieve the number of outputs, number of variables and number of observations
  n_out <- length(svf$outputs)
  n_var <- length(svf$grid$data_grid$phi[[1]][[1]])
  n_obs <- nrow(svf$data)

  # Create variable names for 'w' and 'xi' based on the sizes
  var_names <- c()

  # Generate variable names for w (weights) and xi (slack variables)
  for (out in 1:n_out) {
    for (var in 1:n_var) {
      var_names <- c(var_names, paste("w", out, var, sep = "_"))
    }
  }

  for (out in 1:n_out) {
    for (obs in 1:n_obs) {
      var_names <- c(var_names, paste("xi", out, obs, sep = "_"))
    }
  }

  # Print the objective function (maximize)
  cat("Maximize\n")
  obj_terms <- sapply(1:length(svf$model$cvec), function(i) {
    paste(sprintf("%.2f %s", svf$model$cvec[i], var_names[i]), collapse = " + ")
  })
  cat(" obj:", paste(obj_terms, collapse = " + "), "\n")

  # Print the constraints (Subject To)
  cat("Subject To\n")
  n_constraints <- length(svf$model$bvec)
  for (i in 1:n_constraints) {
    constraint_terms <- sapply(1:ncol(svf$model$Amat), function(j) {
      paste(sprintf("%.2f %s", svf$model$Amat[i, j], var_names[j]), collapse = " + ")
    })
    cat(sprintf(" c%d: %s <= %.2f\n", i, paste(constraint_terms, collapse = " + "), svf$model$bvec[i]))
  }

  # Print the bounds (Bounds)
  if (!is.null(bounds)) {
    cat("Bounds\n")
    for (i in 1:length(bounds)) {
      cat(sprintf(" %.2f <= %s <= %.2f\n", bounds[[i]][1], var_names[i], bounds[[i]][2]))
    }
  } else {
    cat("Bounds\n")
    for (i in 1:length(var_names)) {
      cat(sprintf(" 0 <= %s <= Inf\n", var_names[i]))
    }
  }

  # Print general variables (General)
  cat("General\n")
  for (i in 1:length(var_names)) {
    cat(sprintf(" %s\n", var_names[i]))  # Assuming all variables are general
  }

  # End the model
  cat("End\n")
}

