#' Creates an SSVF object
#'
#' @param method SVF method to be used.
#' @param inputs Inputs to evaluate in the dataset.
#' @param outputs Outputs to evaluate in the dataset.
#' @param data Dataset to evaluate.
#' @param c Values of the model's hyperparameter C.
#' @param eps Values of the model's epsilon hyperparameter.
#' @param d Value of the model's hyperparameter d.
#'
#' @return An object of class SSVF.
#' @examples
#'
#' data <- data.frame(
#'   x1 = c(1, 2, 3, 10),
#'   x2 = c(2, 4, 3, 10),
#'   y1 = c(1, 1, 2, 3)
#' )
#'
#' method <- "ssvf"
#' inputs <- c("x1", "x2")
#' outputs <- c("y1")
#' c_value <- 2.0
#' eps_value <- 0
#' d_value <- 4
#'
#' svf_model <- SSVF(method, inputs, outputs, data, c_value, eps_value, d_value)
#'
#' svf_model <- train_ssvf(svf_model)
#'
#' print_ssvf_model(svf_model)
#'
#' svf_model <- solve_ssvf(svf_model)
#'
#' print(svf_model$solution$w)
#'
#' print(svf_model$solution$xi)
#'
#' @export
SSVF <- function(method, inputs, outputs, data, c, eps, d) {
  svf <- list(method = method, inputs = inputs, outputs = outputs, data = data, c = c, eps = eps, d = d)
  class(svf) <- c("SSVF", "SVF")
  return(svf)
}

#' Trains the SSVF model
#'
#' This function generates the SSVF model but does not solve the optimization problem.
#' It only prepares the model to be solved later.
#'
#' @param svf SSVF object.
#'
#' @return The SSVF object with the model configuration (without solution).
#'
#' @export
train_ssvf <- function(svf) {
  y_df <- svf$data[, svf$outputs, drop = FALSE]
  y <- as.matrix(y_df)

  n_out <- ncol(y_df)
  n_obs <- nrow(y_df)

  # Create the model grid
  svf$grid <- create_grid_svfgrid(SVFGrid(svf$data, svf$inputs, svf$outputs, svf$d))
  n_var <- length(svf$grid$data_grid$phi[[1]][[1]][[1]])

  total_variables <- n_out * n_var + n_out * n_obs

  # Create the objective function coefficient vector
  cvec <- c(rep(1, n_out * n_var), rep(svf$c, n_out * n_obs))

  # Construct the constraints
  Amat <- matrix(0, nrow = 2 * n_out * n_obs, ncol = total_variables)
  bvec <- vector("numeric", length = 2 * n_out * n_obs)
  sense <- rep("<=", 2 * n_out * n_obs)

  for (out in 1:n_out) {
    for (obs in 1:n_obs) {
      phi_vector <- svf$grid$data_grid$phi[[obs]][[1]][[out]]
      row_index1 <- (out - 1) * 2 * n_obs + (obs - 1) * 2 + 1
      row_index2 <- row_index1 + 1

      w_indices <- ((out - 1) * n_var + 1):((out - 1) * n_var + n_var)
      xi_index <- n_out * n_var + (out - 1) * n_obs + obs

      Amat[row_index1, w_indices] <- -phi_vector
      Amat[row_index2, w_indices] <- phi_vector
      Amat[row_index2, xi_index] <- -1

      bvec[row_index1] <- -y[obs, out]
      bvec[row_index2] <- y[obs, out] + svf$eps
    }
  }

  # Save the configuration in the SSVF object without solving it
  svf$model <- list(cvec = cvec, Amat = Amat, bvec = bvec, sense = sense)

  return(svf)
}

#' Solves the SSVF model
#'
#' This function solves the previously trained SSVF model.
#' It extracts the solution from the model after it has been trained.
#'
#' @param svf SSVF object with a trained model.
#'
#' @return A list containing the solutions for the 'w' and 'xi' variables.
#'
#' @export
solve_ssvf <- function(svf) {
  # Check if the model has been trained
  if (is.null(svf$model)) {
    stop("The model has not been trained. Please run `train_ssvf` first.")
  }

  # Extract the configuration of the optimization problem
  cvec <- svf$model$cvec
  Amat <- svf$model$Amat
  bvec <- svf$model$bvec
  sense <- svf$model$sense

  # Solve the optimization problem using lpSolve
  result <- lpSolve::lp(direction = "min", objective.in = cvec, const.mat = Amat, const.dir = sense, const.rhs = bvec)

  # If the model was solved successfully, save the solution
  if (result$status == 0) {
    solution <- result$solution

    # Extract the solutions for the 'w' and 'xi' variables
    n_out <- length(svf$outputs)
    n_var <- length(svf$grid$data_grid$phi[[1]][[1]][[1]])
    n_obs <- nrow(svf$data)

    n_w_vars <- n_out * n_var
    n_xi_vars <- n_out * n_obs

    w_solution <- solution[1:n_w_vars]
    xi_solution <- solution[(n_w_vars + 1):(n_w_vars + n_xi_vars)]

    mat_w <- vector("list", n_out)
    for (out in seq_len(n_out)) {
      start_index <- (out - 1) * n_var + 1
      end_index <- out * n_var
      mat_w[[out]] <- round(w_solution[start_index:end_index], 6)
    }

    mat_xi <- vector("list", n_out)
    for (out in seq_len(n_out)) {
      start_index <- (out - 1) * n_obs + 1
      end_index <- out * n_obs
      mat_xi[[out]] <- round(xi_solution[start_index:end_index], 6)
    }

    svf$solution <- list(w = mat_w, xi = mat_xi)
  } else {
    stop("The optimization was not successful. Status: ", result$status)
  }

  return(svf)
}
