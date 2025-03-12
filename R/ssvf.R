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
#' svf_model <- train(svf_model)
#'
#' print_model(svf_model)
#'
#' svf_model <- solve(svf_model)
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
train <- function(svf) {
  y_df <- svf$data[, svf$outputs, drop = FALSE]
  y <- as.matrix(y_df)

  n_out <- ncol(y_df)
  n_obs <- nrow(y_df)

  # Create the model grid
  svf$grid <- create_grid(SVFGrid(svf$data, svf$inputs, svf$outputs, svf$d))
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
solve <- function(svf) {
  # Check if the model has been trained
  if (is.null(svf$model)) {
    stop("The model has not been trained. Please run `train` first.")
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

#' Plot Support Vector Frontier (SVF) Estimation
#'
#' This function generates a plot of the estimated Support Vector Frontier (SVF) for one or two input variables.
#' The plot is either 1D or 2D depending on whether one or two input variables are provided.
#' The estimation is calculated using the \code{get_estimation_svf} function for each point in the input variable range(s).
#'
#' @param svf A list object containing the solution to a Support Vector Frontier model. It must contain a resolved model, which can be obtained by calling the \code{solve} function. This object should have a \code{solution} field, input data (\code{data}), and output data (\code{outputs}).
#' @param input1 A string representing the name of the first input variable in the \code{svf$data} list. This variable will be plotted on the x-axis of the plot.
#' @param input2 A string representing the name of the second input variable in the \code{svf$data} list. This is only required if you want to generate a 2D plot. If not provided, the function will generate a 1D plot. Default is \code{NULL}.
#' @param grid_size A numeric value that defines the resolution of the grid for the estimation. It determines how many points are used to generate the estimation. The default value is 10.
#'
#' @details
#' If only one input variable (\code{input1}) is specified, the function will create a 1D plot of the SVF estimation across the range of the \code{input1} variable.
#' If both \code{input1} and \code{input2} are specified, the function will generate a 2D surface plot of the SVF estimation over a grid defined by the \code{input1} and \code{input2} ranges.
#' The \code{get_estimation_svf} function is called to estimate the SVF for each point in the specified grid.
#'
#' @return A Plotly object representing the generated plot. The plot includes:
#' - A line representing the estimated SVF (in 1D) or a surface representing the estimated SVF (in 2D).
#' - Markers representing the Data Management Units (DMUs) from the \code{svf$data}.
#'
#' @examples
#' data <- data.frame(
#' x1 = c(1, 2, 3, 10),
#' x2 = c(2, 4, 3, 10),
#' y1 = c(1, 1, 2, 3))
#' method <- "ssvf"
#' inputs <- c("x1", "x2")
#' outputs <- c("y1")
#' c_value <- 1
#' eps_value <- 0.1
#' d_value <- 2
#'
#' svf_model <- SSVF(method, inputs, outputs, data, c_value, eps_value, d_value)
#'
#' svf_model <- train(svf_model)
#'
#'
#' svf_model <- solve(svf_model)
#'
#' fig <- plot_svf_estimation(svf_model, "x1", "x2",  grid_size = 100)
#'
#' fig
#'
#' @import plotly
#' @export
plot_svf_estimation <- function(svf, input1, input2 = NULL, grid_size = 10) {

  # Check if the model has been solved (i.e., if `svf$solution` exists)
  if (is.null(svf$solution)) stop("The model has not been solved. Run `solve` first.")

  # Define the range of values for the first input variable
  input1_range <- seq(0, max(svf$data[[input1]]), length.out = grid_size)

  # Case when only one input variable is provided (1D plot)
  if (is.null(input2)) {

    # Estimate the SVF for each value of input1
    estimations <- sapply(input1_range, function(x) get_estimation(svf, c(x)))

    # Create the 1D plot using Plotly
    fig <- plot_ly()
    fig <- add_lines(fig, x = input1_range, y = estimations, name = "SVF Estimation",
                     line = list(color = "blue"))
    fig <- add_markers(fig, x = svf$data[[input1]], y = svf$data[[svf$outputs[1]]],
                       name = "DMUs", marker = list(color = "red", size = 6))
    fig <- layout(fig, title = "SVF Estimation (1D)",
                  xaxis = list(title = input1),
                  yaxis = list(title = "Estimation"))

    # Case when two input variables are provided (2D plot)
  } else {

    # Define the range of values for the second input variable
    input2_range <- seq(0, max(svf$data[[input2]]), length.out = grid_size)
    # Create a grid of all combinations of input1 and input2 values
    grid <- expand.grid(input1_range, input2_range)

    # Estimate the SVF for each combination of input1 and input2
    estimations <- apply(grid, 1, function(x) get_estimation(svf, c(x[1], x[2])))

    # Reshape the estimations into a matrix for the surface plot
    z_matrix <- matrix(estimations, nrow = grid_size, byrow = TRUE)

    # Create the 2D surface plot using Plotly
    fig <- plot_ly()
    fig <- add_surface(fig, x = input1_range, y = input2_range, z = z_matrix,
                       colorscale = "Blues", opacity = 0.7)
    fig <- add_markers(fig, x = svf$data[[input1]], y = svf$data[[input2]], z = svf$data[[svf$outputs[1]]],
                       name = "DMUs", marker = list(color = "red", size = 4, symbol = "circle"))
    fig <- layout(fig, title = "SVF Estimation (2D)",
                  scene = list(
                    xaxis = list(title = input1),
                    yaxis = list(title = input2),
                    zaxis = list(title = "Estimation")
                  ))
  }

  # Return the generated Plotly figure
  return(fig)
}
