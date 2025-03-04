#' GRID Class for SVF model
#'
#' This class represents a grid over which the SVF model is applied.
#' A grid is a partition of the input space divided into cells.
#'
#' @details To create a GRID object, provide a DataFrame with the data,
#' the names of the input (inputs) and output (outputs) variables, and
#' the number of partitions (d).
#'
#' @examples
#' data <- data.frame(x = sample(1:10, 4, replace = TRUE),
#'                    y = sample(1:10, 4, replace = TRUE))
#' grid <- GRID(data, inputs = c("x"), outputs = c("y"), d = 2)
#' print(grid)
#'
#' @name GRID
#'
#' @param data DataFrame with the dataset on which the grid is built.
#' @param inputs List of inputs.
#' @param outputs List of outputs.
#' @param d Number of partitions in which the grid is divided.
#' @field data_grid Data grid, initially NULL.
#' @field knot_list List of grid nodes, initially NULL.
#'
#'
#' @export
GRID <- function(data, inputs, outputs, d) {
  if (!is.data.frame(data)) stop("data must be a data frame")
  if (!all(inputs %in% colnames(data))) stop("Some inputs are not in the dataset")
  if (!all(outputs %in% colnames(data))) stop("Some outputs are not in the dataset")
  if (!is.numeric(d) || d <= 0) stop("d must be a positive number")

  structure(
    list(data = data, inputs = inputs, outputs = outputs, d = d, data_grid = NULL, knot_list = NULL),
    class = "GRID"
  )
}

#' Function to return the cell in which an observation is located in the grid
#'
#' This method searches for a specific DMU in the grid and returns the cell
#' in which that observation is located.
#'
#' @param grid GRID object.
#' @param dmu Observation to search for in the grid.
#'
#' @return Vector with the position of the observation in the grid.
#'
#' @examples
#'data <- data.frame(x1 = c(1, 2, 3, 4),
#'                   x2 = c(2, 4, 3, 2),
#'                   x3 = c(1, 1, 2, 3),
#'                   y1 = c(1, 1, 2, 3))
#'
#'inputs <- c("x1", "x2")
#'outputs <- c("y1")
#'d <- 2
#'
#'grid_instance <- GRID(data, inputs, outputs, d)
#'
#'grid_instance$knot_list <- list(list(1, 2.5, 4), list(2, 3, 4), list(1,2,3))
#'dmu <- c(3, 5, 1)
#'position <- search_dmu.GRID(grid_instance, dmu)
#'
#'print(paste("Position in the grid: (", paste(position, collapse = ", "), ")", sep = ""))
#'
#' @export
search_dmu.GRID <- function(grid, dmu) {
  if (is.null(grid$knot_list)) stop("knot_list is NULL. Ensure the grid has been initialized properly.")

  r <- lapply(grid$knot_list, unlist)
  cell <- numeric(length(dmu))

  for (l in seq_along(dmu)) {
    found <- FALSE
    for (m in seq_along(r[[l]])) {
      trans <- transformation(dmu[l], r[[l]][m])
      if (trans < 0) {
        cell[l] <- m - 1
        found <- TRUE
        break
      } else if (trans == 0) {
        cell[l] <- m
        found <- TRUE
        break
      }
    }
    if (!found) {
      cell[l] <- length(r[[l]])
    }
  }
  return(cell)
}
#' Transformation of values in the GRID
#'
#' This function evaluates whether the value of an observation is greater than, equal to,
#' or less than the value of a node in the grid. It returns 1 if greater,
#' 0 if equal, and -1 if less.
#'
#' @param x_i Value of the cell to evaluate.
#' @param t_k Value of the node to compare with.
#'
#' @return Result of the comparison: 1, 0, -1.
#'
#' @examples
#'result <- transformation(2, 5)
#'cat("Transformation result:", result, "\n")
#'
#'result <- transformation(3, 3)
#'cat("Transformation result:", result, "\n")
#'
#'result <- transformation(4, 2)
#'cat("Transformation result:", result, "\n")
#'
#' @export
transformation <- function(x_i, t_k) {
  z <- x_i - t_k
  if (z < 0) {
    return(-1)
  } else if (z == 0) {
    return(0)
  } else {
    return(1)
  }
}
