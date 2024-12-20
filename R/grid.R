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
#' data <- data.frame(x = rnorm(10), y = rnorm(10))
#' grid <- GRID(data, inputs = c("x"), outputs = c("y"), d = 4)
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
#' @example examples/example_grid.R
#'
#' @export
GRID <- function(data, inputs, outputs, d) {
  structure(list( data = data, inputs = inputs, outputs = outputs, d = d, data_grid = NULL, knot_list = NULL), class = "GRID")
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
#' @example examples/example_search.R
#'
#' @export
search_dmu.GRID <- function(grid, dmu) {
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
#' @example examples/example_transform.R
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
