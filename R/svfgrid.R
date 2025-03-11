#' Constructor for the SVFGrid class
#'
#' This function creates an instance of the SVFGrid class, which extends the
#' GRID class with additional functionalities for handling grids in the
#' context of SVF analysis.
#'
#' @param data Dataset used to construct the grid.
#' @param inputs List of input variables.
#' @param outputs List of output variables.
#' @param d Number of partitions to divide the grid into.
#'
#' @return An object of class SVFGrid.
#'
#' @examples
#'data <- data.frame(x1 = c(1, 2, 3, 4), x2 = c(5, 6, 7, 8), y1 = c(9, 1, 2, 3))
#'
#'print_grid <- function(x) {
#'  cat("GRID Data:\n")
#'  cat("----------------------------------\n")
#'  cat(sprintf("Inputs: %s\n", paste(x$inputs, collapse = ", ")))
#'  cat(sprintf("Outputs: %s\n", paste(x$outputs, collapse = ", ")))
#'  cat(sprintf("Data Dimensions: %d rows, %d columns\n", nrow(x$data), ncol(x$data)))
#'  cat(sprintf("  d (Number of partitions): %d\n", x$d))
#'
#'  cat("Data Preview (first few rows only):\n")
#'  if (nrow(x$data) > 0) {
#'    print(head(x$data))
#'  } else {
#'    cat("No data available.\n")
#'  }
#'
#'  cat("----------------------------------\n")
#'  invisible(x)
#'}
#'
#'inputs <- c("x1", "x2")
#'outputs <- c("y1")
#'d <- 2
#'
#'grid_obj <- SVFGrid(data, inputs, outputs, d)
#'print_grid(grid_obj)

#' @export
SVFGrid <- function(data, inputs, outputs, d) {
  if (!is.data.frame(data)) stop("data must be a data frame")
  if (!is.character(inputs)) stop("inputs must be a character vector")
  if (!is.character(outputs)) stop("outputs must be a character vector")
  if (!all(inputs %in% colnames(data))) stop("Not all inputs are in the dataset")
  if (!all(outputs %in% colnames(data))) stop("Not all outputs are in the dataset")
  if (!is.numeric(d) || d <= 0) stop("d must be a positive number")
  grid <- list(
    data = data,
    inputs = inputs,
    outputs = outputs,
    d = d,
    grid_properties = data.frame(),
    data_grid = data.frame()
  )
  class(grid) <- c("SVFGrid", "GRID")
  return(grid)
}

#' Create a grid based on data and hyperparameter d
#'
#' This function creates a grid using the provided data and hyperparameter `d`.
#' The grid represents the data in a space divided into cells defined by `d`.
#'
#' @param grid An SVFGrid object.
#'
#' @return The SVFGrid object with the created grid.
#'
#' @examples
#'
#' data <- data.frame(x1 = c(1, 2, 3, 4), x2 = c(2, 4, 3, 2), y1 = c(1, 1, 2, 3))
#'
#' inputs <- c("x1", "x2")
#' outputs <- c("y1")
#' d <- 2
#' grid_obj <- SVFGrid(data, inputs, outputs, d)
#' grid_obj <- create_grid_svfgrid(grid_obj)
#'
#' @export
create_grid_svfgrid <- function(grid) {
  x <- grid$data[, grid$inputs, drop = FALSE]
  n_dim <- ncol(x)
  knot_list <- list()
  knot_index <- list()

  for (col in seq_len(n_dim)) {
    knot_min <- min(x[, col], na.rm = TRUE)
    knot_max <- max(x[, col], na.rm = TRUE)
    knots <- seq(knot_min, knot_max, length.out = grid$d + 1)
    knot_list[[col]] <- knots
    knot_index[[col]] <- 1:(grid$d + 1)
  }

  grid$knot_list <- knot_list

  if (n_dim == 1) {
    id_cells <- as.data.frame(knot_index[[1]])
    colnames(id_cells) <- grid$inputs
    values <- as.data.frame(knot_list[[1]])
    colnames(values) <- grid$inputs
  } else {
    id_cells <- expand.grid(knot_index)
    id_cells <- id_cells[, ncol(id_cells):1]
    values <- expand.grid(rev(knot_list))
    values <- values[, ncol(values):1]
  }

  grid$grid_properties <- list(id_cells = id_cells, values = values, phi = vector("list", nrow(values)))
  grid <- calculate_grid_properties_svfgrid(grid)
  grid <- calculate_data_grid_svfgrid(grid)

  return(grid)
}


#' Calculate the transformation (phi) value of an observation in the grid
#'
#' This function computes and returns the phi value for a specific grid cell,
#' based on the grid data and the cell position.
#'
#' @param grid An SVFGrid object.
#' @param cell Position of the observation in the grid.
#'
#' @return A list containing the phi values for the specified cell.
#'
#' @examples
#'
#' data <- data.frame(x = c(1, 3, 5), y = c(2, 4, 6))
#' inputs <- c("x")
#' outputs <- c("y")
#' d <- 2
#' grid <- SVFGrid(data, inputs, outputs, d)
#' grid <- create_grid_svfgrid(grid)
#' cell <- c(1)
#' phi_result <- calculate_dmu_phi_svfgrid(grid, cell)
#' print(phi_result)
#'
#' @export
calculate_dmu_phi_svfgrid <- function(grid, cell) {
  id_cells <- grid$grid_properties$id_cells
  phi <- apply(id_cells, 1, function(row) as.numeric(all(cell >= row)))
  n_outputs <- length(grid$outputs)
  phi <- replicate(n_outputs, phi, simplify = FALSE)
  return(list(phi))
}

#' Add transformation values (phi) to the grid dataframe
#'
#' This function calculates and adds additional information to the grid's
#' dataframe, including the phi values for each cell and the contiguous cells.
#'
#' @param grid An SVFGrid object.
#'
#' @return The updated SVFGrid object with the grid dataframe.
#'
#' @examples
#'
#' data <- data.frame(x = c(1, 3, 5), y = c(2, 4, 6))
#' inputs <- c("x")
#' outputs <- c("y")
#' d <- 2
#' grid <- SVFGrid(data, inputs, outputs, d)
#' grid <- create_grid_svfgrid(grid)
#' grid <- calculate_grid_properties_svfgrid(grid)
#'
#' @export
calculate_grid_properties_svfgrid <- function(grid) {
  n <- nrow(grid$grid_properties$id_cells)

  grid$grid_properties$phi <- lapply(seq_len(n), function(i) {
    cell <- as.numeric(grid$grid_properties$values[i, ])
    p <- search_dmu_grid(grid, cell)
    calculate_dmu_phi_svfgrid(grid, p)[[1]]
  })

  grid$grid_properties$c_cells <- lapply(seq_len(n), function(i) {
    cell <- as.numeric(grid$grid_properties$id_cells[i, ])
    search_contiguous_cell(cell)
  })

  return(grid)
}

#' Add transformation values to each observation in data_grid
#'
#' This function processes each observation in `data_grid` using the specified
#' inputs, calculating phi values and contiguous cells, and updates the grid object.
#'
#' @param grid An SVFGrid object.
#'
#' @return The modified grid object with phi and c_cells results added.
#'
#' @examples
#'
#' data <- data.frame(x = c(1, 3, 5, 7), y = c(2, 4, 6, 8))
#' inputs <- c("x")
#' outputs <- c("y")
#' d <- 2
#' grid <- SVFGrid(data, inputs, outputs, d)
#' grid <- create_grid_svfgrid(grid)
#' grid <- calculate_data_grid_svfgrid(grid)
#'
#' @export
calculate_data_grid_svfgrid<- function(grid) {
  grid$data_grid <- grid$data[, c(grid$inputs, grid$outputs), drop = FALSE]

  grid$data_grid$phi <- lapply(seq_len(nrow(grid$data_grid)), function(i) {
    x <- as.numeric(grid$data_grid[i, grid$inputs])
    p <- search_dmu_grid(grid, x)
    calculate_dmu_phi_svfgrid(grid, p)
  })

  grid$data_grid$c_cells <- lapply(seq_len(nrow(grid$data_grid)), function(i) {
    x <- as.numeric(grid$data_grid[i, grid$inputs])
    p <- search_dmu_grid(grid, x)
    search_contiguous_cell(p)
  })

  return(grid)
}

#' Find contiguous cells in SVFGrid
#'
#' This function identifies and returns the cells contiguous to a specified cell
#' in the grid. Contiguous cells share at least one edge or vertex with the cell.
#'
#' @param cell Vector specifying the cell position in the grid.
#'
#' @return A list of cells contiguous to the specified cell.
#'
#' @examples
#'
#'cell <- c(2, 2)
#'contiguous_cells <- search_contiguous_cell(cell)
#'
#' @export
search_contiguous_cell <- function(cell) {
  Filter(Negate(is.null), lapply(seq_along(cell), function(dim) {
    if (cell[dim] > 1) {
      new_cell <- cell
      new_cell[dim] <- new_cell[dim] - 1
      return(new_cell)
    }
    NULL
  }))
}
