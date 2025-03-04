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
#' @example examples/example_svfgrid.R
#'
#' @export
SVFGrid <- function(data, inputs, outputs, d) {
  grid <- list(
    data = data,
    inputs = inputs,
    outputs = outputs,
    d = d,
    df_grid = data.frame(),
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
#' @example examples/example_create_grid.R
#'
#' @export
create_grid.SVFGrid <- function(grid) {
  x <- grid$data[, grid$inputs, drop = FALSE]
  n_dim <- ncol(x)

  # Generate knots and indices
  knot_list <- lapply(seq_len(n_dim), function(col) {
    seq(
      min(x[, col], na.rm = TRUE),
      max(x[, col], na.rm = TRUE),
      length.out = grid$d + 1
    )
  })
  knot_index <- lapply(seq_along(knot_list), function(idx) seq_len(grid$d + 1))

  grid$knot_list <- knot_list

  # Create cell combinations and values
  id_cells <- expand.grid(knot_index)
  values <- expand.grid(rev(knot_list))

  id_cells <- id_cells[, ncol(id_cells):1]  # Reorder columns
  values <- values[, ncol(values):1]

  grid$df_grid <- list(
    id_cells = id_cells,
    values = values,
    phi = vector("list", nrow(values))
  )

  grid <- calculate_df_grid.SVFGrid(grid)
  grid <- calculate_data_grid.SVFGrid(grid)

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
#' @example examples/example_phi.R
#'
#' @export
calculate_dmu_phi.SVFGrid <- function(grid, cell) {
  id_cells <- grid$df_grid$id_cells
  phi <- apply(id_cells, 1, function(row) all(cell >= row))
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
#' @example examples/example_df.R
#'
#' @export
calculate_df_grid.SVFGrid <- function(grid) {
  n <- nrow(grid$df_grid$id_cells)

  grid$df_grid$phi <- lapply(seq_len(n), function(i) {
    cell <- as.numeric(grid$df_grid$values[i, ])
    p <- search_dmu.GRID(grid, cell)
    calculate_dmu_phi.SVFGrid(grid, p)[[1]]
  })

  grid$df_grid$c_cells <- lapply(seq_len(n), function(i) {
    cell <- as.numeric(grid$df_grid$id_cells[i, ])
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
#' @example examples/example_data.R
#'
#' @export
calculate_data_grid.SVFGrid <- function(grid) {
  grid$data_grid <- grid$data[, c(grid$inputs, grid$outputs), drop = FALSE]

  grid$data_grid$phi <- lapply(seq_len(nrow(grid$data_grid)), function(i) {
    x <- as.numeric(grid$data_grid[i, grid$inputs])
    p <- search_dmu.GRID(grid, x)
    calculate_dmu_phi.SVFGrid(grid, p)
  })

  grid$data_grid$c_cells <- lapply(seq_len(nrow(grid$data_grid)), function(i) {
    x <- as.numeric(grid$data_grid[i, grid$inputs])
    p <- search_dmu.GRID(grid, x)
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
#' @example examples/example_contiguous.R
#'
#' @export
search_contiguous_cell <- function(cell) {
  lapply(seq_along(cell), function(dim) {
    if (cell[dim] > 1) {
      new_cell <- cell
      new_cell[dim] <- new_cell[dim] - 1
      return(new_cell)
    }
    NULL
  }) |> purrr::compact()
}
