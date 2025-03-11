# tests/testthat/test-svfgrid.R

library(testthat)

create_test_data <- function() {
  data.frame(
    x1 = c(1, 2, 3, 4),
    x2 = c(2, 4, 3, 2),
    y1 = c(1, 1, 2, 3)
  )
}

test_that("SVFGrid creates an object correctly", {
  data <- create_test_data()
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  d <- 2

  grid_obj <- SVFGrid(data, inputs, outputs, d)

  expect_s3_class(grid_obj, "SVFGrid")
  expect_equal(grid_obj$data, data)
  expect_equal(grid_obj$inputs, inputs)
  expect_equal(grid_obj$outputs, outputs)
  expect_equal(grid_obj$d, d)
})

test_that("create_grid_svfgrid creates a grid correctly", {
  data <- create_test_data()
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  d <- 2
  grid_obj <- SVFGrid(data, inputs, outputs, d)

  grid_obj <- create_grid_svfgrid(grid_obj)

  expect_true(!is.null(grid_obj$knot_list))
  expect_true(!is.null(grid_obj$grid_properties$id_cells))
  expect_true(!is.null(grid_obj$grid_properties$values))
})

test_that("calculate_grid_properties_svfgrid calculates grid properties correctly", {
  data <- create_test_data()
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  d <- 2
  grid_obj <- SVFGrid(data, inputs, outputs, d)

  grid_obj <- create_grid_svfgrid(grid_obj)
  grid_obj <- calculate_grid_properties_svfgrid(grid_obj)

  expect_true(!is.null(grid_obj$grid_properties$phi))
  expect_true(!is.null(grid_obj$grid_properties$c_cells))
})

test_that("calculate_dmu_phi_svfgrid calculates phi correctly for a DMU", {
  data <- create_test_data()
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  d <- 2
  grid_obj <- SVFGrid(data, inputs, outputs, d)

  grid_obj <- create_grid_svfgrid(grid_obj)
  cell <- c(1, 2)

  phi_result <- calculate_dmu_phi_svfgrid(grid_obj, cell)

  expect_type(phi_result, "list")
  expect_equal(length(phi_result), 1)
})

test_that("calculate_data_grid_svfgrid calculates phi and c_cells correctly", {
  data <- create_test_data()
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  d <- 2
  grid_obj <- SVFGrid(data, inputs, outputs, d)

  grid_obj <- create_grid_svfgrid(grid_obj)
  grid_obj <- calculate_data_grid_svfgrid(grid_obj)

  expect_true(all(sapply(grid_obj$data_grid$phi, is.list)))
  expect_true(!is.null(grid_obj$data_grid$c_cells))
})

test_that("search_contiguous_cell finds contiguous cells correctly", {
  cell <- c(2, 2)
  contiguous_cells <- search_contiguous_cell(cell)

  expect_true(length(contiguous_cells) > 0)
  expect_true(all(sapply(contiguous_cells, is.numeric)))
})
