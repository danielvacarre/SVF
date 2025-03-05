library(testthat)

test_that("SVFGrid constructor works correctly", {
  data <- data.frame(x = sample(1:10, 5, replace = TRUE),
                     y = sample(1:10, 5, replace = TRUE))

  grid <- SVFGrid(data, inputs = c("x"), outputs = c("y"), d = 2)

  expect_s3_class(grid, "SVFGrid")
  expect_s3_class(grid, "GRID")
  expect_true(is.list(grid))
  expect_equal(grid$d, 2)
})

test_that("create_grid_svfgrid generates correct knots", {
  data <- data.frame(x1 = c(1, 2, 3, 4), x2 = c(2, 4, 3, 2), y = c(1, 1, 2,3))
  grid <- SVFGrid(data, inputs = c("x1", "x2"), outputs = c("y"), d = 2)

  grid <- create_grid_svfgrid(grid)

  expect_length(grid$knot_list, 2)
  expect_equal(length(grid$knot_list[[1]]), grid$d + 1)
})

test_that("calculate_dmu_phi works as expected", {
  data <- data.frame(x = c(1, 2, 3), y = c(4, 5, 6))
  grid <- SVFGrid(data, inputs = c("x"), outputs = c("y"), d = 2)
  grid <- create_grid_svfgrid(grid)

  cell <- c(1, 2)
  phi_values <- calculate_dmu_phi_svfgrid(grid, cell)

  expect_true(is.list(phi_values))
  expect_length(phi_values[[1]], nrow(grid$df_grid$id_cells))
})

test_that("search_contiguous_cell finds neighbors", {
  cell <- c(2, 3)

  neighbors <- search_contiguous_cell(cell)

  expect_true(is.list(neighbors))
  expect_true(length(neighbors) <= length(cell))
})
