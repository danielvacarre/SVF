library(testthat)
library(SVF)

test_that("GRID initializes correctly", {
  data <- data.frame(x = 1:4, y = 5:8)
  grid <- GRID(data, inputs = c("x"), outputs = c("y"), d = 2)

  expect_s3_class(grid, "GRID")  # Verifica que sea de clase GRID
  expect_equal(grid$data, data)  # Verifica que los datos coincidan
  expect_equal(grid$d, 2)        # Verifica que d se asignó correctamente
  expect_null(grid$data_grid)    # Verifica que data_grid es NULL inicialmente
  expect_null(grid$knot_list)    # Verifica que knot_list es NULL inicialmente
})

test_that("GRID fails with incorrect inputs", {
  data <- data.frame(x = 1:4, y = 5:8)

  expect_error(GRID(data, inputs = c("z"), outputs = c("y"), d = 2),
               "Some inputs are not in the dataset")  # Error por columna inexistente

  expect_error(GRID(data, inputs = c("x"), outputs = c("z"), d = 2),
               "Some outputs are not in the dataset") # Error por columna inexistente

  expect_error(GRID(data, inputs = c("x"), outputs = c("y"), d = -1),
               "d must be a positive number")  # Error por d negativo
})

test_that("search_dmu_gridfinds correct cell positions", {
  data <- data.frame(x = 1:4, y = 5:8)
  grid <- GRID(data, inputs = c("x"), outputs = c("y"), d = 2)

  # Definir los nodos de la malla
  grid$knot_list <- list(
    list(1, 2, 3, 4),
    list(5, 6, 7, 8)
  )

  dmu <- c(3, 6)
  position <- search_dmu_grid(grid, dmu)

  expect_equal(position, c(3, 2))  # Verifica que la posición en la malla sea la esperada
})

test_that("transformation function returns correct values", {
  expect_equal(transformation(2, 5), -1)  # Menor
  expect_equal(transformation(3, 3), 0)   # Igual
  expect_equal(transformation(4, 2), 1)   # Mayor
})
