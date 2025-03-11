# tests/testthat/test-ssvf.R

library(testthat)

# Helper function to create test data
create_test_data <- function() {
  data.frame(
    x1 = c(1, 2, 3, 10),
    x2 = c(2, 4, 3, 10),
    y1 = c(1, 1, 2, 3)
  )
}

test_that("SSVF creates an object correctly", {
  data <- create_test_data()
  method <- "ssvf"
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  c_value <- 2.0
  eps_value <- 0
  d_value <- 4

  ssvf_obj <- SSVF(method, inputs, outputs, data, c_value, eps_value, d_value)

  expect_s3_class(ssvf_obj, "SSVF")
  expect_equal(ssvf_obj$method, method)
  expect_equal(ssvf_obj$inputs, inputs)
  expect_equal(ssvf_obj$outputs, outputs)
  expect_equal(ssvf_obj$data, data)
  expect_equal(ssvf_obj$c, c_value)
  expect_equal(ssvf_obj$eps, eps_value)
  expect_equal(ssvf_obj$d, d_value)
})

test_that("train_ssvf prepares the model correctly", {
  data <- create_test_data()
  method <- "ssvf"
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  c_value <- 2.0
  eps_value <- 0
  d_value <- 4

  ssvf_obj <- SSVF(method, inputs, outputs, data, c_value, eps_value, d_value)
  ssvf_obj <- train_ssvf(ssvf_obj)

  expect_true(!is.null(ssvf_obj$model))
  expect_true(!is.null(ssvf_obj$grid))
  n_w = length(ssvf_obj$grid$data_grid$phi[[1]][[1]][[1]]) * length(ssvf_obj$outputs)
  n_xi = nrow(ssvf_obj$data) * length(ssvf_obj$outputs)
  expect_equal(length(ssvf_obj$model$cvec), n_w + n_xi)
})

test_that("solve_ssvf solves the model correctly", {
  data <- create_test_data()
  method <- "ssvf"
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  c_value <- 2.0
  eps_value <- 0
  d_value <- 4

  ssvf_obj <- SSVF(method, inputs, outputs, data, c_value, eps_value, d_value)
  ssvf_obj <- train_ssvf(ssvf_obj)
  ssvf_obj <- solve_ssvf(ssvf_obj)

  expect_true(!is.null(ssvf_obj$solution))
  expect_true(all(sapply(ssvf_obj$solution$w, is.numeric)))
  expect_true(all(sapply(ssvf_obj$solution$xi, is.numeric)))
})

test_that("get_estimation_svf calculates output estimations correctly", {
  data <- create_test_data()
  method <- "ssvf"
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  c_value <- 2.0
  eps_value <- 0
  d_value <- 4

  ssvf_obj <- SSVF(method, inputs, outputs, data, c_value, eps_value, d_value)
  ssvf_obj <- train_ssvf(ssvf_obj)
  ssvf_obj <- solve_ssvf(ssvf_obj)

  dmu <- c(2, 3)  # Example DMU (input values)
  estimation <- get_estimation_svf(ssvf_obj, dmu)

  expect_true(is.numeric(estimation))
  expect_equal(length(estimation), length(ssvf_obj$outputs))
})

test_that("print_ssvf_model prints the model correctly", {
  data <- create_test_data()
  method <- "ssvf"
  inputs <- c("x1", "x2")
  outputs <- c("y1")
  c_value <- 2.0
  eps_value <- 0
  d_value <- 4

  ssvf_obj <- SSVF(method, inputs, outputs, data, c_value, eps_value, d_value)
  ssvf_obj <- train_ssvf(ssvf_obj)

  # Capture the printed output
  captured_output <- capture.output(print_ssvf_model(ssvf_obj))

  expect_true(length(captured_output) > 0)
  expect_true(any(grepl("Minimize", captured_output)))
  expect_true(any(grepl("Subject To", captured_output)))
})

