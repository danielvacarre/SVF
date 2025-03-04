# Create the example dataset
data <- data.frame(x1 = c(1, 2, 3, 4),
                   x2 = c(2, 4, 3, 2),
                   x3 = c(1, 1, 2, 3),
                   y1 = c(1, 1, 2, 3))

# Define lists of inputs, outputs, and the number of partitions
inputs <- c("x1", "x2")
outputs <- c("y1")
d <- 2

grid_instance <- GRID(data, inputs, outputs, d)

grid_instance$knot_list <- list(list(1, 2.5, 4), list(2, 3, 4), list(1,2,3))
dmu <- c(3, 5,1)
position <- search_dmu.GRID(grid_instance, dmu)

print(paste("Position in the grid: (", paste(position, collapse = ", "), ")", sep = ""))
