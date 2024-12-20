data <- data.frame(x = rnorm(10), y = rnorm(10))
grid <- GRID(data, inputs = c("x"), outputs = c("y"), d = 4)
print(grid)
