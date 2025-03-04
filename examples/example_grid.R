data <- data.frame(x = sample(1:10, 4, replace = TRUE),
                    y = sample(1:10, 4, replace = TRUE))
grid <- GRID(data, inputs = c("x"), outputs = c("y"), d = 2)
print(grid)
