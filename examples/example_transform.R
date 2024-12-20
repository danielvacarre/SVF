# Evaluate if an observation is less than the grid node value
result <- transformation(2, 5)
cat("Transformation result:", result, "\n")
# Output: -1

# Evaluate if an observation is equal to the grid node value
result <- transformation(3, 3)
cat("Transformation result:", result, "\n")
# Output: 0

# Evaluate if an observation is greater than the grid node value
result <- transformation(4, 2)
cat("Transformation result:", result, "\n")
# Output: 1
