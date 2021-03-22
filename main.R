devtools::load_all()

load("./examples/example.RData")

print(transformation(1,2))
print(transformation(2,2))
print(transformation(3,2))

x<-data[c("x1","x2")]

create_matrix_partitions(x,2)


