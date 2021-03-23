#' Title transformation
#'
#' Esta función evalúa si el valor de una celda es mayor o menor al de un nodo del grid.
#' Si es mayor devuelve 1, si es igual devuelve 0 y si es menor devuelve -1.
#'
#'
#' @param x_j: valor de la celda a evaluar
#' @param t_l_j: valor del nodo con el que se quiere comparar
#'
#' @return res: resultado de la transformación
#' @export
#'
#' @examples
#'
transformation <- function(x_j, t_l_j){
  z <- as.numeric(x_j) - t_l_j
  if (z < 0) {res = -1}
  else if (z == 0) {res = 0}
  else if (z > 0) {res = 1}
  res
}


#' Title create_matrix_partitions
#'
#' Función que crea las particiones de cada dimensión. Coge cada dimensión de inputs y la trocea en d particiones equidistantes
#'
#'
#' @param x:
#' @param d:
#'
#' @return
#' @export
#'
#' @examples
#'
create_matrix_partitions <- function(x,d){
  # Número de columnas de x
  n_dim = ncol(x)

  # Defino t: puntos del grid
  t = c()

  # Defino t_ind: índices (posiciones) de los puntos del grid
  t_ind = c()

  for(col in 1:n_dim){
    ts = c()
    t_max = max(x[,col])
    t_min = min(x[,col])
    amplitud = (t_max-t_min)/d
    for(i in 0:(d)){
      t_i = t_min + i*amplitud
      ts = append(ts,t_i)
    }
    t = append(t,ts)
    t_ind = append(t_ind,seq(0,length(ts)-1))
  }

  t = matrix(t, ncol=d+1, byrow=TRUE)
  r = t(t) # La traspuesta de t
  t_ind = matrix(t_ind, ncol=d+1, byrow=TRUE)

  # Producto cartesiano de los índices, para obtener los puntos del grid
  lista_comb <- list()
  for(i in 1:n_dim){
    lista_comb[[i]] <- t_ind[i,]
  }
  vector_subind <- expand.grid(lista_comb)
  vector_subind <- vector_subind[order(vector_subind$Var1), ]

  # Producto cartesiano de los puntos, para obtener el grid
  lista_comb_p <- list()
  for(i in 1:n_dim){
    lista_comb_p[[i]] <- t[i,]
  }
  grid_points <- expand.grid(lista_comb_p)
  grid_points <- grid_points[order(grid_points$Var1), ]

  # Le doy formato en caso de que tengamos un unico input
  if (ncol(x)==1){
    vector_subind <- matrix(vector_subind, ncol=ncol(x), byrow=T)
    grid_points <- matrix(grid_points, ncol=ncol(x), byrow=T)
  }

  salidas <- list(t=t,r=r,t_ind=t_ind,vector_subind=vector_subind,
                  grid_points=grid_points)
  return(salidas)
}

#Aqui debería estar:
  #calculate_cv_mse
  #calculate_pos_phi
  #calculate_value_phi
  #cross_validation
  #estimacion
  #modify_model
  #svf
  #svf_lp
  #transformation
  #dataframe_d



