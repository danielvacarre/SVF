#' Title transformation
#'
#' Esta función evalua si el valor de una celda es mayor o menor al de un nodo del grid.
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


