# Definition of the SVFPrimalSolution class
#' Creates an SVFPrimalSolution object
#'
#' @param w Values of the weights w from the solved model.
#' @param xi Values of xi from the solved model.
#'
#' @return An object of class SVFPrimalSolution.
#'
#' @examples
#'
#' w_values <- c(0.5, 1.5, 2.5)
#' xi_values <- c(0.1, 0.2, 0.3)
#' solution <- SVFPrimalSolution(w_values, xi_values)
#' class(solution)
#' solution$w
#' solution$xi
#'
#' @export
SVFPrimalSolution <- function(w, xi) {
  solution <- list(w = w, xi = xi)
  class(solution) <- "SVFPrimalSolution"
  return(solution)
}
