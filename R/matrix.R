#' Generate matrix L from `sparse_chol` output
#' 
#' Generates the L matrix of the LDL decomposition from the output of 
#' the `sparse_chol` function
#' 
#' @param mat List returned by `sparse_chol`
#' @return A matrix of class `dsCMatrix`
#' @export
sparse_L <- function(mat){
  return(Matrix::sparseMatrix(i = mat$Ai, p=mat$Ap, x=mat$Ax, triangular = TRUE))
}

#' Generate matrix D from `sparse_chol` output
#' 
#' Generates the D matrix of the LDL decomposition from the output of 
#' the `sparse_chol` function
#' 
#' @param mat List returned by `sparse_chol`
#' @return A matrix of class `ddiMatrix`
#' @export
sparse_D <- function(mat){
  return(Matrix::Diagonal(x = mat$D))
}
