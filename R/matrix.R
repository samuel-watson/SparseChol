#' Generate matrix L from `sparse_chol` output
#' 
#' Generates the L matrix of the LDL decomposition from the output of 
#' the `sparse_chol` function
#' 
#' @param mat List returned by `sparse_chol`
#' @return A matrix of class `dsCMatrix`
#' @importFrom Matrix sparseMatrix 
#' @export
sparse_L <- function(mat){
  M <- Matrix::sparseMatrix(i = mat$Ai+1, p=mat$Ap, x=mat$Ax, triangular = TRUE)
  diag(M) <- 1
  return(M)
}

#' Generate matrix D from `sparse_chol` output
#' 
#' Generates the D matrix of the LDL decomposition from the output of 
#' the `sparse_chol` function
#' 
#' @param mat List returned by `sparse_chol`
#' @return A matrix of class `ddiMatrix`
#' @importFrom Matrix Diagonal
#' @export
sparse_D <- function(mat){
  return(Matrix::Diagonal(x = mat$D))
}

#' Generate Cholesky decomposition from Matrix class `dsCMatrix`
#' 
#' Generates the Cholesky decomposition L as A == LL^T from a 
#' sparse matrix
#' 
#' @param mat A matrix of class `dsCMatrix`
#' @return A matrix of class `ddiMatrix`
#' @importFrom Matrix sparseMatrix Diagonal
#' @export
LL_Cholesky <- function(mat){
  out <- sparse_chol(length(mat@p)-1,mat@p,mat@i,mat@x)
  M <- Matrix::sparseMatrix(i = out$Ai+1, p=out$Ap, x=out$Ax, triangular = TRUE)
  diag(M) <- 1
  return(M%*%Matrix::Diagonal(x = sqrt(out$D)))
}

#' Generate LDL decomposition from Matrix class `dsCMatrix`
#' 
#' Generates the Cholesky decomposition L as A == LL^T from a sparse 
#' matrix
#' 
#' @param mat A matrix of class `dsCMatrix`
#' @return A list of matrices L and D
#' @importFrom Matrix sparseMatrix Diagonal
#' @export
LDL_Cholesky <- function(mat){
  out <- sparse_chol(length(mat@p)-1,mat@p,mat@i,mat@x)
  M <- Matrix::sparseMatrix(i = out$Ai+1, p=out$Ap, x=out$Ax, triangular = TRUE)
  diag(M) <-1
  return(list(L = M, D = Matrix::Diagonal(x=out$D)))
}


