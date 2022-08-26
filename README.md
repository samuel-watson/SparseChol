# SparseChol

A c++ implementation of the approach to sparse, symmetric matrix decomposition described by Timothy Davis (https://fossies.org/linux/SuiteSparse/LDL/Doc/ldl_userguide.pdf).
The header file in /inst/include/sparsechol.h defines the class SparseChol, which can be implemented in other c++ applications for R with Rcpp. The R function
`sparse_chol()` provides an R interface using compressed column form as per the `CsparseMatrix` in the Matrix package.
