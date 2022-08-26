#include "../inst/include/sparsechol.h"
using namespace Rcpp;

//' Sparse Cholesky decomposition
//' 
//' Sparse Cholesky decomposition
//' 
//' @details
//' Generates the LDL decomposition of a symmetrix, sparse matrix using the method 
//' described by Timothy Davis (see references). Required input is a matrix
//' in sparse format from the matrix package, see \link[Matrix]{sparseMatrix}
//' @param n Integer specifying the dimension of the matrix
//' @param Ai Integer vector specifying the row positions of the non-zero values of the matrix
//' @param Ap numeric (integer valued) vector of pointers, one for each column (or row), to the initial (zero-based) index of elements in the column (or row).
//' @param Ax values of the non-zero matrix entries
//' @return A list with elements n, Ai, Ap, Ax (corresponding to above arguments) for matrix L, and element D, which 
//' contains the diagonal values of matrix D.
// [[Rcpp::export]]
Rcpp::List sparse_chol(int n,
                       std::vector<int> Ap,
                       std::vector<int> Ai,
                       std::vector<double> Ax){
  sparse mat;
  mat.n = n;
  mat.Ap = Ap;
  mat.Ai = Ai;
  std::for_each(mat.Ai.begin(), mat.Ai.end(), [](int &n){ n--; });
  mat.Ax = Ax;
  
  SparseChol chol(&mat);
  int d = chol.ldl_numeric();
  Rcpp::Rcout << "d: " << d;
  std::for_each(chol.L->Ai.begin(), chol.L->Ai.end(), [](int &n){ n++; });
  return Rcpp::List::create(_["n"] = chol.L->n,_["Ap"] = chol.L->Ap,
                            _["Ai"] = chol.L->Ai,_["Ax"] = chol.L->Ax,
                            _["D"] = chol.D);
}
