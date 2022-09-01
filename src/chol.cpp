#include "../inst/include/SparseChol.h"
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
//' @examples
//' n <- 10
//' Ap  <- c(0, 1, 2, 3, 4, 6, 7, 9, 11, 15, 19)
//' Ai <- c(1, 2, 3, 4, 2,5, 6, 5,7, 5,8, 1,5,8,9, 2,5,7,10)
//' Ax = c(1.7, 1., 1.5, 1.1, .02,2.6, 1.2, .16,1.3, .09,1.6,
//'           .13,.52,.11,1.4, .01,.53,.56,3.1)
//' out <-sparse_chol(n,Ap,Ai,Ax)
//' sparse_L(out)
//' sparse_D(out)
// [[Rcpp::export]]
Rcpp::List sparse_chol(int n,
                       std::vector<int> Ap,
                       std::vector<int> Ai,
                       std::vector<double> Ax){
  sparse mat(Ap);
  mat.n = n;
  mat.Ai = Ai;
  if(Ai[0] != 0)std::for_each(mat.Ai.begin(), mat.Ai.end(), [](int &n){ n--; });
  if(Ap[0] != 0)std::for_each(mat.Ap.begin(), mat.Ap.end(), [](int &n){ n--; });
  mat.Ax = Ax;
  
  SparseChol chol(&mat);
  int d = chol.ldl_numeric();
  Rcpp::Rcout << "d: " << d;
  // if(Ai[0] != 0)std::for_each(chol.L->Ai.begin(), chol.L->Ai.end(), [](int &n){ n++; });
  // if(Ap[0] != 0)std::for_each(chol.L->Ap.begin(), chol.L->Ap.end(), [](int &n){ n++; });
  return Rcpp::List::create(_["n"] = chol.L->n,_["Ap"] = chol.L->Ap,
                            _["Ai"] = chol.L->Ai,_["Ax"] = chol.L->Ax,
                            _["D"] = chol.D);
}
