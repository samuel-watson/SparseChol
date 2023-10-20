#include "sparse.h"

// This is a quick test of the sparse class
// Will add more tests and automatic running
// [[Rcpp::export]]
void testSparse(){
  sparse A(4,3,true);
  A.insert(0,0,1);
  A.insert(0,2,2);
  A.insert(1,1,1);
  A.insert(2,1,3);
  A.insert(3,0,2);
  A.insert(3,2,3);
  Rcpp::Rcout << "\nMatrix A: \nAp:";
  for(auto i: A.Ap)Rcpp::Rcout << " " << i;
  Rcpp::Rcout << "\nAi:";
  for(auto i: A.Ai)Rcpp::Rcout << " " << i;
  Rcpp::Rcout << "\nAx:";
  for(auto i: A.Ax)Rcpp::Rcout << " " << i;
  Rcpp::Rcout << "\nTest access to elements A(2,1) is " << A(2,1) << " it should be 3";
  MatrixXd B(3,3);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      B(i,j) = i + 1 + j*3;
    }
  }
  MatrixXd AB = A * B;
  Rcpp::Rcout << "\n" << AB;
  Rcpp::Rcout << "\n This should equal:\n 7 16 25\n 2  5  8\n 6 15 24\n11 26 41";
  Rcpp::Rcout << "\n And the transpose multiplication: \n";
  A.transpose();
  MatrixXd BA = B.transpose() * A;
  Rcpp::Rcout << BA;
}