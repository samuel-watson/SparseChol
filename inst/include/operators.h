#pragma once

#include "sparsematrix.h"

// some operators
inline sparse operator*(sparse A, const sparse& B){
  A *= B;
  return A;
}

inline dblvec operator*(const sparse& A, const dblvec& B){
  if(A.m != B.size())Rcpp::stop("wrong dimension in sparse-vector multiplication");
  dblvec AB(A.n,0.0);
  double val;
  int i,j;
  for(i = 0; i < A.n; i++){
    for(j = A.Ap[i]; j < A.Ap[i+1]; j++){
      AB[i] += A.Ax[j]*B[A.Ai[j]];
    }
  }
  return AB;
}

// multiplication of sparse and diagonal of a vector
inline sparse operator%(sparse A, const dblvec& x){
  A %= x;
  return A;
}

inline sparse operator+(sparse A, const sparse& B){
  A += B;
  return A;
}

// right multiplication with a diagonal matrix represented a double vector
inline sparse& sparse::operator%=(const dblvec& x){
  for(int i = 0; i < Ax.size(); i++){
    Ax[i] *= x[Ai[i]];
  }
  return *this;
}

inline sparse identity(int n){
  sparse A(n,n);
  intvec ones(n);
  std::iota(std::begin(ones),std::end(ones),0);
  A.Ap = ones;
  A.Ap.push_back(n);
  A.Ai = ones;
  A.Ax = dblvec(n,1);
  return A;
}

// had to wrap some of these in a namespace to prevent problems with
// reverse compatibility on CRAN where the functions are defined in glmmrBase v0.4.6
// I can't submit both at once as they have to both be compared to one another.
// Will remove this namespace once glmmrBase is updated.
namespace SparseOperators {

using namespace Eigen;

inline MatrixXd sparse_to_dense(const sparse& m,
                                bool symmetric = true,
                                bool rowmajor = true){
  MatrixXd D = MatrixXd::Zero(m.n,m.m);
  if(rowmajor){
    for(int i = 0; i < m.n; i++){
      for(int j = m.Ap[i]; j < m.Ap[i+1]; j++){
        D(i,m.Ai[j]) = m.Ax[j];
        if(symmetric) D(m.Ai[j],i) = m.Ax[j];
      }
    }
  } else {
    for(int i = 0; i < m.m; i++){
      for(int j = m.Ap[i]; j < m.Ap[i+1]; j++){
        D(m.Ai[j],i) = m.Ax[j];
        if(symmetric) D(m.Ai[j],i) = m.Ax[j];
      }
    }
  }
  
  return D;
}

inline sparse dense_to_sparse(const MatrixXd& A,
                              bool symmetric = true){
  sparse As(A.rows(),A.cols(),A.data(),true); // this doesn't account for symmetric yet
  return As;
}

template<typename Derived>
inline MatrixXd operator*(const sparse& A, const DenseBase<Derived>& B){
  int m = B.cols();
  MatrixXd AB(A.n,m);
  AB.setZero();
  double val;
  if(A.rowMajor){
    for(int i = 0; i < A.n; i++){ //cycle over rows
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){ //elements in rows
        val = A.Ax[j]; // element in position (i,Ai[j])
        for(int k = 0; k<m; k++){
          AB(i,k) += val*B(A.Ai[j],k);
        }
      }
    }
  } else {
    for(int i = 0; i < A.m; i++){ //loop over columns
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){ //elements in columns
        val = A.Ax[j]; // element in position (Ai[j],i)
        for(int k = 0; k<m; k++){
          AB(A.Ai[j],k) += val*B(i,k);
        }
      }
    }
  }
  return AB;
}

template<typename Derived>
inline MatrixXd operator*(const DenseBase<Derived>& B, const sparse& A){
  int m = B.cols();
  MatrixXd AB(A.m,B.rows());
  AB.setZero();
  double val;
  if(A.rowMajor){
    for(int i = 0; i < A.n; i++){ //cycle over rows
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){ //elements in rows
        val = A.Ax[j]; // element in position (i,Ai[j])
        for(int k = 0; k<m; k++){
          AB(A.Ai[j],k) += val*B(k,i);
        }
      }
    }
    return AB;
  } else {
    for(int i = 0; i < A.m; i++){ //loop over columns
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){ //elements in columns
        val = A.Ax[j]; // element in position (Ai[j],i)
        for(int k = 0; k<m; k++){
          AB(i,k) += val*B(k,A.Ai[j]);
        }
      }
    }
    return AB.transpose();
  }
}

inline VectorXd operator*(const sparse& A, const VectorXd& B){
  VectorXd AB = VectorXd::Zero(A.n);
  if(A.rowMajor){
    for(int i = 0; i < A.n; i++){
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){
        AB(i) += A.Ax[j]*B(A.Ai[j]);
      }
    }
  } else {
    for(int i = 0; i < A.n; i++){
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){
        AB(A.Ai[j]) += A.Ax[j]*B(i);
      }
    }
  }
  return AB;
}

// multiplication of sparse and diagonal of a vector
inline sparse operator%(const sparse& A, const VectorXd& x){
  sparse Ax(A);
  if(A.rowMajor){
    for(unsigned int i = 0; i < A.Ax.size(); i++){
      Ax.Ax[i] *= x(Ax.Ai[i]);
    }
  } else {
    for(unsigned int i = 0; i < A.m; i++){
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){
        Ax.Ax[j] *= x(i);
      }
    } 
  }
  return Ax;
}

}
