#ifndef SPARSE_H
#define SPARSE_H

#include <vector>
#include <vector>
#include <algorithm>
#include <RcppEigen.h>

typedef std::vector<int> intvec;
typedef std::vector<double> dblvec;

class sparse {
public:
  int n; //rows
  int m; //cols
  intvec Ap;
  intvec Ai;
  dblvec Ax;
  // initialises an empty matrix all zeroes
  sparse(int n_, int m_);
  sparse(int n_, int m_, bool rowMajor);
  // this constructor uses column major formatting by default
  // change rowMajor = true for row major
  sparse(int n_, int m_, 
         const double* x,
         bool rowMajor = false);
  // this constructor uses column major formatting by default
  // change rowMajor = true for row major
  sparse(int n_, int m_, 
         const Rcpp::NumericMatrix& x,
         bool rowMajor = false);
  
  sparse(int n_, int m_, 
         const Eigen::MatrixXd& x,
         bool rowMajor = false);
  
  sparse(std::vector<int> p);
  sparse(){};
  sparse(const sparse& sp);
  sparse& operator=(sparse B);
  void insert(int row, int col, double x, bool rowMajor = false);
  void transpose();
  dblvec dense(bool symmetric = true);
  sparse& operator+=(const sparse& B);
  sparse& operator*=(const sparse& B);
  // right multiplication with a diagonal matrix represented a double vector
  sparse& operator%=(const dblvec& x);
};

inline sparse::sparse(int n_, int m_): n(n_), m(m_) {};

inline sparse::sparse(int n_, int m_, bool rowMajor): n(n_), m(m_) {
  if(rowMajor){
    Ap = intvec(n+1,0);
  } else {
    Ap = intvec(m+1,0);
  }
};

inline sparse::sparse(int n_, int m_, 
       const double* x,
       bool rowMajor) : n(n_), m(m_){
  if(rowMajor){
    for(int i = 0; i < n; i++){
      Ap.push_back(Ai.size());
      for(int j = 0; j < m; j++){
        double val = x[i + j*n];// assumes column major ordering
        if(val!=0){
          Ax.push_back(val);
          Ai.push_back(j);
        }
      }
    }
    Ap.push_back(Ax.size());
  } else {
    for(int i = 0; i < m; i++){
      Ap.push_back(Ai.size());
      for(int j = 0; j < n; j++){
        double val = x[j + i*n];// assumes column major ordering
        if(val!=0){
          Ax.push_back(val);
          Ai.push_back(j);
        }
      }
    }
    Ap.push_back(Ax.size());
  }
  
};
// this constructor uses column major formatting by default
// change rowMajor = true for row major
inline sparse::sparse(int n_, int m_, 
       const Rcpp::NumericMatrix& x,
       bool rowMajor) : n(n_), m(m_){
  if(rowMajor){
    for(int i = 0; i < n; i++){
      Ap.push_back(Ai.size());
      for(int j = 0; j < m; j++){
        double val = x(i, j);
        if(val!=0){
          Ax.push_back(val);
          Ai.push_back(j);
        }
      }
    }
    Ap.push_back(Ax.size());
  } else {
    for(int i = 0; i < m; i++){
      Ap.push_back(Ai.size());
      for(int j = 0; j < n; j++){
        double val = x(j, i);
        if(val!=0){
          Ax.push_back(val);
          Ai.push_back(j);
        }
      }
    }
    Ap.push_back(Ax.size());
  }
};

inline sparse::sparse(int n_, int m_, 
                      const Eigen::MatrixXd& x,
                      bool rowMajor) : n(n_), m(m_){
  if(rowMajor){
    for(int i = 0; i < n; i++){
      Ap.push_back(Ai.size());
      for(int j = 0; j < m; j++){
        double val = x(i, j);
        if(val!=0){
          Ax.push_back(val);
          Ai.push_back(j);
        }
      }
    }
    Ap.push_back(Ax.size());
  } else {
    for(int i = 0; i < m; i++){
      Ap.push_back(Ai.size());
      for(int j = 0; j < n; j++){
        double val = x(j, i);
        if(val!=0){
          Ax.push_back(val);
          Ai.push_back(j);
        }
      }
    }
    Ap.push_back(Ax.size());
  }
};

inline sparse::sparse(std::vector<int> p): Ap(p) {
  n = Ap.size() - 1;
  Ai = std::vector<int>(Ap[n]);
  Ax = std::vector<double>(Ap[n]);
};

inline sparse::sparse(const sparse& sp) : n(sp.n), m(sp.m), Ap(sp.Ap), Ai(sp.Ai), Ax(sp.Ax) {};

inline sparse& sparse::operator=(sparse B){
  n = B.n;
  m = B.m;
  Ap.swap(B.Ap);
  Ai.swap(B.Ai);
  Ax.swap(B.Ax);
  return *this;
}

inline void sparse::insert(int row, int col, double x, bool rowMajor){
  // this will fail if the matrix is not initialised
  if(rowMajor){
    int p = 0;
    if(Ap[row+1] - Ap[row] > 0){
      for(int j = Ap[row]; j < Ap[row+1]; j++){
        if(Ai[j] < col){
          p++;
        } else {
          break;
        }
      }
    }
    if(Ap[row]+p >= Ai.size()){
      Ai.push_back(col);
      Ax.push_back(x);
    } else {
      Ai.insert(Ai.begin()+Ap[row]+p,col);
      Ax.insert(Ax.begin()+Ap[row]+p,x);
    }
    for(int i = row+1; i < Ap.size(); i++)Ap[i]++;
  } else {
    int p = 0;
    if(Ap[col+1] - Ap[col] > 0){
      for(int j = Ap[col]; j < Ap[col+1]; j++){
        if(Ai[j] < row){
          p++;
        } else {
          break;
        }
      }
    }
    if(Ap[col]+p >= Ai.size()){
      Ai.push_back(row);
      Ax.push_back(x);
    } else {
      Ai.insert(Ai.begin()+Ap[col]+p,row);
      Ax.insert(Ax.begin()+Ap[col]+p,x);
    }
    for(int i = col+1; i < Ap.size(); i++)Ap[i]++;
  }
}

inline void sparse::transpose(){
  int nnz = Ax.size();
  sparse B;
  B.m = n;
  B.n = m;
  B.Ap = intvec(B.n+2,0);
  B.Ai = intvec(nnz,0);
  B.Ax = dblvec(nnz,0);
  for (int i = 0; i < nnz; ++i) {
    ++B.Ap[Ai[i] + 2];
  }
  for (int i = 2; i < B.Ap.size(); ++i) {
    B.Ap[i] += B.Ap[i - 1];
  }
  for (int i = 0; i < n; i++) {
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      const int new_index = B.Ap[Ai[j] + 1]++;
      B.Ax[new_index] = Ax[j];
      B.Ai[new_index] = i;
    }
  }
  B.Ap.pop_back(); // pop that one extra
  Ap = B.Ap;
  Ai = B.Ai;
  Ax = B.Ax;
  n = B.n;
  m = B.m;
};

inline dblvec sparse::dense(bool symmetric){
  dblvec D(n*m,0);
  for(int i = 0; i < n; i++){
    for(int j = Ap[i]; j < Ap[i+1]; j++){
      D[i+n*Ai[j]] = Ax[j];
      if(symmetric && Ai[j]!=i) D[Ai[j]+i*n] = D[i+n*Ai[j]];
    }
  }
  return D;
};

inline sparse& sparse::operator+=(const sparse& B){
  // this function has no checks on dimensions
  sparse AB;
  double val;
  int i,j;
  intvec tmpAi;
  dblvec tmpAx;
  for(i = 0; i<n; i++){
    AB.Ap.push_back(AB.Ai.size());
    tmpAi.clear();
    tmpAx.clear();
    tmpAi = intvec(Ai.begin()+Ap[i],Ai.begin()+Ap[i+1]);
    tmpAx = dblvec(Ax.begin()+Ap[i],Ax.begin()+Ap[i+1]);
    for(j = B.Ap[i]; j < B.Ap[i+1]; j++){
      auto elem = std::lower_bound(tmpAi.begin(),tmpAi.end(),B.Ai[j]);
      int idx = elem - tmpAi.begin();
      if(elem!=tmpAi.end() && *elem==B.Ai[j]){
        tmpAx[idx] += B.Ax[j];
      } else {
        tmpAi.insert(elem,B.Ai[j]);
        tmpAx.insert(tmpAx.begin()+idx,B.Ax[j]);
      }
    }
    AB.Ax.insert(AB.Ax.end(),tmpAx.begin(),tmpAx.end());
    AB.Ai.insert(AB.Ai.end(),tmpAi.begin(),tmpAi.end());
  }
  AB.Ap.push_back(AB.Ax.size());
  Ax = AB.Ax;
  Ap = AB.Ap;
  Ai = AB.Ai;
  return *this;
};

inline sparse& sparse::operator*=(const sparse& B){
  if(m != B.n)Rcpp::stop("wrong dimension in sparse-sparse multiplication");
  sparse AB;
  double val;
  intvec tmpAi;
  dblvec tmpAx;
  int i,j,k;
  for(i = 0; i < n; i++){
    AB.Ap.push_back(AB.Ai.size());
    tmpAi.clear();
    tmpAx.clear();
    for(j = Ap[i]; j < Ap[i+1]; j++){
      for(k = B.Ap[Ai[j]]; k < B.Ap[Ai[j]+1]; k++){
        val = Ax[j]*B.Ax[k];
        auto elem = std::lower_bound(tmpAi.begin(),tmpAi.end(),B.Ai[k]);
        int idx = elem - tmpAi.begin();
        if(elem!=tmpAi.end() && *elem==B.Ai[k]){
          tmpAx[idx] += val;
        } else {
          tmpAi.insert(elem,B.Ai[k]);
          tmpAx.insert(tmpAx.begin()+idx,val);
        }
      }
    }
    AB.Ax.insert(AB.Ax.end(),tmpAx.begin(),tmpAx.end());
    AB.Ai.insert(AB.Ai.end(),tmpAi.begin(),tmpAi.end());
  }
  AB.Ap.push_back(AB.Ax.size());
  Ax = AB.Ax;
  Ap = AB.Ap;
  Ai = AB.Ai;
  m = B.m;
  return *this;
};

// right multiplication with a diagonal matrix represented a double vector
inline sparse& sparse::operator%=(const dblvec& x){
  for(int i = 0; i < Ax.size(); i++){
    Ax[i] *= x[Ai[i]];
  }
  return *this;
}

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

#endif