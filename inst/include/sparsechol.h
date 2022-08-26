#ifndef SPARSECHOL_H
#define SPARSECHOL_H
#include <Rcpp.h>
#include <vector>
// Code and methods from : https://fossies.org/linux/SuiteSparse/LDL/Doc/ldl_userguide.pdf
// Author : Tim Davies

struct sparse {
  int n;
  std::vector<int> Ap;
  std::vector<int> Ai;
  std::vector<double> Ax;
};

class SparseChol{
  int n;
  std::vector<int> Flag;
  std::vector<int> Parent;
  std::vector<int> Pattern;
  public:
    sparse* A_;
    sparse* L;
    std::vector<int> Lnz;
    std::vector<double> D;
    std::vector<double> Y;
    
    
    SparseChol(
      sparse* A
    ): Flag(A->n), Parent(A->n), Pattern(A->n), A_(A), Lnz(A->n) {
      n = A_->n;
      L = new sparse;
      L->n = n;
      L->Ap = std::vector<int>(n+1);
      ldl_symbolic();
      L->Ai = std::vector<int>(L->Ap[n]);
      L->Ax = std::vector<double>(L->Ap[n]);
      D = std::vector<double>(n);
      Y = std::vector<double>(n);
    }
    
    void ldl_symbolic(){
      int k, p, i;
      for (k = 0 ; k < n ; k++)
      {
        // L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
        Parent[k] = -1 ; // parent of k is not yet known */
        Flag[k] = k ; // mark node k as visited */
        Lnz[k] = 0 ; // count of nonzeros in column k of L */
        for (p = A_->Ap[k] ; p < A_->Ap[k+1] ; p++)
        {
          i = A_->Ai[p];
          //Rcpp::Rcout << "\n k:" << k << " i:" << i;
          if (i < k)
          {
            // follow path from i to root of etree, stop at flagged node */
            for ( ; Flag[i] != k ; i = Parent[i])
            {
              // find parent of i if not yet determined */
              if (Parent[i] == -1) Parent[i] = k ;
              Lnz[i]++ ; // L (k,i) is nonzero */
              Flag[i] = k ; // mark i as visited */
            }
          }
        }
      }
      // construct Lp index array from Lnz column counts */
      L->Ap[0] = 0 ;
      for (int k = 0 ; k < n ; k++)
      {
        L->Ap[k+1] = L->Ap[k] + Lnz[k] ;
      }
    }
    
    int ldl_numeric(){
      int p, len;
      //Rcpp::Rcout << "\nNLoop1";
      for (int k = 0 ; k < n ; k++){
        // compute nonzero Pattern of kth row of L, in topological order */
        Y[k] = 0.0 ; // Y(0:k) is now all zero */
        int top = n ; // stack for pattern is empty */
        Flag[k] = k ; // mark node k as visited */
        Lnz[k] = 0 ; // count of nonzeros in column k of L */
        int p2 = A_->Ap[k+1];
        for (p = A_->Ap[k] ; p < p2 ; p++)
        {
          int i = A_->Ai[p]; // get A(i,k) */
          if (i <= k)
          {
            Y[i] += A_->Ax[p]; // scatter A(i,k) into Y (sum duplicates) */
            for (len = 0 ; Flag[i] != k ; i = Parent[i])
            {
              Pattern[len++] = i ; // L(k,i) is nonzero */
              Flag[i] = k ; // mark i as visited */
            }
            while(len > 0) Pattern[--top] = Pattern[--len];
          }
        }
        // compute numerical values kth row of L (a sparse triangular solve) */
        D[k] = Y[k]; // get D(k,k) and clear Y(k) */
        Y[k] = 0.0 ;
        for ( ; top < n ; top++)
        {
          int i = Pattern[top]; // Pattern [top:n-1] is pattern of L(:,k) */
          double yi = Y[i]; // get and clear Y(i) */
          Y[i] = 0.0 ;
          p2 = L->Ap[i] + Lnz[i] ;
          for (p = L->Ap[i] ; p < p2 ; p++)
          {
            Y[L->Ai[p]] -= L->Ax[p] * yi ;
          }
          double l_ki = yi / D[i] ; // the nonzero entry L(k,i) */
          D[k] -= l_ki * yi ;
          L->Ai[p] = k ; // store L(k,i) in column form of L */
          L->Ax[p] = l_ki ;
          Lnz[i]++ ; // increment count of nonzeros in col i */
        }
        if (D[k] == 0.0) return(k) ; // failure, D(k,k) is zero */
      }
      return (n) ; // success, diagonal of D is all nonzero */
    }
      
    void ldl_lsolve(double* x){
      int p,p2;
      for (int j = 0 ; j < n ; j++)
      {
        p2 = L->Ap[j+1] ;
        for (p = L->Ap[j] ; p < p2 ; p++)
        {
          x[L->Ai[p]] -= L->Ax[p] * x[j] ;
        }
      }
    }
    
    void ldl_dsolve(double* x){
      int j ;
      for (j = 0 ; j < n ; j++)
      {
        x[j] /= D[j];
      }
    }
    
    void ldl_ltsolve(double * x){
      int j, p, p2 ;
      for (j = n-1 ; j >= 0 ; j--)
      {
        p2 = L->Ap[j+1];
        for (p = L->Ap[j] ; p < p2 ; p++)
        {
          x[j] -= L->Ax[p] * x[L->Ai[p]] ;
        }
      }
    }
    
    ~SparseChol(){
      delete L;
    }
    
};

#endif