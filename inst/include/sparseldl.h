/*

 This code is adapted from the LDL code reported at https://fossies.org/linux/SuiteSparse/LDL/Doc/ldl_userguide.pdf
 distributed under the GNU Lesser General Public License.
 
 LDL Copyright (c) 2005-2013 by Timothy A. Davis.
 LDL is also available under other licenses; contact the author for details.
 http://suitesparse.com

--------------------------------------------------------------------------------

LDL is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

LDL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 */
#pragma once

#include <cmath>
#include "sparsematrix.h"
#include "operators.h"

class SparseChol {
  int n;
  intvec Flag;
  intvec Parent;
  intvec Pattern;
  intvec LAp;
  public:
    sparse A_; // matrix to factorise
    sparse L;
    intvec Lnz;
    dblvec D;
    dblvec Y;
    
    SparseChol(
      const sparse& A
    ): Flag(A.n), Parent(A.n), Pattern(A.n), A_(A), Lnz(A.n) {
      n = A_.n;
      LAp = intvec(n+1);
      ldl_symbolic();
      L = sparse(LAp);
      L.m = n;
      D = dblvec(n);
      Y = dblvec(n);
    }
    
    SparseChol() {};
    
    void update(const sparse& A){
      Flag.resize(A.n);
      Parent.resize(A.n);
      Pattern.resize(A.n);
      A_ = A;
      Lnz.resize(A.n);
      n = A_.n;
      LAp = intvec(n+1);
      ldl_symbolic();
      L = sparse(LAp);
      L.m = A_.n;
      D = dblvec(n);
      Y = dblvec(n);
    }
    
    void ldl_symbolic(){
      int k, p, i, kk, p2;
      for (k = 0 ; k < n ; k++)
      {
        // L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
        Parent[k] = -1 ; // parent of k is not yet known */
        Flag[k] = k ; // mark node k as visited */
        Lnz[k] = 0 ; // count of nonzeros in column k of L */
#ifdef ENABLE_DEBUG_SCHOL
      Rcpp::Rcout <<  "\nLDL Symbolic. Permuted: " << A_.use_permuted;  
#endif
        kk = A_.use_permuted ? A_.P[k] : k;
        p2 = A_.Ap[kk+1];
        for (p = A_.Ap[k] ; p < p2 ; p++)
        {
          i = A_.use_permuted ? A_.Pinv[A_.Ai[p]] : A_.Ai[p];
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
      LAp[0] = 0 ;
      for (int k = 0 ; k < n ; k++)
      {
        LAp[k+1] = LAp[k] + Lnz[k] ;
      }
    }
    
    int ldl_numeric(){
      int p, len, kk, p2, i;
      for (int k = 0 ; k < n ; k++){
        // compute nonzero Pattern of kth row of L, in topological order */
        Y[k] = 0.0 ; // Y(0:k) is now all zero */
        int top = n ; // stack for pattern is empty */
        Flag[k] = k ; // mark node k as visited */
        Lnz[k] = 0 ; // count of nonzeros in column k of L */
#ifdef ENABLE_DEBUG_SCHOL
        Rcpp::Rcout << "\nLDL Numeric. Permuted: " << A_.use_permuted;  
#endif
        kk = A_.use_permuted ? A_.P[k] : k;
        p2 = A_.Ap[kk+1];
        for (p = A_.Ap[k] ; p < p2 ; p++)
        {
          i = A_.use_permuted ? A_.Pinv[A_.Ai[p]] : A_.Ai[p];
          if (i <= k)
          {
            Y[i] += A_.Ax[p]; // scatter A(i,k) into Y (sum duplicates) */
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
          p2 = L.Ap[i] + Lnz[i] ;
          for (p = L.Ap[i] ; p < p2 ; p++)
          {
            Y[L.Ai[p]] -= L.Ax[p] * yi ;
          }
          double l_ki = yi / D[i] ; // the nonzero entry L(k,i) */
          D[k] -= l_ki * yi ;
          L.Ai[p] = k ; // store L(k,i) in column form of L */
          L.Ax[p] = l_ki ;
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
        p2 = L.Ap[j+1] ;
        for (p = L.Ap[j] ; p < p2 ; p++)
        {
          x[L.Ai[p]] -= L.Ax[p] * x[j] ;
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
    
    void ldl_d2solve(double* x){
      int j ;
      for (j = 0 ; j < n ; j++)
      {
        x[j] /= sqrt(D[j]);
      }
    }
    
    void ldl_ltsolve(double * x){
      int j, p, p2 ;
      for (j = n-1 ; j >= 0 ; j--)
      {
        p2 = L.Ap[j+1];
        for (p = L.Ap[j] ; p < p2 ; p++)
        {
          x[j] -= L.Ax[p] * x[L.Ai[p]] ;
        }
      }
    }
    
    sparse LD(){
      sparse I = identity(L.n);
      I += L;
      I.transpose();
      dblvec Dsq(D);
      for(auto& d: Dsq) d = sqrt(d);
      I %= Dsq;
      I.n = L.n;
      I.m = L.m;
      return I;
    }
    
    
    // variant to modify in place
    void LD(sparse& mat){
      mat = identity(L.n);
      mat += L;
      mat.transpose();
      dblvec Dsq(D);
      for(auto& d: Dsq) d = sqrt(d);
      mat %= Dsq;
      mat.n = L.n;
      mat.m = L.m;
    }
};
