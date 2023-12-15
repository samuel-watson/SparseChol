/*
License for the AMD algorithm and LDL decomposition algorithms 
 
 AMD, Copyright (c), 1996-2022, Timothy A. Davis,
Patrick R. Amestoy, and Iain S. Duff.  All Rights Reserved.

Availability:
  
  http://www.suitesparse.com
  
  -------------------------------------------------------------------------------
    AMD License: BSD 3-clause:
  -------------------------------------------------------------------------------
    
    Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the organizations to which the authors are
affiliated, nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#pragma once 

#include <vector>
#include <algorithm>
#include <RcppEigen.h>

#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) ((i < EMPTY) ? FLIP (i) : (i))
// #define ENABLE_DEBUG

typedef std::vector<int> intvec;
typedef std::vector<double> dblvec;

#ifdef ENABLE_DEBUG
inline void CHECK_INDEX(int idx, intvec& vec, int ERR = 0){
  if(idx > vec.size()-1){
    Rcpp::Rcout << "\nERROR: " << ERR;
    Rcpp::Rcout << "\nIndex > size. Index: " << idx << " size: " << vec.size();
    Rcpp::stop("INDEX OUT OF RANGE");
  }
  if(idx < 0){
    Rcpp::Rcout << "\nERROR: " << ERR;
    Rcpp::Rcout << "\nIndex < 0. Index: " << idx;
    Rcpp::stop("INDEX < 0");
  }
}
#endif


struct AMDInfo{
  int status = 0;
  int n = 0;
  int nz = 0;
  int symmetry = 0;
  int nzdiag = 0;
  int nz_a_plus_at = 0;
  double lnz = 0;
  double ndiv = 0;
  double nms_lu = 0;
  double nms_ldl = 0;
  double dmax = 1;
  double f = 0;
  double r = 0;
  double s = 0;
  double lnzme = 0;
  int ndense = 0;
};

class sparse {
  friend class SparseChol;
  
public:
  int n; //rows
  int m; //cols
  intvec Ap;
  intvec Ai;
  dblvec Ax;
  bool rowMajor = true;
  // initialises an empty matrix all zeroes
  sparse(int n_, int m_);
  sparse(int n_, int m_, bool rowMajor_);
  // this constructor uses column major formatting by default
  // change rowMajor = true for row major
  sparse(int n_, int m_, 
         const double* x,
         bool rowMajor_ = true);
  // this constructor uses column major formatting by default
  // change rowMajor = true for row major
  sparse(int n_, int m_, 
         const Rcpp::NumericMatrix& x,
         bool rowMajor_ = true);
  
  sparse(int n_, int m_, 
         const Eigen::MatrixXd& x,
         bool rowMajor_ = true);
  
  sparse(std::vector<int> p);
  sparse(){};
  sparse(const sparse& sp);
  double operator()(const int row, const int col);
  sparse& operator=(sparse B);
  void insert(int row, int col, double x);
  void transpose();
  dblvec dense(bool symmetric = true);
  sparse& operator+=(const sparse& B);
  sparse& operator*=(const sparse& B);
  sparse& operator%=(const dblvec& x);
  void calculate_amd_permute();
  intvec permute();
  intvec permute_inv(); 
protected:
  intvec P;
  intvec Pinv;
  AMDInfo info;
  void AMD_aat(intvec& Len, intvec& Tp);
  int AMD_post_tree(int root, int k_in, intvec& Child, const intvec& Sibling, intvec& Order, intvec& Stack);
  void AMD_order();
  bool use_permuted = false;
  
};

inline int clear_flag (int wflg, int wbig, intvec& W)
{
  int x ;
  if (wflg < 2 || wflg >= wbig)
  {
    for(auto& w: W)if (w != 0) w = 1 ;
    wflg = 2 ;
  }
  return (wflg) ;
}

inline sparse::sparse(int n_, int m_): n(n_), m(m_) {};

inline sparse::sparse(int n_, int m_, bool rowMajor_): n(n_), m(m_) {
  rowMajor = rowMajor_;
  if(rowMajor){
    Ap = intvec(n+1,0);
  } else {
    Ap = intvec(m+1,0);
  }
};

inline sparse::sparse(int n_, int m_, 
       const double* x,
       bool rowMajor_) : n(n_), m(m_){
  rowMajor = rowMajor_;
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
       bool rowMajor_) : n(n_), m(m_){
  rowMajor = rowMajor_;
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
                      bool rowMajor_) : n(n_), m(m_){
  rowMajor = rowMajor_;
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

inline sparse::sparse(const sparse& sp) : n(sp.n), m(sp.m), Ap(sp.Ap), Ai(sp.Ai), Ax(sp.Ax),
  P(sp.P), Pinv(sp.Pinv), use_permuted(sp.use_permuted), rowMajor(sp.rowMajor) {};

inline sparse& sparse::operator=(sparse B){
  n = B.n;
  m = B.m;
  Ap.swap(B.Ap);
  Ai.swap(B.Ai);
  Ax.swap(B.Ax);
  P.swap(B.P);
  Pinv.swap(B.Pinv);
  use_permuted = B.use_permuted;
  rowMajor = B.rowMajor;
  return *this;
}

inline double sparse::operator()(const int row, const int col){
  bool found = false;
  int i;
  if(rowMajor){
    for(i = Ap[row]; i < Ap[row+1]; i++){
      if(Ai[i] == col){
        found = true;
        break;
      }
    }
    if(!found){
      return 0;
    } else {
      return Ax[i];
    }
  } else {
    for(i = Ap[col]; i < Ap[col+1]; i++){
      if(Ai[i] == row){
        found = true;
        
        break;
      }
    }
    if(!found){
      return 0;
    } else {
      return Ax[i];
    }
  }
}

inline intvec sparse::permute(){
  if(P.size() == 0)AMD_order();
  return P;
}

inline void sparse::calculate_amd_permute(){
  if(P.size() == 0)AMD_order();
}


inline intvec sparse::permute_inv(){
  if(P.size() == 0)AMD_order();
  return Pinv;
}

inline void sparse::insert(int row, int col, double x){
  // this will fail if the matrix is not initialised
  if(Ap.size()==0)Rcpp::stop("Matrix not properly initialised");
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

// WARNIING THIS HASN'T BEEN UPDATED FOR IF THE SPARSE MATRICES ARE STORED IN COLUMN MAJOR ORDERING
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

inline void sparse::AMD_aat(intvec& Len, intvec& Tp)
{
  double sym ;
  size_t nzaat ;
  for(auto& k: Len)k = 0;
  int nzdiag = 0 ;
  int nzboth = 0 ;
  int nz = Ap.back();
  int p1, p2, pj2, p, pj;
  int n = Ap.size() - 1;

  for (int k = 0 ; k < n ; k++)
  {
    p1 = Ap[k];
    p2 = Ap[k+1];
    for (p = p1 ; p < p2 ; )
    {
      int j = Ai[p];
      if (j < k)
      {
        Len[j]++;
        Len[k]++;
        p++ ;
      }
      else if (j == k)
      {
        p++ ;
        nzdiag++ ;
        break ;
      }
      else
      {
        break ;
      }
#ifdef ENABLE_DEBUG
      CHECK_INDEX(j+1,Ap,464);
      CHECK_INDEX(j+1,Tp,465);
#endif
      pj2 = Ap[j+1] ;
      for (pj = Tp[j] ; pj < pj2 ; )
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(pj,Ai,471);
#endif
        int i = Ai [pj] ;
        if (i < k)
        {
          Len[i]++;
          Len[j]++;
          pj++ ;
        }
        else if (i == k)
        {
          pj++ ;
          nzboth++ ;
          break ;
        }
        else
        {
          break ;
        }
      }
#ifdef ENABLE_DEBUG
      CHECK_INDEX(j,Tp,492);
#endif
      Tp[j] = pj ;
    }
#ifdef ENABLE_DEBUG
    CHECK_INDEX(k,Tp,497);
#endif
    Tp[k] = p ;
  }
  for (int j = 0 ; j < n ; j++)
  {
    for (pj = Tp[j] ; pj < Ap[j+1] ; pj++)
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(pj,Ai,490);
#endif
      int i = Ai [pj] ;
      Len[i]++;
      Len[j]++;
    }
  }
  if (nz == nzdiag)
  {
    sym = 1 ;
  }
  else
  {
    sym = (2 * (double) nzboth) / ((double) (nz - nzdiag)) ;
  }
  nzaat = 0 ;
  for (int k = 0 ; k < n ; k++)nzaat += Len[k];

  info.status = 0;
  info.n = n;
  info.nz = nz;
  info.symmetry = sym;
  info.nzdiag = nzdiag;
  info.nz_a_plus_at = nzaat;
}

inline int sparse::AMD_post_tree(int root,
                                 int k_in,
                                 intvec& Child,
                                 const intvec& Sibling,
                                 intvec& Order,
                                 intvec& Stack)
{
  int k = k_in;
  int head = 0;
  Stack[0] = root;
  while (head >= 0)
  {
#ifdef ENABLE_DEBUG
    CHECK_INDEX(head,Stack,545);
#endif
    int i = Stack[head];
#ifdef ENABLE_DEBUG
    CHECK_INDEX(i,Child,549);
#endif
    if (Child[i] != EMPTY)
    {
      for (int f = Child[i]; f != EMPTY; f = Sibling[f])head++;
      int h = head;
      for (int f = Child[i]; f != EMPTY; f = Sibling[f])Stack[h--] = f;
      Child[i] = EMPTY;
    }
    else
    {
      head--;
#ifdef ENABLE_DEBUG
      CHECK_INDEX(i,Order,562);
#endif
      Order[i] = k++;
    }
  }
  return k;
}

inline void sparse::AMD_order()
{
  use_permuted = true;
  double mem = 0;
  int nz = Ap.back();
  int nn = Ap.size() - 1;
  intvec Len(nn,0);
  intvec Pe(nn,0);
  intvec Nv(nn,0);
  intvec Head(nn,0);
  intvec Elen(nn,0);
  intvec Degree(nn,0);
  intvec Tp(nn,0); // W
  
  if(P.size() == 0 || Pinv.size() ==0){
    P.resize(nn);
    Pinv.resize(nn);
  }
  std::fill(P.begin(), P.end(), 0.0);
  std::fill(Pinv.begin(), Pinv.end(), 0.0);
  info.n = nn;
#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nSTART AAT";
#endif
  AMD_aat (Len, P);
  int slen = info.nz_a_plus_at;			/* space for matrix */
  slen += info.nz_a_plus_at/5 ;			/* add elbow room */
  for (int i = 0 ; i < 7 ; i++)slen += nn ;
  int iwlen = slen - 6*nn;
  
#ifdef ENABLE_DEBUG
    Rcpp::Rcout << "\nSlen: " << slen << " iwlen: " << iwlen;
#endif
    
  intvec Iw(iwlen,0);

  int pfree = 0;
  for (int j = 0 ; j < nn; j++)
  {
    Pe[j] = pfree ;
    Nv[j] = pfree ;
    pfree += Len[j] ;
  }
#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nLoop 1, pfree: " << pfree;
#endif

  for (int k = 0 ; k < nn; k++)
  {
#ifdef ENABLE_DEBUG
    Rcpp::Rcout << "\nConstruct row/column k=" << k << " of A+A";
    CHECK_INDEX(k+1,Ap,619);
#endif
    int p1 = Ap[k];
    int p2 = Ap[k+1];
    int p;
    for (p = p1 ; p < p2 ; )
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(p,Ai,627);
#endif
      int j = Ai[p] ;
#ifdef ENABLE_DEBUG
      CHECK_INDEX(j,Nv,632);
#endif
      if (j < k)
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(Nv[j],Iw,634);
        CHECK_INDEX(Nv[k],Iw,635);
#endif
        Iw[Nv[j]++] = k;
        Iw[Nv[k]++] = j;
        p++ ;
      }
      else if (j == k)
      {
        p++ ;
        break ;
      }
      else /* j > k */
      {
        break ;
      }
#ifdef ENABLE_DEBUG
      CHECK_INDEX(j+1,Ap,651);
      if(Ap[j] > Tp[j] || Tp[j] > Ap[j+1])Rcpp::stop("Error 655 ASSERTION");
#endif
      int pj2 = Ap[j+1] ;
      int pj;
      for (pj = Tp[j] ; pj < pj2 ; )
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(pj,Ai,658);
#endif
        int i = Ai[pj] ;
        if (i < k)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(j,Nv,664);
          CHECK_INDEX(Nv[j],Iw,665);
          CHECK_INDEX(Nv[i],Iw,667);
#endif
          Iw[Nv[i]++] = j;
          Iw[Nv[j]++] = i;
          pj++ ;
        }
        else if (i == k)
        {
          pj++ ;
          break ;
        }
        else /* i > k */
        {
          break ;
        }
      }
#ifdef ENABLE_DEBUG
      CHECK_INDEX(j,Tp,683);
#endif
      Tp[j] = pj;
    }
#ifdef ENABLE_DEBUG
    CHECK_INDEX(k,Tp,688);
#endif
    Tp[k] = p;
  }
  
#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nLoop 2, Tp: ";
  for(const auto& t: Tp) Rcpp::Rcout << t << " ";
#endif

  for (int j = 0 ; j < nn ; j++)
  {
    for (int pj = Tp[j] ; pj < Ap[j+1] ; pj++)
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(pj,Ai,703);
      CHECK_INDEX(Nv[j],Iw,704);
#endif
      int i = Ai[pj];
#ifdef ENABLE_DEBUG
      CHECK_INDEX(Nv[i],Iw,708);
#endif
      Iw[Nv[i]++] = j ;
      Iw[Nv[j]++] = i ;
    }
  }
#ifdef ENABLE_DEBUG
  for (int j = 0 ; j < nn-1 ; j++){
    if(Nv[j]!=Pe[j+1])Rcpp::stop("Error 720 ASSERTION");
  }
  if(Nv[nn-1]!=pfree)Rcpp::stop("722 NV != pfree");
  Rcpp::Rcout << "\nLoop 3, Iw: ";
for(const auto& t: Iw) Rcpp::Rcout << t << " ";
#endif


  /* initialize output statistics */
  std::fill(Tp.begin(),Tp.end(),0.0);
  std::fill(Nv.begin(),Nv.end(),0.0);

  int me = 0;
  int mindeg = 0 ;
  int ncmpa = 0 ;
  int nel = 0 ;
  int lemax = 0 ;
  double alpha = 10;
  int aggressive = 1;
  unsigned int hash = 0;
  double dense = alpha * sqrt ((double) nn) ;
  dense = dense > 16 ? dense : 16.0;
  dense = dense < nn ? dense : nn;
  
#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nInitialise output statistics, dense: " << dense;
#endif

  for (int i = 0 ; i < nn ; i++)
  {
    P[i] = EMPTY;
    Head[i] = EMPTY;
    Pinv[i] = EMPTY;
    Nv [i] = 1;
    Tp[i] = 1;
    Elen[i] = 0;
    Degree[i] = Len[i];
  }

  int wbig = INT_MAX - nn;
  int wflg = clear_flag(0, wbig, Tp) ;
  info.ndense = 0 ;
  int deg, inext, ilast;
  
  for (int i = 0 ; i < nn ; i++)
  {
    deg = Degree[i] ;
    if (deg == 0)
    {
      Elen [i] = FLIP(1);
      nel++ ;
      Pe[i] = EMPTY;
      Tp[i] = 0;
    }
    else if (deg > dense)
    {
#ifdef ENABLE_DEBUG
      Rcpp::Rcout << "\nDense node " << i << " degree " << deg;
#endif
      info.ndense++ ;
      Nv[i] = 0 ;
      Elen[i] = EMPTY ;
      nel++ ;
      Pe[i] = EMPTY ;
    }
    else
    {
      inext = Head[deg];
#ifdef ENABLE_DEBUG
      CHECK_INDEX(deg,Head,776);
#endif
      if(inext != EMPTY) {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(inext,P,780);
#endif
        P[inext] = i;
      }
      Pinv[i] = inext;
      Head[deg] = i;
    }
  }
  
#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nLoop 4, nel: " << nel << " ndense: " << info.ndense;
#endif

  while (nel < nn)
  {
    for (deg = mindeg ; deg < n ; deg++)
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(deg,Head,795);
#endif
      me = Head[deg];
      if (me != EMPTY) break;
    }
    mindeg = deg ;
    inext = Pinv[me];
    if(inext != EMPTY) P[inext] = EMPTY;
    
#ifdef ENABLE_DEBUG
    CHECK_INDEX(deg,Head,805);
    CHECK_INDEX(me,Elen,806);
#endif
    
    Head[deg] = inext;
    int elenme = Elen[me];
    int nvpiv = Nv[me];
#ifdef ENABLE_DEBUG
    if(nvpiv <= 0)Rcpp::stop("830 nvpiv <= 0");
#endif
    nel += nvpiv;
    Nv[me] = -nvpiv;
    int degme = 0 ;

    int pme1, pme2;
    if (elenme == 0)
    {
      pme1 = Pe[me];
      pme2 = pme1 - 1;
      for (int p = pme1 ; p <= pme1 + Len[me] - 1; p++)
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(p,Iw,824);
#endif
        int i = Iw[p];
        
#ifdef ENABLE_DEBUG
        CHECK_INDEX(i,Nv,829);
        CHECK_INDEX(ilast,Pinv,831);
#endif
        
        int nvi = Nv[i] ;
        if (nvi > 0)
        {
          degme += nvi;
          Nv[i] = -nvi;
          Iw[++pme2] = i;
          int ilast = P[i];
          int inext = Pinv[i];
          if (inext != EMPTY) {
#ifdef ENABLE_DEBUG
            CHECK_INDEX(inext,P,847);
#endif
            P[inext] = ilast;
          }
          if (ilast != EMPTY)
          {
            Pinv[ilast] = inext;
          }
          else
          {
#ifdef ENABLE_DEBUG
            CHECK_INDEX(Degree[i],Head,850);
#endif
            Head[Degree[i]] = inext;
          }
        }
      }
    }
    else
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(me,Pe,860);
#endif
      int p = Pe[me];
      pme1 = pfree;
      int slenme = Len[me] - elenme;
      int e, pj, ln;
      for (int knt1 = 1 ; knt1 <= elenme + 1 ; knt1++)
      {
        if (knt1 > elenme)
        {
          e = me ;
          pj = p ;
          ln = slenme ;
#ifdef ENABLE_DEBUG
          Rcpp::Rcout << "\nSearch sv: " << me << " " << pj << " " << ln;
#endif
        }
        else
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(p,Iw,877);
#endif
          e = Iw [p++];
#ifdef ENABLE_DEBUG
          CHECK_INDEX(e,Pe,878);
#endif
          pj = Pe [e];
          ln = Len [e];
#ifdef ENABLE_DEBUG
          Rcpp::Rcout << "Search element e " << e << " in me " << me;
          if(Elen[e] >= EMPTY)Rcpp::stop("914 Elen-e > empty");
          if(Tp[e] <= 0)Rcpp::stop("914 W-e < 0");
          if(pj < 0)Rcpp::stop("914 pj < 0");
#endif
        }
        
#ifdef ENABLE_DEBUG
        if(ln < 0) Rcpp::stop("921 ln < 0");
#endif 
        
        for (int knt2 = 1 ; knt2 <= ln ; knt2++)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(pj,Iw,887);
#endif
          int i = Iw[pj++];
#ifdef ENABLE_DEBUG
          CHECK_INDEX(i,Nv,888);
#endif
          int nvi = Nv [i] ;
#ifdef ENABLE_DEBUG
          Rcpp::Rcout << "\nVars i " << i << " elen[i] " << Elen[i] << " Nv[i] " << Nv[i] << " wflg " << wflg;
#endif
          if (nvi > 0)
          {
            if (pfree >= iwlen)
            {
#ifdef ENABLE_DEBUG
              Rcpp::Rcout << "\nGARBAGE COLLECTION";
              CHECK_INDEX(me,Pe,899);
              CHECK_INDEX(e,Pe,900);
#endif
              Pe[me] = p;
              Len[me] -= knt1;
              if(Len[me] == 0) Pe[me] = EMPTY;
              Pe[e] = pj;
              Len[e] = ln - knt2;
              if(Len[e] == 0) Pe[e] = EMPTY;
              ncmpa++;
              for (int j = 0 ; j < n ; j++)
              {
                int pn = Pe [j];
                if (pn >= 0)
                {
#ifdef ENABLE_DEBUG
                  CHECK_INDEX(j,Pe,915);
                  CHECK_INDEX(pn,Iw,916);
#endif
                  Pe[j] = Iw[pn];
                  Iw[pn] = FLIP(j);
                }
              }
              int psrc = 0;
              int pdst = 0;
              int pend = pme1 - 1;

              while (psrc <= pend)
              {
#ifdef ENABLE_DEBUG
                CHECK_INDEX(psrc,Iw,929);
#endif
                int j = FLIP(Iw[psrc++]);
                if (j >= 0)
                {
#ifdef ENABLE_DEBUG
                  CHECK_INDEX(pdst,Iw,935);
                  CHECK_INDEX(j,Pe,936);
#endif
                  Iw[pdst] = Pe[j] ;
                  Pe[j] = pdst++;
                  int lenj = Len[j];
                  for (int knt3 = 0 ; knt3 <= lenj - 2 ; knt3++)
                  {
#ifdef ENABLE_DEBUG
                    CHECK_INDEX(pdst,Iw,944);
                    CHECK_INDEX(psrc,Iw,945);
#endif
                    Iw[pdst++] = Iw[psrc++];
                  }
                }
              }
              int p1 = pdst;
              for (psrc = pme1 ; psrc <= pfree-1 ; psrc++)
              {
                Iw[pdst++] = Iw[psrc];
              }
              pme1 = p1 ;
              pfree = pdst ;
#ifdef ENABLE_DEBUG
              CHECK_INDEX(me,Pe,959);
              CHECK_INDEX(e,Pe,960);
#endif
              pj = Pe [e] ;
              p = Pe [me] ;

            }
            degme += nvi;
#ifdef ENABLE_DEBUG
            CHECK_INDEX(i,Nv,968);
            CHECK_INDEX(pfree,Iw,969);
#endif
            Nv[i] = -nvi;
            Iw[pfree++] = i;
            
#ifdef ENABLE_DEBUG
            Rcpp::Rcout << "\nVars i " << i << " nv " << Nv[i];
#endif
            
            int ilast = P[i];
            int inext = Pinv[i];
            if (inext != EMPTY) {
#ifdef ENABLE_DEBUG
              CHECK_INDEX(inext,Iw,970);
#endif
              P[inext] = ilast;
            }
            if (ilast != EMPTY)
            {
#ifdef ENABLE_DEBUG
              CHECK_INDEX(ilast,Pinv,981);
#endif
              Pinv[ilast] = inext;
            }
            else
            {
#ifdef ENABLE_DEBUG
              CHECK_INDEX(Degree[i],Head,988);
#endif
              Head[Degree[i]] = inext;
            }
          }
        }
        if (e != me)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(e,Pe,997);
          Rcpp::Rcout << "\nELEMENT " << e << " => " << me;
#endif
          Pe[e] = FLIP(me);
          Tp[e] = 0;
        }
      }
      pme2 = pfree - 1;
    }
    
#ifdef ENABLE_DEBUG
    CHECK_INDEX(me,Pe,1006);
#endif
    Degree[me] = degme;
    Pe[me] = pme1;
    Len[me] = pme2 - pme1 + 1;
    Elen[me] = FLIP(nvpiv + degme);
    
#ifdef ENABLE_DEBUG
    Rcpp::Rcout << "\nNew element structure: length = " << pme2-pme1+1 << "\n";
    for (int pme = pme1 ; pme <= pme2 ; pme++) Rcpp::Rcout << " " << Iw[pme];
    Rcpp::Rcout << "\nme: ";
#endif
    
    wflg = clear_flag (wflg, wbig, Tp) ;

    for (int pme = pme1 ; pme <= pme2 ; pme++)
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(pme,Iw,1017);
#endif
      int i = Iw[pme];
#ifdef ENABLE_DEBUG
      CHECK_INDEX(i,Elen,1021);
#endif
      int eln = Elen[i];
      
#ifdef ENABLE_DEBUG
      Rcpp::Rcout << "\ni " << i << " Elen " << eln;
#endif
      
      if (eln > 0)
      {
        int nvi = -Nv[i];
        int wnvi = wflg - nvi;
        for (int p = Pe[i] ; p <= Pe[i] + eln - 1 ; p++)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(p,Iw,1031);
#endif
          int e = Iw[p];
#ifdef ENABLE_DEBUG
          CHECK_INDEX(e,Tp,1035);
#endif
          int we = Tp[e];
#ifdef ENABLE_DEBUG
          Rcpp::Rcout << "\nVars e " << e << " we " << we;
#endif
          if (we >= wflg)
          {
#ifdef ENABLE_DEBUG
            Rcpp::Rcout << "  Unabsorbed, first time seen ";
#endif
            we -= nvi ;
          }
          else if (we != 0)
          {
#ifdef ENABLE_DEBUG
            Rcpp::Rcout << "  Unabsorbed ";
#endif
            we = Degree [e] + wnvi ;
          }
#ifdef ENABLE_DEBUG
          Rcpp::Rcout << "\n ";
#endif
          Tp[e] = we;
        }
      }
    }

    for (int pme = pme1 ; pme <= pme2 ; pme++)
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(pme,Iw,1054);
#endif
      int i = Iw[pme];
#ifdef ENABLE_DEBUG
      CHECK_INDEX(i,Pe,1058);
      Rcpp::Rcout << "\nUpdating: i " << i << " elen " << Elen[i] << " len " << Len[i];
#endif
      int p1 = Pe[i];
      int p2 = p1 + Elen[i] - 1;
      int pn = p1;
      int hash = 0 ;
      int deg = 0 ;
      if (aggressive)
      {
        for (int p = p1 ; p <= p2 ; p++)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(p,Iw,1070);
#endif
          int e = Iw[p];
#ifdef ENABLE_DEBUG
          CHECK_INDEX(e,Tp,1074);
#endif
          int we = Tp[e];
          if (we != 0)
          {
            int dext = we - wflg ;
            if (dext > 0)
            {
#ifdef ENABLE_DEBUG
              CHECK_INDEX(pn,Iw,1083);
#endif
              deg += dext;
              Iw[pn++] = e;
              hash += e;
#ifdef ENABLE_DEBUG
              Rcpp::Rcout << "\ne: " << e << " hash = " << hash;
#endif
            }
            else
            {
#ifdef ENABLE_DEBUG
              Rcpp::Rcout << "\nElement " << e << " => " << me << " (aggressive)";
              if(dext != 0)Rcpp::stop("1175 dext != 0");
#endif
              Pe[e] =FLIP(me);
              Tp[e] = 0;
            }
          }
        }
      }
      else
      {
        for (int p = p1 ; p <= p2 ; p++)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(p,Iw,1102);
#endif
          int e = Iw[p];
#ifdef ENABLE_DEBUG
          CHECK_INDEX(e,Tp,1106);
#endif
          int we = Tp[e] ;
          if (we != 0)
          {
#ifdef ENABLE_DEBUG
            CHECK_INDEX(pn,Iw,1112);
#endif
            int dext = we - wflg ;
            deg += dext ;
            Iw[pn++] = e;
            hash += e;
#ifdef ENABLE_DEBUG
            Rcpp::Rcout << "\ne: " << e << " hash = " << hash;
#endif
          }
        }
      }
#ifdef ENABLE_DEBUG
      CHECK_INDEX(i,Elen,1122);
#endif
      Elen[i] = pn - p1 + 1 ;
      int p3 = pn ;
      int p4 = p1 + Len[i];
      for (int p = p2 + 1 ; p < p4 ; p++)
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(p,Iw,1130);
#endif
        int j = Iw[p];
#ifdef ENABLE_DEBUG
        CHECK_INDEX(j,Nv,1134);
#endif
        int nvj = Nv[j];
        if (nvj > 0)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(pn,Iw,1140);
#endif
          deg += nvj;
          Iw[pn++] = j;
          hash += j;
#ifdef ENABLE_DEBUG
          Rcpp::Rcout << "\nVars: s: " << j << " hash " << hash << " Nv[j] " << nvj;
#endif
        }
      }

      if (Elen[i] == 1 && p3 == pn)
      {
#ifdef ENABLE_DEBUG
        Rcpp::Rcout << "\nMASS i "<< i << " => parent e " << me;
#endif
        Pe[i] = FLIP(me);
        int nvi = -Nv[i];
        degme -= nvi;
        nvpiv += nvi;
        nel += nvi;
        Nv[i] = 0;
        Elen[i] = EMPTY;

      }
      else
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(pn,Iw,1162);
        CHECK_INDEX(p3,Iw,1163);
        CHECK_INDEX(p1,Iw,1164);
#endif
        Degree[i] = Degree[i] < deg ? Degree[i] : deg;
        Iw[pn] = Iw[p3];
        Iw[p3] = Iw[p1];
        Iw[p1] = me;
        Len[i] = pn - p1 + 1;
        hash = hash % n;
#ifdef ENABLE_DEBUG
        CHECK_INDEX(hash,Head,1173);
#endif
        int j = Head[hash];
        if (j <= EMPTY)
        {
          Pinv[i] = FLIP(j);
          Head[hash] = FLIP(i);
        }
        else
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(j,P,1184);
#endif
          Pinv[i] = P[j];
          P[j] = i;
        }
#ifdef ENABLE_DEBUG
        CHECK_INDEX(i,P,1190);
#endif
        P[i] = hash;
      }
    }
#ifdef ENABLE_DEBUG
    CHECK_INDEX(me,Degree,1196);
#endif
    Degree[me] = degme;
    lemax = lemax > degme ? lemax : degme;
    wflg += lemax ;
    wflg = clear_flag (wflg, wbig, Tp) ;
    
#ifdef ENABLE_DEBUG
    Rcpp::Rcout << "\nSupervariable detection:";
#endif 
    
    for (int pme = pme1 ; pme <= pme2 ; pme++)
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(pme,Iw,1205);
#endif
      int i = Iw[pme];
#ifdef ENABLE_DEBUG
      CHECK_INDEX(i,Nv,1209);
      Rcpp::Rcout << "\nConsider i "<< i << " nv " << Nv[i];
#endif
      if (Nv[i] < 0)
      {
        hash = P[i];
#ifdef ENABLE_DEBUG
        CHECK_INDEX(hash,Head,1215);
#endif
        int j = Head[hash];
        if (j == EMPTY)
        {
          i = EMPTY;
        }
        else if (j < EMPTY)
        {
          i = FLIP(j);
          Head[hash] = EMPTY;
        }
        else
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(j,P,1230);
#endif
          i = P[j];
          P[j] = EMPTY;
        }
        
#ifdef ENABLE_DEBUG
        Rcpp::Rcout << "\n----i " << i << " hash " << hash;
#endif  
        
        while (i != EMPTY && Pinv[i] != EMPTY)
        {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(i, Len,1238);
          Rcpp::Rcout << "\ncompare i " << i << " and j " << j;
#endif
          int ln = Len [i] ;
          int eln = Elen [i] ;
          for (int p = Pe [i] + 1 ; p <= Pe [i] + ln - 1 ; p++)
          {
#ifdef ENABLE_DEBUG
            CHECK_INDEX(Iw[p],Tp,1245);
#endif
            Tp[Iw[p]] = wflg;
          }
          int jlast = i ;
          j = Pinv [i] ;
          while (j != EMPTY)
          {
#ifdef ENABLE_DEBUG
            CHECK_INDEX(j,Len,1254);
#endif
            int ok = (Len [j] == ln) && (Elen [j] == eln) ;
            for (int p = Pe [j] + 1 ; ok && p <= Pe [j] + ln - 1 ; p++)if (Tp[Iw[p]] != wflg) ok = 0;
            if (ok)
            {
#ifdef ENABLE_DEBUG
              CHECK_INDEX(j,Pe,1261);
              CHECK_INDEX(i,Nv,1262);
              CHECK_INDEX(jlast,Pinv,1263);
              Rcpp::Rcout << "\n found it! j " << j << " => i " << i;
#endif
              Pe [j] = FLIP (i) ;
              Nv [i] += Nv [j] ;
              Nv [j] = 0 ;
              Elen [j] = EMPTY ;
              j = Pinv [j] ;
              Pinv [jlast] = j ;
            }
            else
            {
              jlast = j ;
              j = Pinv [j] ;
            }
          }
#ifdef ENABLE_DEBUG
          CHECK_INDEX(i,Pinv,1279);
#endif
          wflg++ ;
          i = Pinv [i];
        }
      }
    }
    
#ifdef ENABLE_DEBUG
    Rcpp::Rcout << "\nDetect done";
#endif
    
    int p = pme1 ;
    int nleft = n - nel ;
    for (int pme = pme1 ; pme <= pme2 ; pme++)
    {
#ifdef ENABLE_DEBUG
      CHECK_INDEX(pme,Iw,1291);
#endif
      int i = Iw [pme] ;
#ifdef ENABLE_DEBUG
      CHECK_INDEX(i,Nv,1297);
#endif
      int nvi = -Nv [i] ;
#ifdef ENABLE_DEBUG
      Rcpp::Rcout << "\nRestore i " << i << " " << nvi;
#endif
      if (nvi > 0)
      {
        Nv[i] = nvi ;
        deg = Degree[i] + degme - nvi ;
        deg = deg < nleft - nvi ? deg : nleft - nvi;
#ifdef ENABLE_DEBUG
        CHECK_INDEX(deg,Head,1304);
#endif
        int inext = Head [deg] ;

        if (inext != EMPTY) {
#ifdef ENABLE_DEBUG
          CHECK_INDEX(inext,P,1308);
#endif
          P[inext] = i ;
        }
        Pinv[i] = inext ;
        P[i] = EMPTY ;
        Head [deg] = i ;
        mindeg = mindeg < deg ? mindeg : deg;
        Degree[i] = deg ;
        Iw[p++] = i ;
      }
    }
#ifdef ENABLE_DEBUG
    CHECK_INDEX(me,Nv,1320);
    Rcpp::Rcout << "\nRestore done";
    Rcpp::Rcout << "\n ME = " <<  me << " done";
#endif
    Nv[me] = nvpiv;
    Len[me] = p - pme1;
    if (Len[me] == 0)
    {
      Pe[me] = EMPTY;
      Tp[me] = 0;
    }
    if (elenme != 0)
    {
      pfree = p ;
    }

    info.f = nvpiv;
    info.r = degme + info.ndense;
    info.dmax = info.dmax > info.f + info.r ? info.dmax : info.f + info.r;
    info.lnzme = info.f * info.r + (info.f - 1) * info.f / 2;
    info.lnz += info.lnzme;
    info.ndiv += info.lnzme;
    info.s = info.f * info.r * info.r + info.r * (info.f - 1) * info.f + (info.f - 1) * info.f * (2 * info.f - 1) / 6;
    info.nms_lu += info.s;
    info.nms_ldl += (info.s + info.lnzme) / 2;
#ifdef ENABLE_DEBUG
    Rcpp::Rcout << "\nFinalise done nel " << nel << " n " << n;
    for (int pme = Pe [me] ; pme <= Pe [me] + Len [me] - 1 ; pme++)
    {
      Rcpp::Rcout << " " << Iw[pme];
    }
#endif
  }

  info.f = info.ndense;
  info.dmax = info.dmax > (double)info.ndense ? info.dmax : (double)info.ndense;
  info.lnzme = (info.f - 1) * info.f / 2;
  info.lnz += info.lnzme;
  info.ndiv += info.lnzme;
  info.s = (info.f - 1) * info.f * (2 * info.f - 1) / 6;
  info.nms_lu += info.s;
  info.nms_ldl += (info.s + info.lnzme) / 2;


  for (int i = 0 ; i < nn ; i++)Pe[i] = FLIP(Pe[i]);
  for (int i = 0 ; i < nn ; i++)Elen[i] = FLIP(Elen[i]);
  
  
  
  for (int i = 0 ; i < nn ; i++)
  {
    if (Nv[i] == 0)
    {
#ifdef ENABLE_DEBUG
      Rcpp::Rcout << "\n Path compression, i unordered " << i;
#endif
      int j = Pe [i] ;
      if (j == EMPTY){
#ifdef ENABLE_DEBUG
        CHECK_INDEX(j,Pe,1365);
        Rcpp::Rcout << "\n j: " << j << " is a dense variable";
#endif
        continue;
      }
#ifdef ENABLE_DEBUG
      CHECK_INDEX(j,Pe,1365);
#endif
      while (Nv [j] == 0)j = Pe[j];
      int e = j ;
      j = i ;
#ifdef ENABLE_DEBUG
      CHECK_INDEX(j,Nv,1371);
      Rcpp::Rcout << " j: " << j;
#endif
      while (Nv [j] == 0)
      {
        int jnext = Pe [j] ;
#ifdef ENABLE_DEBUG
        Rcpp::Rcout << " j: " << j << " jnext " << jnext;
#endif
        Pe [j] = e ;
        j = jnext ;
      }
    }
  }
  
  
  // POST ORDER
  // child = head 
  // sibling = Pinv
  // stack = P
  for (int j = 0; j < nn; j++)
  {
    Head[j] = EMPTY;
    Pinv[j] = EMPTY;
  }

  for (int j = nn - 1; j >= 0; j--)
  {
    if (Nv[j] > 0)
    {
      int parent = Pe[j];
      if (parent != EMPTY)
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(parent,P,1396);
#endif
        Pinv[j] = Head[parent];
        Head[parent] = j;
      }
    }
  }

  for (int i = 0; i < nn; i++)
  {
    if (Nv[i] > 0 && P[i] != EMPTY)
    {
      int fprev = EMPTY;
      int maxfrsize = EMPTY;
      int bigfprev = EMPTY;
      int bigf = EMPTY;
      for (int f = Head[i]; f != EMPTY; f = Pinv[f])
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(f,Elen,1415);
#endif
        int frsize = Elen[f];
        if (frsize >= maxfrsize)
        {
          maxfrsize = frsize;
          bigfprev = fprev;
          bigf = f;
        }
        fprev = f;
      }
#ifdef ENABLE_DEBUG
      CHECK_INDEX(bigf,Pinv,1427);
#endif
      int fnext = Pinv[bigf];
      if (fnext != EMPTY)
      {
        if (bigfprev == EMPTY)
        {
          Head[i] = fnext;
        }
        else
        {
          Pinv[bigfprev] = fnext;
        }
#ifdef ENABLE_DEBUG
        CHECK_INDEX(bigf,Pinv,1441);
        CHECK_INDEX(fprev,Pinv,1442);
#endif
        Pinv[bigf] = EMPTY;
        Pinv[fprev] = bigf;
      }
    }
  }

  for (int i = 0; i < nn; i++)Tp[i] = EMPTY;

  int k = 0;
  for (int i = 0; i < nn; i++)if (Pe[i] == EMPTY && Nv[i] > 0)k = AMD_post_tree(i, k, Head, Pinv, Tp, P);

  for (k = 0 ; k < nn ; k++)
  {
    Head [k] = EMPTY ;
    Pinv [k] = EMPTY ;
  }
  
  for (int e = 0 ; e < nn ; e++)
  {
    int k = Tp[e] ;
    if (k != EMPTY)Head[k] = e;
  }
  
#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nFinal reorder.\nTp:";
  for(const auto& i: Tp) Rcpp::Rcout << i << " ";
  Rcpp::Rcout << "\nHead:";
  for(const auto& i: Head) Rcpp::Rcout << i << " ";
#endif
  
  
  nel = 0 ;
  for (k = 0 ; k < nn ; k++)
  {
    int e = Head [k] ;
    if (e == EMPTY) break ;
    Pinv [e] = nel ;
    nel += Nv [e] ;
  }

#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nSet up Pinv. nel: " << nel << " Pe:\n";
  for(const auto& i: Pe) Rcpp::Rcout << i << " ";
#endif
  
  for (int i = 0 ; i < nn ; i++)
  {
    if (Nv[i] == 0)
    {
      int e = Pe [i] ;
      if (e != EMPTY)
      {
#ifdef ENABLE_DEBUG
        CHECK_INDEX(e,Pinv,1482);
#endif
        Pinv [i] = Pinv [e] ;
        Pinv [e]++ ;
      }
      else
      {
        Pinv [i] = nel++ ;
      }
    }
  }
  
#ifdef ENABLE_DEBUG
  Rcpp::Rcout << "\nFinal loop. Pinv: ";
  for(const auto& i: Pinv) Rcpp::Rcout << i << " ";
#endif

  for (int i = 0 ; i < nn ; i++)
  {
    int k = Pinv[i];
#ifdef ENABLE_DEBUG
    CHECK_INDEX(k,P,1498);
#endif
    P[k] = i;
  }

}