# SparseChol

A c++ implementation of the approach to sparse, symmetric matrix LDL decomposition described by Timothy Davis (https://fossies.org/linux/SuiteSparse/LDL/Doc/ldl_userguide.pdf).
The header file in /inst/include/sparsechol.h defines the class SparseChol, which can be implemented in other c++ applications for R with Rcpp. The R function
`sparse_chol()` provides an R interface using compressed column form as per the `CsparseMatrix` in the Matrix package.

## An example in R
```
> n <- 10
> Ap  <- c(0, 1, 2, 3, 4, 6, 7, 9, 11, 15, ANZ)
> Ai <- c(1, 2, 3, 4, 2,5, 6, 5,7, 5,8, 1,5,8,9, 2,5,7,10)
> Ax = c(1.7, 1., 1.5, 1.1, .02,2.6, 1.2, .16,1.3, .09,1.6,
           .13,.52,.11,1.4, .01,.53,.56,3.1)
> Matrix::sparseMatrix(i = Ai, p=Ap, x=Ax, symmetric = TRUE)
10 x 10 sparse Matrix of class "dsCMatrix"
                                                    
 [1,] 1.70 .    .   .   .    .   .    .    0.13 .   
 [2,] .    1.00 .   .   0.02 .   .    .    .    0.01
 [3,] .    .    1.5 .   .    .   .    .    .    .   
 [4,] .    .    .   1.1 .    .   .    .    .    .   
 [5,] .    0.02 .   .   2.60 .   0.16 0.09 0.52 0.53
 [6,] .    .    .   .   .    1.2 .    .    .    .   
 [7,] .    .    .   .   0.16 .   1.30 .    .    0.56
 [8,] .    .    .   .   0.09 .   .    1.60 0.11 .   
 [9,] 0.13 .    .   .   0.52 .   .    0.11 1.40 .   
[10,] .    0.01 .   .   0.53 .   0.56 .    .    3.10
           
> out <-sparse_chol(n,Ap,Ai,Ax)
> sparse_L(out)
10 x 10 sparse Matrix of class "dtCMatrix"
                                                                             
 [1,] 1.00000000 .    . . .          .  .            .           .          .
 [2,] .          1.00 . . .          .  .            .           .          .
 [3,] .          .    1 . .          .  .            .           .          .
 [4,] .          .    . 1 .          .  .            .           .          .
 [5,] .          0.02 . . 1.00000000 .  .            .           .          .
 [6,] .          .    . . .          1  .            .           .          .
 [7,] .          .    . . 0.06154793 .  1.000000000  .           .          .
 [8,] .          .    . . 0.03462071 . -0.004293535  1.00000000  .          .
 [9,] 0.07647059 .    . . 0.20003077 . -0.024807089  0.05752527  1.00000000 .
[10,] .          0.01 . . 0.20380058 .  0.408782664 -0.01006831 -0.07185228 1
> sparse_D(out)
10 x 10 diagonal matrix of class "ddiMatrix"
      [,1] [,2] [,3] [,4] [,5]   [,6] [,7]     [,8]    [,9]     [,10]   
 [1,]  1.7    .    .    .      .    .        .       .        .        .
 [2,]    .    1    .    .      .    .        .       .        .        .
 [3,]    .    .  1.5    .      .    .        .       .        .        .
 [4,]    .    .    .  1.1      .    .        .       .        .        .
 [5,]    .    .    .    . 2.5996    .        .       .        .        .
 [6,]    .    .    .    .      .  1.2        .       .        .        .
 [7,]    .    .    .    .      .    . 1.290152       .        .        .
 [8,]    .    .    .    .      .    .        . 1.59686        .        .
 [9,]    .    .    .    .      .    .        .       . 1.279965        .
[10,]    .    .    .    .      .    .        .       .        . 2.769568
```

## An example in C++
```
#include <sparsechol.h>
main(){
  sparse mat;
  mat.n = 19;
  mat.Ap = {0, 1, 2, 3, 4, 6, 7, 9, 11, 15, ANZ};
  mat.Ai = {0, 1, 2, 3, 1,4, 5, 4,6, 4,7, 0,4,7,8, 1,4,6,9 };
  mat.Ax = {1.7, 1., 1.5, 1.1, .02,2.6, 1.2, .16,1.3, .09,1.6,
            .13,.52,.11,1.4, .01,.53,.56,3.1};
  SparseChol chol(&mat);
  int d = chol.ldl_numeric();
  //print off-diagonal values of matrix L, for example
   for (auto& k : chol.L->Ax)
     cout << k << " ";
}
```

## Solvers
Also included in the `SparseChol` class are LDL solvers for $Lx=y$ and $Dx=y$
