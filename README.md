# SparseChol

The package originally started as just a header library to implement the sparse, symmetric matrix LDL decomposition algorithm described by Timothy Davis (https://fossies.org/linux/SuiteSparse/LDL/Doc/ldl_userguide.pdf), but now also includes a growing sparse matrix class that is compataible with the Eigen library. 
Sparse matrices are possible in Eigen, but hard to work with and lacks LDL decomposition, so this package is meant to provide a simple interface with relatively 
efficient functionality. However, these functions will be optimised in future versions. We use the sparse functionality in the [glmmrBase](https://github.com/samuel-watson/glmmrBase/tree/master) and associated packages to store and use, for example, sparse design and covariance matrices.

The header file in /inst/include/sparse.h imports the classes for the library and can be included in other projects using Rcpp in R. 
The R function `sparse_chol()` provides an R interface using compressed column form as per the `CsparseMatrix` in the Matrix package.

## C++ API
### Sparse matrix class
The sparse matrix class `sparse` can be initialised in several ways including from an `Rcpp::NumericMatrix` and an `Eigen::MatrixXd`. It is effectively a container for 
a sparse matrix representation of the matrix. It can also just be initialised by specifying the rows and columns and later inserting elements. For example,
```
#include <sparse.h>
#include <RcppEigen.h> //assuming this is for an R package

// to create a 4x3 matrix, stored in row major order
sparse A(4,3,true);
A.insert(0,0,1);
A.insert(0,2,2);
A.insert(1,1,1);
A.insert(2,1,3);
A.insert(3,0,2);
A.insert(3,2,3);

// generate an Eigen matrix
MatrixXd B(3,3);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      B(i,j) = i + 1 + j*3;
    }
  }

MatrixXd AB = A * B; // multiply sparse by sparse

// generate a transpose of the matrix
sparse At = A;
At.transpose();
// and multiply
At *= A;
```

### Sparse LDL decomposition
The `SparseChol` class is responsible for the sparse LDL decomposition. We initialise the class using a `sparse` matrix.
```
#include <sparse.h>

SparseChol sp(mat); // where mat is of class sparse
int d = sp.ldl_numeric(); // run decomposition, returns dimension of matrix if success
//to retrieve the Cholesky decomposion LD^0.5
sparse LD = sp.LD();
// or we can return L and D separately

// to solve the linear system Ax = v where v is a std::vector<double> we pass a reference to the first element of the vector
// the system is solved in place so that v will contain the result
spchol.ldl_lsolve(&v[0]); // Lx = v
spchol.ldl_d2solve(&v[0]); // Dx = v

```

## R functions
We can use the LDL decomposition in R with the associated functions provided for demonstration. 

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


