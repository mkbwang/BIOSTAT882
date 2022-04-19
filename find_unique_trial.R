# find unique trial
library(Rcpp)
library(RcppArmadillo)

example <- sample.int(10, 60, replace=TRUE)
example <- matrix(example, nrow=15, ncol=4)

sourceCpp(code = "
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//[[Rcpp::export]]
Rcpp::List get_unique(arma::mat& mymatrix){
          int row_count = mymatrix.n_rows;
          int col_count = mymatrix.n_cols;
          arma::vec uniq_count = zeros(col_count);
          arma::mat uniq_vals, position_vals;
          uniq_vals.zeros(size(mymatrix));
          position_vals.zeros(size(mymatrix));
          for (int i=0;i < col_count;i++){
            arma::vec selected_col = mymatrix.col(i);
            arma::vec all_vals = unique(selected_col);
            int counts = all_vals.n_elem;
            uniq_count(i) = counts;
            uniq_vals(span(0, counts-1), i) = all_vals;
            arma::vec position_vec = zeros(row_count);
            for (int j=0; j<counts; j++){
              position_vec.elem(find(selected_col == all_vals(j))).fill(j);
              position_vals.col(i) = position_vec;
            }
          }
          Rcpp::List  output = Rcpp::List::create(
            Named(\"UniqVal\") = uniq_vals,
            Named(\"Uniqcount\") = uniq_count,
            Named(\"Positions\") = position_vals
          );
          return output;
}")

output <- get_unique(example)
