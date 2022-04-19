library(Rcpp)
library(dplyr)
library(matrixcalc)

rm(list=ls())
sourceCpp('GaussianProcessReg.cpp')

train_y = read.csv('mnist_100_train_Y.csv') %>% as.matrix() 

train_x = read.csv('mnist_100_train_X.csv') %>% as.matrix()
train_x = matrix(as.numeric(train_x), ncol=ncol(train_x))
rounded_train_x = round(train_x, 2)
unique_count <- rep(0, ncol(rounded_train_x))
for (i in 1:length(unique_count)){
  uniq_vals <- unique(rounded_train_x[,i])
  unique_count[i] <- length(uniq_vals)
}

selected_indices <- which(unique_count >= 10)

subset_train_x <- rounded_train_x[,selected_indices]

# # first look at training data
# singularities_train <- rep(0, ncol(subset_train_x))
# min_eigen_value_train <- rep(0,ncol(subset_train_x))
# for (i in 1:ncol(subset_train_x)){
#   x_vec = subset_train_x[, i]
#   count_x <- table(x_vec)
#   uniq_x <- as.numeric(names(count_x))
#   distmat <- as.matrix(dist(uniq_x))
#   kernel_mat <- exp( - distmat / 2)
#   eigen_res <- eigen(kernel_mat)
#   min_eigen_value_train[i] <- min(eigen_res$values)
#   if (is.singular.matrix(kernel_mat,tol = 1e-323)){
#     singularities_train[i] <- TRUE
#   } else{
#     singularities_train[i] <- FALSE
#   }
# }
# 
# print(singularities_train)
# print(which(min_eigen_value_train<0))
# 


# test_y = read.csv('HW4/mnist_1000_test_Y.csv') %>% as.matrix()
# test_x = read.csv('HW4/mnist_1000_test_X.csv') %>% as.matrix()

# avg_intensity <- c()
# avg_nonzeros <- c()
# for (i in 1:ncol(train_x)){
#   avg_intensity <- c(avg_intensity, mean(train_x[, i]))
#   avg_nonzeros <- c(avg_nonzeros, mean(train_x[, i] != 0))
# }
# usable_cols <- which(avg_nonzeros > 0.8)
# train_x_subset <- train_x[, usable_cols]


result = Bayes_shrinkage_GP_reg(y=train_y, X=subset_train_x,
                                prior_delta_prob=0.5,
                                A=1,
                                mcmc_sample = 1000,
                                burnin = 5000,
                                thinning = 20,
                                verbose = 200,
                                save_profile = 10)

save(result, file="HW4/model_fitting.RData")


# 
# train_x = testcase$X
# sample_f_x = result$mcmc$f_val[, , 3]
# sample_sigma2 = result$mcmc$sigma_sq[3]
# 
# test_x_vec = runif(10000)
# test_x = matrix(test_x_vec, nrow = 1000)
# 
# predicted_values = GPpredict(train_x, sample_f_x, test_x, sample_sigma2)
# 
# vec1 = train_x[, 1]
# vec2 = test_x[, 2]
# 
# dist1 = as.matrix(dist(vec1))
# kernel11 = exp(-dist1 / 1000)
# kernel12 = matrix(0, nrow = 100, ncol = 1000)
# for (i in 1:100) {
#   for (j in 1:1000) {
#     kernel12[i, j] = exp(-abs(vec1[i] - vec2[j]) / 1000)
#   }
# }
# dist2 = as.matrix(dist(vec2))
# kernel22 = exp(-dist2 / 1000)
# 
# inv_kernel11 = solve(kernel11)
# varmat = kernel22 -  t(kernel12) %*% inv_kernel11 %*% kernel12
# var_result = chol(varmat)
