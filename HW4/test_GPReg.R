library(Rcpp)
library(dplyr)
library(matrixcalc)

rm(list=ls())
sourceCpp('HW4/GaussianProcessReg.cpp')

train_y = read.csv('mnist_100_train_Y.csv') %>% as.matrix() 

# first round the values to two digits
train_x = read.csv('mnist_100_train_X.csv') %>% as.matrix()
train_x = matrix(as.numeric(train_x), ncol=ncol(train_x))
rounded_train_x = round(train_x, 2)

# then I decide to focus on those columns that have at least 10 unique values
unique_count <- rep(0, ncol(rounded_train_x))
for (i in 1:length(unique_count)){
  uniq_vals <- unique(rounded_train_x[,i])
  unique_count[i] <- length(uniq_vals)
}
selected_indices <- which(unique_count >= 10)
subset_train_x <- rounded_train_x[,selected_indices]

# I cannot get this function to run :(
result = Bayes_shrinkage_GP_reg(y=train_y, X=subset_train_x,
                                prior_delta_prob=0.5,
                                A=1,
                                mcmc_sample = 1000,
                                burnin = 5000,
                                thinning = 20,
                                verbose = 200,
                                save_profile = 10)

save(result, file="HW4/model_fitting.RData")


