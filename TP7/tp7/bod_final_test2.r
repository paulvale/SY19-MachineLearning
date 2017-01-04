# LOAD PACKAGES 
# =============
rm(list=ls())
library(MASS)
library(leaps)
library(glmnet)
library(class)
library(e1071)
library(tree)
library(randomForest)
library("neuralnet")
library(nnet)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')


# LOAD DATA
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_5.rData")

qda.acp.error.k5 <- qda.acp.error
qda.lda.error.k5 <- qda.lda.error
qda.forward.error.k5 <- qda.forward.error
lda.acp.error.k5 <- lda.acp.error
lda.lda.error.k5 <- lda.lda.error
lda.forward.error.k5 <- lda.forward.error
knn.acp.error.k5 <- knn.acp.error
knn.lda.error.k5 <- knn.lda.error
knn.forward.error.k5 <- knn.forward.error
logReg.acp.error.k5 <- logReg.acp.error
logReg.lda.error.k5 <- logReg.lda.error
logReg.forward.error.k5 <- logReg.forward.error
nb.acp.error.k5 <- nb.acp.error
nb.lda.error.k5 <- nb.lda.error
nb.forward.error.k5 <- nb.forward.error
svm.acp.error.k5 <- svm.acp.error
svm.lda.error.k5 <- svm.lda.error
svm.forward.error.k5 <- svm.forward.error
svm.tune.acp.error.k5 <- svm.tune.acp.error
svm.tune.lda.error.k5 <- svm.tune.lda.error
svm.tune.forward.error.k5 <- svm.tune.forward.error
tree.acp.error.k5 <- tree.acp.error
tree.lda.error.k5 <- tree.lda.error
tree.forward.error.k5 <- tree.forward.error
rf.acp.error.k5 <- rf.acp.error
rf.lda.error.k5 <- rf.lda.error
rf.forward.error.k5 <- rf.forward.error
nn.acp.error.k5 <- nn.acp.error
nn.lda.error.k5 <- nn.lda.error
nn.forward.error.k5 <- nn.forward.error

load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_10.rData")