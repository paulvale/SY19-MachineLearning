rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/analysis_2.rData")

# KNN
# Evolution en fonction du nombre de composantes prises en comptes
knn.acp.error.mean.k5 <- colMeans(knn.acp.error.k5)
knn.acp.error.mean <- colMeans(knn.acp.error)
min <- min(c(knn.acp.error.mean.k5, knn.acp.error.mean))
max <- max(c(knn.acp.error.mean.k5, knn.acp.error.mean))
plot(knn.acp.error.mean.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.acp.error.mean, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

knn.forward.error.mean.k5 <- colMeans(knn.forward.error.k5)
knn.forward.error.mean <- colMeans(knn.forward.error)
min <- min(c(knn.forward.error.mean.k5, knn.forward.error.mean))
max <- max(c(knn.forward.error.mean.k5, knn.forward.error.mean))
plot(knn.forward.error.mean.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.forward.error.mean, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# Evolution en fonction du nombre K de voisins pris en compte
knn.acp.component <- which(knn.acp.error == min(knn.acp.error), arr.ind = TRUE)[2]
knn.acp.component.k5 <- which(knn.acp.error.k5 == min(knn.acp.error.k5), arr.ind = TRUE)[2]
min <- min(c(knn.acp.error[,knn.acp.component], knn.acp.error.k5[,knn.acp.component.k5]))
max <- max(c(knn.acp.error[,knn.acp.component], knn.acp.error.k5[,knn.acp.component.k5]))
plot(knn.acp.error.k5[,knn.acp.component.k5],type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.acp.error[,knn.acp.component], type="l",xlab="number of neighbours", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

knn.forward.component <- which(knn.forward.error == min(knn.forward.error), arr.ind = TRUE)[2]
knn.forward.component.k5 <- which(knn.forward.error.k5 == min(knn.forward.error.k5), arr.ind = TRUE)[2]
min <- min(c(knn.forward.error[,knn.forward.component], knn.forward.error.k5[,knn.forward.component.k5]))
max <- max(c(knn.forward.error[,knn.forward.component], knn.forward.error.k5[,knn.forward.component.k5]))
plot(knn.forward.error.k5[,knn.forward.component.k5],type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.forward.error[,knn.forward.component], type="l",xlab="number of neighbours",ylim=c(min,max), ylab="MSE", main="Evolution of error with Forward Data")

min <- min(c(knn.lda.error, knn.lda.error.k5))
max <- max(c(knn.lda.error, knn.lda.error.k5))
plot(knn.lda.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.lda.error, type="l", xlab="number of neighbors", ylab="MSE",ylim=c(min,max), main="Evolution of error with FDA Data")


# RF
# Evolution en fonction du nombre de composantes prises en comptes
rf.acp.error.mean.k5 <- colMeans(rf.acp.error.k5)
rf.acp.error.mean <- colMeans(rf.acp.error)
min <- min(c(rf.acp.error.mean.k5, rf.acp.error.mean))
max <- max(c(rf.acp.error.mean.k5, rf.acp.error.mean))
plot(rf.acp.error.mean.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.acp.error.mean, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

rf.forward.error.mean.k5 <- colMeans(rf.forward.error.k5)
rf.forward.error.mean <- colMeans(rf.forward.error)
min <- min(c(rf.forward.error.mean.k5, rf.forward.error.mean))
max <- max(c(rf.forward.error.mean.k5, rf.forward.error.mean))
plot(rf.forward.error.mean.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.forward.error.mean, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# Evolution en fonction du nombre K d'arbres pris en compte
rf.acp.component <- which(rf.acp.error == min(rf.acp.error), arr.ind = TRUE)[2]
rf.acp.component.k5 <- which(rf.acp.error.k5 == min(rf.acp.error.k5), arr.ind = TRUE)[2]
min <- min(c(rf.acp.error[,rf.acp.component], rf.acp.error.k5[,rf.acp.component.k5]))
max <- max(c(rf.acp.error[,rf.acp.component], rf.acp.error.k5[,rf.acp.component.k5]))
plot(rf.acp.error.k5[,rf.acp.component.k5],type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.acp.error[,rf.acp.component], type="l",xlab="number of trees", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

rf.forward.component <- which(rf.forward.error == min(rf.forward.error), arr.ind = TRUE)[2]
rf.forward.component.k5 <- which(rf.forward.error.k5 == min(rf.forward.error.k5), arr.ind = TRUE)[2]
min <- min(c(rf.forward.error[,rf.forward.component], rf.forward.error.k5[,rf.forward.component.k5]))
max <- max(c(rf.forward.error[,rf.forward.component], rf.forward.error.k5[,rf.forward.component.k5]))
plot(rf.forward.error.k5[,rf.forward.component.k5],type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.forward.error[,rf.forward.component], type="l",xlab="number of trees",ylim=c(min,max), ylab="MSE", main="Evolution of error with Forward Data")

min <- min(c(rf.lda.error, rf.lda.error.k5))
max <- max(c(rf.lda.error, rf.lda.error.k5))
plot(rf.lda.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.lda.error, type="l", xlab="number of neighbors", ylab="MSE",ylim=c(min,max), main="Evolution of error with FDA Data")

# LDA
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(lda.acp.error.k5, lda.acp.error))
max <- max(c(lda.acp.error.k5, lda.acp.error))
plot(lda.acp.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(lda.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

min <- min(c(lda.forward.error.k5, lda.forward.error))
max <- max(c(lda.forward.error.k5, lda.forward.error))
plot(lda.forward.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(lda.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# QDA
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(qda.acp.error.k5, qda.acp.error))
max <- max(c(qda.acp.error.k5, qda.acp.error))
plot(qda.acp.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(qda.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

min <- min(c(qda.forward.error.k5, qda.forward.error))
max <- max(c(qda.forward.error.k5, qda.forward.error))
plot(qda.forward.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(qda.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# LogReg
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(logReg.acp.error.k5, logReg.acp.error))
max <- max(c(logReg.acp.error.k5, logReg.acp.error))
plot(logReg.acp.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(logReg.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

min <- min(c(logReg.forward.error.k5, logReg.forward.error))
max <- max(c(logReg.forward.error.k5, logReg.forward.error))
plot(logReg.forward.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(logReg.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# NB
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(nb.acp.error.k5, nb.acp.error))
max <- max(c(nb.acp.error.k5, nb.acp.error))
plot(nb.acp.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(nb.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

min <- min(c(nb.forward.error.k5, nb.forward.error))
max <- max(c(nb.forward.error.k5, nb.forward.error))
plot(nb.forward.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(nb.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# SVM
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(svm.acp.error.k5, svm.acp.error))
max <- max(c(svm.acp.error.k5, svm.acp.error))
plot(svm.acp.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

min <- min(c(svm.forward.error.k5, svm.forward.error))
max <- max(c(svm.forward.error.k5, svm.forward.error))
plot(svm.forward.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# SVM Tune
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(svm.tune.acp.error.k5, svm.tune.acp.error))
max <- max(c(svm.tune.acp.error.k5, svm.tune.acp.error))
plot(svm.tune.acp.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.tune.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

min <- min(c(svm.tune.forward.error.k5, svm.tune.forward.error))
max <- max(c(svm.tune.forward.error.k5, svm.tune.forward.error))
plot(svm.tune.forward.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.tune.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")

# Tree
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(tree.acp.error.k5, tree.acp.error))
max <- max(c(tree.acp.error.k5, tree.acp.error))
plot(tree.acp.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(tree.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data")

min <- min(c(tree.forward.error.k5, tree.forward.error))
max <- max(c(tree.forward.error.k5, tree.forward.error))
plot(tree.forward.error.k5,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(tree.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data")
