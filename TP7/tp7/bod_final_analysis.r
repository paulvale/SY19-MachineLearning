rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_10_forward.rData")

# KNN
# Evolution en fonction du nombre de composantes prises en comptes
knn.acp.error.mean <- colMeans(knn.acp.error)
plot(knn.acp.error.mean, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / KNN")

knn.forward.error.mean <- colMeans(knn.forward.error)
plot(knn.forward.error.mean, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / KNN")

knn.f.error.mean <- colMeans(knn.f.error)
plot(knn.f.error.mean, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / KNN")

# Evolution en fonction du nombre K de voisins pris en compte
knn.acp.error.mean <- rowMeans(knn.acp.error)
plot(knn.acp.error.mean, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with ACP Data / KNN")

knn.forward.error.mean <- rowMeans(knn.forward.error)
plot(knn.forward.error.mean, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with Forward ACP / KNN")

knn.f.error.mean <- rowMeans(knn.f.error)
plot(knn.f.error.mean, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with Forward Data / KNN")

plot(knn.lda.error, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with FDA Data / KNN")


# RF
# Evolution en fonction du nombre de composantes prises en comptes
rf.acp.error.mean <- colMeans(rf.acp.error)
plot(rf.acp.error.mean, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / RF")

rf.forward.error.mean <- colMeans(rf.forward.error)
plot(rf.forward.error.mean, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / RF")

rf.f.error.mean <- colMeans(rf.f.error)
plot(rf.f.error.mean, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / RF")

# Evolution en fonction du nombre K d'arbres pris en compte
rf.acp.error.mean <- rowMeans(rf.acp.error)
plot(rf.acp.error.mean, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with ACP Data / RF")

rf.forward.error.mean <- rowMeans(rf.forward.error)
plot(rf.forward.error.mean, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with Forward ACP / RF")

rf.f.error.mean <- rowMeans(rf.f.error)
plot(rf.f.error.mean, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with Forward Data / RF")

plot(rf.lda.error, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with FDA Data / RF")

# LDA
# Evolution en fonction du nombre de composantes prises en comptes
plot(lda.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / LDA")

plot(lda.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / LDA")

plot(lda.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / LDA")

# QDA
# Evolution en fonction du nombre de composantes prises en comptes
plot(qda.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / QDA")

plot(qda.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / QDA")

plot(qda.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / QDA")

# LogReg
# Evolution en fonction du nombre de composantes prises en comptes
plot(logReg.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / LogReg")

plot(logReg.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / LogReg")

plot(logReg.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / LogReg")

# NB
# Evolution en fonction du nombre de composantes prises en comptes
plot(nb.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / NB")

plot(nb.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / NB")

plot(nb.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / NB")

# SVM
# Evolution en fonction du nombre de composantes prises en comptes
plot(svm.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / SVM")

plot(svm.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / SVM")

plot(svm.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / SVM")

# SVM Tune
# Evolution en fonction du nombre de composantes prises en comptes
plot(svm.tune.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data Tune / SVM")

plot(svm.tune.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP Tune / SVM")

plot(svm.tune.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data Tune / SVM ")

# Tree
# Evolution en fonction du nombre de composantes prises en comptes
plot(tree.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / Tree")

plot(tree.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / Tree")

plot(tree.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / Tree")
