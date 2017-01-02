rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/analysis.rData")

# KNN
# Evolution en fonction du nombre de composantes prises en comptes
knn.acp.error.mean <- colMeans(knn.acp.error)
plot(knn.acp.error.mean, type="l", xlab="number of componants", ylab="MSE", main="Evolution of error with ACP Data")

knn.forward.error.mean <- colMeans(knn.forward.error)
plot(knn.forward.error.mean, type="l", xlab="number of componants", ylab="MSE", main="Evolution of error with Forward Data")

# Evolution en fonction du nombre K de voisins pris en compte
knn.acp.component <- which(knn.acp.error == min(knn.acp.error), arr.ind = TRUE)[2]
plot(knn.acp.error[,knn.acp.component], type="l",xlab="number of neighbors", ylab="MSE", main="Evolution of error with ACP Data")

knn.forward.component <- which(knn.acp.error == min(knn.acp.error), arr.ind = TRUE)[2]
plot(knn.forward.error[,knn.forward.component], type="l",xlab="number of neighbors", ylab="MSE", main="Evolution of error with Forward Data")

plot(knn.lda.error, type="l", xlab="number of neighbors", ylab="MSE", main="Evolution of error with FDA Data")
