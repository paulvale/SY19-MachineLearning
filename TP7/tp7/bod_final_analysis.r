rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_final.rData")

# KNN
# Evolution en fonction du nombre de composantes prises en comptes
knn.acp.error.mean <- colMeans(knn.acp.error)
knn.forward.error.mean <- colMeans(knn.forward.error)
knn.f.error.mean <- colMeans(knn.f.error)

min <- min(knn.acp.error.mean, knn.forward.error.mean, knn.f.error.mean)
max <- max(knn.acp.error.mean, knn.forward.error.mean, knn.f.error.mean)

plot(knn.acp.error.mean, type="l", col="blue",xlab="number of principal components", ylab="MSE", main="Evolution of error", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      "),lty=1, lwd=2.5,col=c("blue","red","green"))
par(new=TRUE)
plot(knn.forward.error.mean, type="l",col="red", xlab="", ylab="", main="",ylim=c(min,max))
par(new=TRUE)
plot(knn.f.error.mean, type="l", col="green",xlab="", ylab="", main="", ylim=c(min,max))
par(new=FALSE)

# Evolution en fonction du nombre K de voisins pris en compte
knn.acp.error.mean <- rowMeans(knn.acp.error)
knn.forward.error.mean <- rowMeans(knn.forward.error)
knn.f.error.mean <- rowMeans(knn.f.error)

min <- min(knn.acp.error.mean, knn.forward.error.mean, knn.f.error.mean, knn.lda.error)
max <- max(knn.acp.error.mean, knn.forward.error.mean, knn.f.error.mean, knn.lda.error)

plot(knn.acp.error.mean, type="l",col="blue", xlab="number of neighbors", ylab="MSE", main="Evolution of error", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      ","FDA"),lty=1, lwd=2.5,col=c("blue","red","green","black"))
par(new=TRUE)
plot(knn.forward.error.mean, type="l", col="red",xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(knn.f.error.mean, type="l", col="green",xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(knn.lda.error, type="l", xlab="", ylab="", main="", ylim=c(min,max))
par(new=FALSE)

min <- min(knn.acp.error[20,], knn.forward.error[12,], knn.f.error[15,])
max <- max(knn.acp.error[20,], knn.forward.error[12,], knn.f.error[15,])
plot(knn.acp.error[20,], type="l",col="blue", xlab="number of neighbors", ylab="MSE", main="Evolution of error", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      ","FDA"),lty=1, lwd=2.5,col=c("blue","red","green","black"))
par(new=TRUE)
plot(knn.forward.error[12,], type="l", col="red",xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(knn.f.error[15,], type="l", col="green",xlab="", ylab="", main="", ylim=c(min,max))
par(new=FALSE)

# RF
# Evolution en fonction du nombre de composantes prises en comptes
rf.acp.error.mean <- colMeans(rf.acp.error)
rf.forward.error.mean <- colMeans(rf.forward.error)
rf.f.error.mean <- colMeans(rf.f.error)

min <- min(rf.acp.error.mean, rf.forward.error.mean, rf.f.error.mean)
max <- max(rf.acp.error.mean, rf.forward.error.mean, rf.f.error.mean)

plot(rf.acp.error.mean, type="l", col="blue",xlab="number of principal components", ylab="MSE", main="Evolution of error", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      "),lty=1, lwd=2.5,col=c("blue","red","green"))
par(new=TRUE)
plot(rf.forward.error.mean, type="l", col="red",xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(rf.f.error.mean, type="l",col="green", xlab="", ylab="", main="", ylim=c(min,max))
par(new=FALSE)

# Evolution en fonction du nombre K d'arbres pris en compte
rf.acp.error.mean <- rowMeans(rf.acp.error)
rf.forward.error.mean <- rowMeans(rf.forward.error)
rf.f.error.mean <- rowMeans(rf.f.error)

min <- min(rf.acp.error.mean, rf.forward.error.mean, rf.f.error.mean, rf.lda.error)
max <- max(rf.acp.error.mean, rf.forward.error.mean, rf.f.error.mean, rf.lda.error)

plot(vectorTree,rf.acp.error.mean, type="l", col="blue",xlab="number of trees", ylab="MSE", main="Evolution of error", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      ","FDA"),lty=1, lwd=2.5,col=c("blue","red","green","black"))
par(new=TRUE)
plot(vectorTree,rf.forward.error.mean, type="l", col="red",xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(vectorTree,rf.f.error.mean, type="l",col="green", xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(vectorTree,rf.lda.error, type="l", xlab="", ylab="", main="", ylim=c(min,max))
par(new=FALSE)

# LDA
# Evolution en fonction du nombre de composantes prises en comptes
#plot(lda.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / LDA")
#plot(lda.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / LDA")
#plot(lda.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / LDA")

# QDA
# Evolution en fonction du nombre de composantes prises en comptes
#plot(qda.acp.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with ACP Data / QDA")
#plot(qda.forward.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward ACP / QDA")
#plot(qda.f.error, type="l", xlab="number of principal components", ylab="MSE", main="Evolution of error with Forward Data / QDA")

# LogReg
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(logReg.acp.error, logReg.forward.error, logReg.f.error)
max <- max(logReg.acp.error, logReg.forward.error, logReg.f.error)

plot(logReg.acp.error, type="l",col="blue", xlab="number of principal components", ylab="MSE", main="Evolution of error", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      "),lty=1, lwd=2.5,col=c("blue","red","green"))
par(new=TRUE)
plot(logReg.forward.error, type="l",col="red", xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(logReg.f.error, type="l",col="green", xlab="", ylab="", main="", ylim=c(min,max))
par(new=FALSE)

# NB
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(nb.acp.error, nb.forward.error, nb.f.error)
max <- max(nb.acp.error, nb.forward.error, nb.f.error)

plot(nb.acp.error, type="l", col="blue",xlab="number of principal components", ylab="MSE", main="Evolution of error", ylim=c(min,max))
par(new=TRUE)
plot(nb.forward.error, type="l",col="red", xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(nb.f.error, type="l", col="green",xlab="", ylab="", main="", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      "),lty=1, lwd=2.5,col=c("blue","red","green"))
par(new=FALSE)

# SVM
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(svm.acp.error, svm.forward.error, svm.f.error)
max <- max(svm.acp.error, svm.forward.error, svm.f.error)

plot(svm.acp.error, type="l", col="blue", xlab="number of principal components", ylab="MSE", main="Evolution of error", ylim=c(min,max))
par(new=TRUE)
plot(svm.forward.error, type="l", col="red",xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(svm.f.error, type="l", col="green",xlab="", ylab="", main="", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      "),lty=1, lwd=2.5,col=c("blue","red","green"))
par(new=FALSE)

# SVM Tune
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(svm.tune.acp.error, svm.tune.forward.error, svm.tune.f.error)
max <- max(svm.tune.acp.error, svm.tune.forward.error, svm.tune.f.error)

plot(svm.tune.acp.error, type="l", col="blue", xlab="number of principal components", ylab="MSE", main="Evolution of error", ylim=c(min,max))
par(new=TRUE)
plot(svm.tune.forward.error, type="l",col="red", xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(svm.tune.f.error, type="l", col="green", xlab="", ylab="", main=" ", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      "),lty=1, lwd=2.5,col=c("blue","red","green"))
par(new=FALSE)

# Tree
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(tree.acp.error, tree.forward.error, tree.f.error)
max <- max(tree.acp.error, tree.forward.error, tree.f.error)

plot(tree.acp.error, type="l",col="blue", xlab="number of principal components", ylab="MSE", main="Evolution of error", ylim=c(min,max))
legend('topright',c("ACP","Fwd","Fwd ACP      "),lty=1, lwd=2.5,col=c("blue","red","green"))
par(new=TRUE)
plot(tree.forward.error, type="l",col="red", xlab="", ylab="", main="", ylim=c(min,max))
par(new=TRUE)
plot(tree.f.error, type="l", col="green", xlab="", ylab="", main="", ylim=c(min,max))
par(new=FALSE)