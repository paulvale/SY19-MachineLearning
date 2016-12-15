# LOAD PACKAGES 
# =============
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
X.dim <- dim(X)

for( i in 1:X.dim[1]){
  I<-matrix(X[i,],60,70)
  I1 <- apply(I, 1, rev)
  image(t(I1),col=gray(0:255 / 255))
}


acp.X <- prcomp(X)
plot(acp.X)
print(cumsum(100 * acp.X$sdev^2 / sum(acp.X$sdev^2)))

