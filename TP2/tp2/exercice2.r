# qu'est ce que ca veut dire le 95% 
# et pour un nouveau cas qu'est ce que ca veut dire
# diff entre l'intervalle de confiance et de prediction
# on tire une fois les valeurs puis on y touche plus 
# les x sont supposes fixes
# donne des valeurs aux coeffs, b1 , B2 , ...

# on tire des Yi pour ces donnees la 
# pour avoir un nouveau jeu de donnees, on tire de nouveaux Yi mais pas de Xi

# Yi suit une loi normale(Bo+B1Xi +B2X2, sigma^2)
# et ca on peut le repeter plusieurs fois
cat("Verification par simulation que les intervalles de confiance a 95% sur les parametres Beta_j \n contiennent bien la vraie valeur des parametres dans environ 95% des cas")
cat("\n==========\n\n")
n<-20
x1<-rnorm(n,0,1)
x2<-rnorm(n,0,1)
X<-cbind(rep(1,n),x1,x2)
beta<-c(1,2,3)
Ey<- X %*% beta
sig<-0.5
N<-5000 # nombre d'exemples d'apprentissage
I<-matrix(0,N,3)
for(i in 1:N){
  y<-Ey+rnorm(n,0,sig) 
  reg<-lm(y ~ x1+x2)
  CI<-confint(reg)
  I[i,] <- (CI[,1] <= beta) & (CI[,2] >= beta)
}
# niveaux de confiance des 3 intervalles de conf 
cat("Niveaux de confiance des 3 intervalles de conf :",colMeans(I),"\n")

# probabilite pour le produit cartesien des 3 intervalles de confiance 
# (ce n'est pas une region de confiance au niveau 95 %)
cat("probabilite pour le produit cartesien des 3 intervalles de confiance :",mean(apply(I,1,min)),"\n")
cat("On voit donc bien ici que ce n'est pas une region de confiance au niveau 95 %") 
readline(prompt="Press [enter] to continue")


# Intervalles de confiance et de prediction
cat("Calcul des intervalles de confiance et de prevision","\n","sur Y0 pour une nouvelle valeur x0 = (x10,x20)")
cat("\n==========\n\n")
x0<-c(0.9,0.9)
Ey0=beta[1]+0.9*beta[2]+0.9*beta[3]
N<-5000
IC<-rep(0,N)
IP<-IC

for(i in 1:N){
  y<-Ey+rnorm(n,0,sig) 
  y0<-Ey0+rnorm(1,0,sig)
  reg<-lm(y ~ x1+x2)
  int<-predict(reg,int="c",newdata=data.frame(x1=0.9,x2=0.9))
  IC[i]<-(int[,2] <= Ey0) & (int[,3] >= Ey0)
  int<-predict(reg,int="p",newdata=data.frame(x1=0.9,x2=0.9))
  IP[i]<-(int[,2] <= y0) & (int[,3] >= y0)
}
cat("Moyenne de l'intervalle de prediction :",mean(IP),"\n")
cat("Moyenne de l'intervalle de confiance :",mean(IC),"\n")
cat("L'intervalle de prediction sera toujours plus grand que l'intervalle de confiance")


