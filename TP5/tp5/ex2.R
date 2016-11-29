# LOAD DATA
library("leaps")
library("gam")
prostate.data = read.table("data/prostate.data.txt")

data.train = prostate.data[which(prostate.data$train == TRUE),]
data.test = prostate.data[which(prostate.data$train == FALSE),]
data.train$train = NULL
data.test$train = NULL

reg.fit<-regsubsets(lpsa~.,data=data.train,method="exhaustive",nvmax=8)
plot(reg.fit,scale="bic")
# As we can see in the warning message, we have a linear dependencie which mean that we have
# some redundancy in our data

# lcavol, lweight, lbph, svi
prostate.bic = data.train
prostate.bic$gleason = NULL
prostate.bic$age = NULL
prostate.bic$pgg43 = NULL
prostate.bic$lcp = NULL

data.test$train = NULL
data.test$gleason = NULL
data.test$age = NULL
data.test$pgg43 = NULL
data.test$lcp = NULL

# test de differents modeles
# NS Models
ns1.1 <- gam(lpsa~ns(lcavol, df=3)+lweight+lbph+svi, data= prostate.bic)
ns1.2 <- gam(lpsa~lcavol+ns(lweight, df=3)+lbph+svi, data= prostate.bic)
ns1.3 <- gam(lpsa~lcavol+lweight+ns(lbph, df=3)+svi, data= prostate.bic)

ns2.1 <- gam(lpsa~ns(lcavol, df=3)+ns(lweight, df=3)+lbph+svi, data= prostate.bic)
ns2.2 <- gam(lpsa~ns(lcavol, df=3)+lweight+ns(lbph,df=3)+svi, data= prostate.bic)
ns2.3 <- gam(lpsa~lcavol+ns(lweight, df=3)+ns(lbph, df=3)+svi, data= prostate.bic)

ns3 <- gam(lpsa~ns(lcavol, df=3)+ns(lweight, df=3)+ns(lbph, df=3)+svi, data= prostate.bic)

# NS Predictions
data.test <- data.test[order(data.test$lcavol),]
yns1.1 <- predict(ns1.1, newdata=data.test, interval="c")
yns1.2 <- predict(ns1.2, newdata=data.test, interval="c")
yns1.3 <- predict(ns1.3, newdata=data.test, interval="c")

plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, yns1.1, lty=1, lwd=2, col="blue")
lines(data.test$lcavol, yns1.2, lty=1, lwd=2, col="red")
lines(data.test$lcavol, yns1.3, lty=1, lwd=2, col="green")
#readline(prompt = "Press [Enter]")

yns2.1 <- predict(ns2.1, newdata=data.test, interval="c")
yns2.2 <- predict(ns2.2, newdata=data.test, interval="c")
yns2.3 <- predict(ns2.3, newdata=data.test, interval="c")

plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, yns2.1, lty=1, lwd=2, col="blue")
lines(data.test$lcavol, yns2.2, lty=1, lwd=2, col="red")
lines(data.test$lcavol, yns2.3, lty=1, lwd=2, col="green")
#readline(prompt = "Press [Enter]")

yns3 <- predict(ns3, newdata=data.test, interval="c")
plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, yns3, lty=1, lwd=2, col="blue")
#readline(prompt = "Press [Enter]")

# BS Models
bs1.1 <- gam(lpsa~bs(lcavol, df=3)+lweight+lbph+svi, data= prostate.bic)
bs1.2 <- gam(lpsa~lcavol+bs(lweight, df=3)+lbph+svi, data= prostate.bic)
bs1.3 <- gam(lpsa~lcavol+lweight+bs(lbph, df=3)+svi, data= prostate.bic)

bs2.1 <- gam(lpsa~bs(lcavol, df=3)+bs(lweight, df=3)+lbph+svi, data= prostate.bic)
bs2.2 <- gam(lpsa~bs(lcavol, df=3)+lweight+bs(lbph,df=3)+svi, data= prostate.bic)
bs2.3 <- gam(lpsa~lcavol+bs(lweight, df=3)+bs(lbph, df=3)+svi, data= prostate.bic)

bs3 <- gam(lpsa~bs(lcavol, df=3)+bs(lweight, df=3)+bs(lbph, df=3)+svi, data= prostate.bic)

# BS Predictions
data.test <- data.test[order(data.test$lcavol),]
ybs1.1 <- predict(bs1.1, newdata=data.test, interval="c")
ybs1.2 <- predict(bs1.2, newdata=data.test, interval="c")
ybs1.3 <- predict(bs1.3, newdata=data.test, interval="c")

plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, ybs1.1, lty=1, lwd=2, col="blue")
lines(data.test$lcavol, ybs1.2, lty=1, lwd=2, col="red")
lines(data.test$lcavol, ybs1.3, lty=1, lwd=2, col="green")
#readline(prompt = "Press [Enter]")

ybs2.1 <- predict(bs2.1, newdata=data.test, interval="c")
ybs2.2 <- predict(bs2.2, newdata=data.test, interval="c")
ybs2.3 <- predict(bs2.3, newdata=data.test, interval="c")

plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, ybs2.1, lty=1, lwd=2, col="blue")
lines(data.test$lcavol, ybs2.2, lty=1, lwd=2, col="red")
lines(data.test$lcavol, ybs2.3, lty=1, lwd=2, col="green")
#readline(prompt = "Press [Enter]")

ybs3 <- predict(bs3, newdata=data.test, interval="c")
plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, ybs3, lty=1, lwd=2, col="blue")
#readline(prompt = "Press [Enter]")


# Smooth Splines Models
ss1.1 <- gam(lpsa~s(lcavol, df=3)+lweight+lbph+svi, data= prostate.bic)
ss1.2 <- gam(lpsa~lcavol+s(lweight, df=3)+lbph+svi, data= prostate.bic)
ss1.3 <- gam(lpsa~lcavol+lweight+s(lbph, df=3)+svi, data= prostate.bic)

ss2.1 <- gam(lpsa~s(lcavol, df=3)+s(lweight, df=3)+lbph+svi, data= prostate.bic)
ss2.2 <- gam(lpsa~s(lcavol, df=3)+lweight+s(lbph,df=3)+svi, data= prostate.bic)
ss2.3 <- gam(lpsa~lcavol+s(lweight, df=3)+s(lbph, df=3)+svi, data= prostate.bic)

ss3 <- gam(lpsa~s(lcavol, df=3)+s(lweight, df=3)+s(lbph, df=3)+svi, data= prostate.bic)

# SS Predictions
data.test <- data.test[order(data.test$lcavol),]
yss1.1 <- predict(ss1.1, newdata=data.test, interval="c")
yss1.2 <- predict(ss1.2, newdata=data.test, interval="c")
yss1.3 <- predict(ss1.3, newdata=data.test, interval="c")

plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, yss1.1, lty=1, lwd=2, col="blue")
lines(data.test$lcavol, yss1.2, lty=1, lwd=2, col="red")
lines(data.test$lcavol, yss1.3, lty=1, lwd=2, col="green")
#readline(prompt = "Press [Enter]")

yss2.1 <- predict(ss2.1, newdata=data.test, interval="c")
yss2.2 <- predict(ss2.2, newdata=data.test, interval="c")
yss2.3 <- predict(ss2.3, newdata=data.test, interval="c")

plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, yss2.1, lty=1, lwd=2, col="blue")
lines(data.test$lcavol, yss2.2, lty=1, lwd=2, col="red")
lines(data.test$lcavol, yss2.3, lty=1, lwd=2, col="green")
#readline(prompt = "Press [Enter]")

yss3 <- predict(ss3, newdata=data.test, interval="c")
plot(data.test$lcavol, data.test$lpsa)
lines(data.test$lcavol, yss3, lty=1, lwd=2, col="blue")
#readline(prompt = "Press [Enter]")


# Errors 
errors <- rep(0,21)
errors[1]<- sum((data.test$lpsa - ybs1.1)^2)
errors[2]<- sum((data.test$lpsa - ybs1.2)^2)
errors[3]<- sum((data.test$lpsa - ybs1.3)^2)
errors[4]<- sum((data.test$lpsa - ybs2.1)^2)
errors[5]<- sum((data.test$lpsa - ybs2.2)^2)
errors[6]<- sum((data.test$lpsa - ybs2.3)^2)
errors[7]<- sum((data.test$lpsa - ybs3)^2)
errors[8]<- sum((data.test$lpsa - yns1.1)^2)
errors[9]<- sum((data.test$lpsa - yns1.2)^2)
errors[10]<- sum((data.test$lpsa - yns1.3)^2)
errors[11]<- sum((data.test$lpsa - yns2.1)^2)
errors[12]<- sum((data.test$lpsa - yns2.2)^2)
errors[13]<- sum((data.test$lpsa - yns2.3)^2)
errors[14]<- sum((data.test$lpsa - yns3)^2)
errors[15]<- sum((data.test$lpsa - yss1.1)^2)
errors[16]<- sum((data.test$lpsa - yss1.2)^2)
errors[17]<- sum((data.test$lpsa - yss1.3)^2)
errors[18]<- sum((data.test$lpsa - yss2.1)^2)
errors[19]<- sum((data.test$lpsa - yss2.2)^2)
errors[20]<- sum((data.test$lpsa - yss2.3)^2)
errors[21]<- sum((data.test$lpsa - yss3)^2)

# observation:
# qu'importe le choix que l'on va faire on a une tres bonne estimation ici de nos valeurs
# car nous avons une erreur tres faible

errors <- data.frame(errors)

# on peut voir ici que les modeles qui nous donnent le moins d'erreurs sont les modeles
# 8, 11, 14, 1 et on va prendre le 15 pour avoir au moins un modele de smooth spline
# or bs => 1 a 7 : donc 1 modele
# bs1.1
# ns => 8 a 14 : donc 3 modeles
# ns1.1
# ns2.1
# ns3
# ss => 15
# ss1.1

# On va dorenavant faire une crossValidation pour choisir le meilleur modele parmis ces 5 modeles

data = prostate.data
data$train = NULL
data$gleason = NULL
data$age = NULL
data$pgg43 = NULL
data$lcp = NULL

data.dim <- dim(data)
errors.cv <- rep(0,5)
folds = sample(1:5,data.dim[1], replace=TRUE)

# bs1.1
for(k in (1:5)){
  reg <- gam(lpsa~bs(lcavol, df=3)+lweight+lbph+svi, data= data[folds!=k,])
  pred <- predict(reg, newdata=data[folds==k,], interval="c")
  errors.cv[1]<- errors.cv[1] + sum((data$lpsa[folds==k] - pred)^2)
}
errors.cv[1]<-errors.cv[1]/data.dim[1]

# ns1.1
for(k in (1:5)){
  reg <- gam(lpsa~ns(lcavol, df=3)+lweight+lbph+svi, data= data[folds!=k,])
  pred <- predict(reg, newdata=data[folds==k,], interval="c")
  errors.cv[2]<- errors.cv[2] + sum((data$lpsa[folds==k] - pred)^2)
}
errors.cv[2]<-errors.cv[2]/data.dim[1]

# ns2.1
for(k in (1:5)){
  reg <- gam(lpsa~ns(lcavol, df=3)+ns(lweight, df=3)+lbph+svi, data= data[folds!=k,])
  pred <- predict(reg, newdata=data[folds==k,], interval="c")
  errors.cv[3]<- errors.cv[3] + sum((data$lpsa[folds==k] - pred)^2)
}
errors.cv[3]<-errors.cv[3]/data.dim[1]

# ns3
for(k in (1:5)){
  reg <- gam(lpsa~ns(lcavol, df=3)+ns(lweight, df=3)+ns(lbph, df=3)+svi, data= data[folds!=k,])
  pred <- predict(reg, newdata=data[folds==k,], interval="c")
  errors.cv[4]<- errors.cv[4] + sum((data$lpsa[folds==k] - pred)^2)
}
errors.cv[4]<-errors.cv[4]/data.dim[1]

# ss1.1
for(k in (1:5)){
  reg <- gam(lpsa~s(lcavol, df=3)+lweight+lbph+svi, data= data[folds!=k,])
  pred <- predict(reg, newdata=data[folds==k,], interval="c")
  errors.cv[5]<- errors.cv[5] + sum((data$lpsa[folds==k] - pred)^2)
}
errors.cv[5]<-errors.cv[5]/data.dim[1]

# au final dans notre validation croisee c'est tout de meme notre dernier modele qui nous donne 
# le meilleur resultat
# du coup le modele avec les smooth splines est le meilleur
# Cependant on peut tout de meme voir que le resultat est tres bon pour tous nos modeles
# et qu'ils ont une difference d'erreur tres faibles
plot(errors.cv)






