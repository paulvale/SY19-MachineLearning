# LOAD PACKAGE
library("MASS")
library("splines")

data <- mcycle
data.dim <- dim(data)

# Visualisation des donnees
plot(data$times, data$accel)

# Q1 :
polyVector <- c(3:15)
for(i in polyVector){
  fit <-lm(accel~bs(times,df=i),data=data)
  ypred <- predict(fit, newdata=data.frame(times=0:60),interval="c")
  plot(data$times,data$accel,cex=0.5,xlab="times",ylab="acceleration")
  lines(0:60,ypred[,"fit"],lty=1,col="blue",lwd=2)
  lines(0:60,ypred[,"lwr"],lty=2,col="blue",lwd=2)
  lines(0:60,ypred[,"upr"],lty=2,col="blue",lwd=2)
  #print(i)
  #readline(prompt="Press [enter] to continue")
}

# Observation:
# On prendrait 9 car cela nous donne une bonne approximation sans etre trop eleve


# Q2:
error.bs <- rep(0,length(polyVector))
folds = sample(1:5,data.dim[1], replace=TRUE)
for(i in polyVector){
  for(k in (1:5)){
    reg <- lm(accel~bs(times,df=i),data=data[folds!=k,])
    pred <- predict(reg, newdata=data[folds==k,])
    error.bs[i] <- error.bs[i] + sum((data$accel[folds==k]-pred)^2)
  }
  error.bs[i]<-error.bs[i]/data.dim[1]
}

# Obersvation
# Prendre 7 semble encore plus satisfaisant ! Reduire d'un degre est toujours plus interessant

# Q3 :
# Choix via observation
polyVector <- c(3:15)
for(i in polyVector){
  fit <-lm(accel~ns(times,df=i),data=data)
  ypred <- predict(fit, newdata=data.frame(times=0:60),interval="c")
  plot(data$times,data$accel,cex=0.5,xlab="times",ylab="acceleration")
  lines(0:60,ypred[,"fit"],lty=1,col="blue",lwd=2)
  lines(0:60,ypred[,"lwr"],lty=2,col="blue",lwd=2)
  lines(0:60,ypred[,"upr"],lty=2,col="blue",lwd=2)
  #print(i)
  #readline(prompt="Press [enter] to continue")
}

# Observation:
# On prendrait 7 car cela nous donne une bonne approximation sans etre trop eleve


# Choix via Validation Croisee:
error.ns <- rep(0,length(polyVector))
for(i in polyVector){
  for(k in (1:5)){
    reg <- lm(accel~ns(times,df=i),data=data[folds!=k,])
    pred <- predict(reg, newdata=data[folds==k,])
    error.ns[i] <- error.ns[i] + sum((data$accel[folds==k]-pred)^2)
  }
  error.ns[i]<-error.ns[i]/data.dim[1]
}

# Obersvation
# Prendre 6 semble encore plus satisfaisant ! Reduire d'un degre est toujours plus interessant

# Q4:
# Via le leave-one-out
error.smooth <- rep(0,length(polyVector))
for(i in polyVector){
  for(k in (1:data.dim[1])){
    reg <- smooth.spline(data$times[-k],data$accel[-k],df=i)
    pred <- predict(reg,data$times[k])
    error.smooth[i-2] <- error.smooth[i-2] + ((data$accel[k]-pred$y)^2)
  }
  error.smooth[i-2]<-error.smooth[i-2]/data.dim[1]
}
#plot(polyVector, error.smooth, xlab="degree", ylab="error", type="l")

# On peut voir ici qu'a partir de 10, nous avons une bonne estimation

# Via validation croisee
polyVectorSpecialise <- c(8:12)
error.smooth.cv <- rep(0,length(polyVectorSpecialise))

for(i in polyVectorSpecialise){
  for(k in (1:5)){
    reg <- smooth.spline(data$times[folds!=k],data$accel[folds!=k],df=i)
    pred <- predict(reg,data$times[folds==k])
    error.smooth.cv[i-7] <- error.smooth.cv[i-7] + sum((data$accel[folds==k]-pred$y)^2)
  }
  error.smooth.cv[i-7]<-error.smooth.cv[i-7]/data.dim[1]
}
plot(polyVectorSpecialise, error.smooth.cv, xlab="degree", ylab="error", type="l")
# Observation :
# Via validation croisee, on s'apercoit tout de meme qu'il est interessant d'avoir un plus grande nombre de degre
# on pourrait meme dire via l'observation du graphe que plus notre degre sera grand, mieux ce sera

# Q5 :
fit.bs <-lm(accel~bs(times,df=7),data=data)
ypred.bs <- predict(fit.bs, newdata=data.frame(times=0:60),interval="c")

fit.ns <-lm(accel~ns(times,df=6),data=data)
ypred.ns <- predict(fit.ns, newdata=data.frame(times=0:60),interval="c")

fit.smooth <- smooth.spline(data$times,data$accel,df=12)
ypred.smooth <- predict(fit.smooth,data$times)


plot(data$times, data$accel)
lines(0:60,ypred.bs[,"fit"],lty=1,col="blue",lwd=2)
lines(0:60,ypred.ns[,"fit"],lty=1,col="red",lwd=2)
lines(ypred.smooth$x,ypred.smooth$y, lty=1, col="green", lwd=2)


