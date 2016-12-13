# TP 5

#---------------- Exercice 1 ------------------

library('MASS')
n<-length(mcycle$times)

plot(mcycle$times,mcycle$accel)
test.data<-data.frame(times=seq(0,60,1))

summary(mcycle)


# q1

fit1=lm(accel~poly(times,3),data=mcycle)

plot(mcycle$times,mcycle$accel)
lines(mcycle$times,fit1$fitted.values)

fit2=lm(accel~poly(times,10),data=mcycle)

plot(mcycle$times,mcycle$accel)
lines(mcycle$times,fit2$fitted.values)

# etc...

# q2

# 5-fold cross-validation
K<-5
folds=sample(1:K,n,replace=TRUE)
P<-1:10
N<-length(P)
CV1<-rep(0,N)
for(i in (1:N)){
  print(i)
  for(k in (1:K)){
    fit=lm(accel~poly(times,P[i]),data=mcycle[folds!=k,])
    pred<-predict(fit,newdata=mcycle[folds==k,])
    print(pred)
    CV1[i]<-CV1[i]+ sum((mcycle$accel[folds==k]-pred)^2)
  }
  CV1[i]<-CV1[i]/n
}
plot(P,CV1,type='b',xlab='p',ylab='CV error')

# q3
library(splines)
fit=lm(accel~ns(times,df=7),data=mcycle)
plot(mcycle$times,mcycle$accel)
lines(mcycle$times,fit$fitted.values)


DF<-5:20
N<-length(DF)
CV2<-rep(0,N)
for(i in (1:N)){
  print(i)
  for(k in (1:K)){
    fit=lm(accel~ns(times,df=DF[i]),data=mcycle[folds!=k,])
    pred<-predict(fit,newdata=mcycle[folds==k,])
    CV2[i]<-CV2[i]+ sum((mcycle$accel[folds==k]-pred)^2)
  }
  CV2[i]<-CV2[i]/n
}
plot(DF,CV2,type='b',xlab='df',ylab='CV error')

fit=lm(accel~ns(times,df=10),data=mcycle)
plot(mcycle$times,mcycle$accel)
lines(mcycle$times,fit$fitted.values)


# q4

fit=smooth.spline(mcycle$times,mcycle$accel,cv=TRUE)
dfopt<-fit$df
fit=smooth.spline(mcycle$times,mcycle$accel,df=dfopt)
plot(mcycle$times,mcycle$accel)
lines(fit$x,fit$y)

DF<-5:20
N<-length(DF)
CV3<-rep(0,N)
for(i in (1:N)){
  print(i)
  for(k in (1:K)){
    fit=smooth.spline(mcycle$times[folds!=k],mcycle$acce[folds!=k],df=DF[i])
    pred<-predict(fit,mcycle$times[folds==k])
    CV3[i]<-CV3[i]+ sum((mcycle$accel[folds==k]-pred$y)^2)
  }
  CV3[i]<-CV3[i]/n
}
plot(DF,CV3,type='b',xlab='df',ylab='CV error')

# q5

fit1=lm(accel~poly(times,10),data=mcycle)
fit2=lm(accel~ns(times,df=12),data=mcycle)
fit3=smooth.spline(mcycle$times,mcycle$accel,df=dfopt)

plot(mcycle$times,mcycle$accel)
lines(mcycle$times,fit1$fitted.values)
lines(mcycle$times,fit2$fitted.values,lty=2)
lines(fit3$x,fit3$y,lty=3)

min(CV1)
min(CV2)
min(CV3)
#---------------- Exercice 2 ------------------

prostate<-read.table('data/prostate.data.txt',header = TRUE)  
# changer le chemin d'acces aux donnees

summary(prostate)
prostate$svi<-factor(prostate$svi)

library('leaps')
reg.fit<-regsubsets(lpsa~.-train,data=prostate,method='exhaustive',nvmax=15)
plot(reg.fit,scale="bic")
plot(reg.fit,scale="adjr2")

library(gam)
gam1=lm(lpsa~ns(lcavol,3)+ns(age,3)+ns(lweight,4)+svi,data=prostate)
par(mfrow=c(2,2))
plot.gam(gam1, se=TRUE, col="red")
par(mfrow=c(1,1))


gam2=gam(lpsa~s(lcavol,3)+s(lweight,3)+s(age,3)+svi,data=prostate)
par(mfrow=c(2,2))
plot(gam2, se=TRUE, col="blue")
par(mfrow=c(1,1))
