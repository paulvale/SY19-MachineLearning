#------------------------------
# Exercise 1
#------------------------------

#------------------------------
# Question a

pi<-0.90
mu<-2
sig<- 1
a=10;
c<-1/(2*a)
n<- 100

x<-vector("numeric",n)
y<-x
for(i in 1:n){
	y[i]=sample(c(1,0),size=1,prob=c(pi,1-pi))
	if(y[i]==1){
		x[i]<- rnorm(1,mean=mu,sd=sig)
	}
	else{
		x[i]<-runif(1,min=-a,max=a)
	}
}

boxplot(x)
dotchart(x)
#------------------------------
# Question b

# Observed data Loglikelihood
loglik<- function(theta,x){
	phi <- sapply(x,dnorm,mean=theta[1],sd=theta[2])
	logL <- sum(log(theta[3]*phi+(1-theta[3])*c))
	return(logL)
}

# EM algoritm
em_outlier <- function(x,theta0,a,epsi){
	go_on<-TRUE
	logL0<- loglik(theta0,x)
	t<-0
	c<-1/(2*a)
	n<-length(x)
	print(c(t,logL0))
	while(go_on){
		t<-t+1
		# E-step
		phi <- sapply(x,dnorm,mean=theta0[1],sd=theta0[2])
		y<-  phi*theta0[3]/(phi*theta0[3]+c*(1-theta0[3]))
		# M-step
		S<- sum(y)
		pi<-S/n
		mu<- sum(x*y)/S
		sig<-sqrt(sum(y*(x-mu)^2)/S)
		theta<-c(mu,sig,pi)
		logL<-loglik(theta,x) 
		if (logL-logL0 < epsi){
			go_on <- FALSE
			}
		logL0 <- logL
		theta0<-theta
		print(c(t,logL))
		}
	return(list(loglik=logL,theta=theta,y=y))
}

#------------------------------
# Question c

mu0<-mean(x) 
sig0<-sd(x)
pi0<-0.5
epsi<-1e-8
theta0<-c(mu0,sig0,pi0)
estim<-em_outlier(x,theta0,a,epsi)

plot(x,1-estim$y)



#------------------------------
# Question d

opt<-optim(theta0, loglik, gr = NULL,method = "BFGS", 
           hessian = FALSE,control=list(fnscale=-1),x=x)

print(opt$par)     # estimates computed with optim
print(estim$theta)  # estimates computed with EM


