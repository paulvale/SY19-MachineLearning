# LOAD DATA
library(rgl)

generate <- function(mu,sigma,pii,a){
	normale <- rnorm(100*pii,mu,sigma)
	r <- 100*(1-pii)
	uniforme <- runif(r,min=-a,max=a)
	x <- sample(c(normale,uniforme))
	return(x)
}
# theta = c(mu, sigma, pi)
theta.init <- c(0.2,10,0.9)
a <- 5
myData <- generate(theta.init[1],theta.init[2],theta.init[3],a)

boxplot(myData)

# Nous avons 2 loi dans ce melange de modeles  : 
#	-loi normale (mu,sigma)
#	-loi uniforme ([-a,a])




# === EM Algorithm ===
K <- 30
evolutionTheta <- matrix(0, nrow=K, ncol=3)
evolutionTheta[1,]<- c(0.2,0.8,0.3)
evolutionY <- matrix(0,nrow=K-1, ncol=99)
evolutionLog <- matrix(0, nrow=1, ncol=99)

oldValue <- 1000
newValue <- 0

for (i in 2:K){
  # E step :
  up <- dnorm(myData, mean = evolutionTheta[i-1,1], sd = evolutionTheta[i-1,2], log = FALSE)*evolutionTheta[i-1,3]
  down <- up + (1/(2*a))*(1-evolutionTheta[i-1,3])
  y <- up/down
  one <- rep(1,99)
  evolution <- one - y
  evolutionY[i-1,]<- evolution
  #plot(evolution)
  # On observe au final une evolution au debut puis ca stagne assez rapidement 
  # La reparitition des donnees est plutot bonne des le debut (depend de l'initialisation)
  # En effet, il est possible qu'au debut il ne detecte rien et petit a petit ca marche mieux
  # qu'importe l'initialisation, on converge rapidement
  
  # M step :
  evolutionTheta[i,1] <- sum(y*myData)/sum(y)                 # mu
  evolutionTheta[i,2] <- sqrt((sum(y*(myData-evolutionTheta[i,1])^2))/sum(y))  # sigma
  evolutionTheta[i,3] <- mean(y)   # pi
  
  # L(theta)
  newValue <- sum(log(evolutionTheta[i,3]*dnorm(myData,mean = evolutionTheta[i,1], sd = evolutionTheta[i,2], log = FALSE))+(1-evolutionTheta[i,3])*(1/(2*a)))
  print(abs(newValue - oldValue))
  oldValue <- newValue
}
plot(1:30,evolutionTheta[,1], type="l",ylim=c(0,3)) # noir = mu 
par(new=TRUE)
plot(1:30,evolutionTheta[,2], col="red", type="l" , xlab="", ylab="", ylim=c(0,3)) # rouge = sigma
par(new=TRUE)
plot(1:30,evolutionTheta[,3], col="green", type="l", xlab="", ylab="", ylim=c(0,3)) # pi = vert 

#persp3d(c(1:29), c(1:99), evolutionY, col="skyblue")