generateProf <- function(mu,sigma, pii, a, N){
  # Solution avec boucle donc moins bien
  #for(1 = 1:N){
   # y[i] <- sample(c(1,0), size=1, prob = c(pi, 1-pi))
    #if(y[i] == 1){
     # x[i] <- rnorm(1, mu, sigma)
    #} else {
     # x[i] <- runif(1,min=-a, max=a)
    #}
  #}
  
  # Solution sans boucle optimale
  N1 <- rbinom(1, size=N, prob = pii)
  x1 <- rnorm(N1, mu, sigma)
  x2 <- runif(N-N1, min=-a, max=a)
  
  return(c(x1,x2))
}

theta <- c(2,1,0.9)
a <- 10
N <- 100
precision <- 10e-6
go_on = TRUE

myData <- generateProf(theta[1], theta[2],theta[3], a, N)
boxplot(myData)

# === EM Algorithm ===

logValue <- sum(log(dnorm(myData, mean = theta[1], sd = theta[2], log = FALSE)*theta[3]+(1-theta[3])*(1/(2*a))))

while(go_on){
  # E step :
  up <- dnorm(myData, mean = theta[1], sd = theta[2], log = FALSE)*theta[3]
  down <- up + (1/(2*a))*(1-theta[3])
  y <- up/down
  
  # M step :
  theta[1] <- sum(y*myData)/sum(y)                 # mu
  theta[2] <- sqrt((sum(y*(myData-theta[1])^2))/sum(y))  # sigma
  theta[3] <- mean(y)   # pi
  
  #L(theta)
  newValue <- sum(log(dnorm(myData, mean = theta[1], sd = theta[2], log = FALSE)*theta[3]+(1-theta[3])*(1/(2*a))))

  if(newValue - logValue[length(logValue)] < precision ){
    go_on = FALSE
  }
  logValue <- c(logValue, newValue)
}

plot(logValue)
