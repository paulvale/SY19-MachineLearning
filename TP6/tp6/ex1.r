generate <- function(mu,sigma,pii,a){

	normale <- rnorm(100*pii,mu,sigma)
	uniforme <- runif(100*(1-pii),-a,a)
	x <- sample(c(normale,uniforme),size=100,replace=TRUE)
	return(x)
}


boxplot(generate(0,1,0.8,5))

# Nous avons 2 loi dans ce mélange de modeles  : 
#	-loi normale (mu,sigma)
#	-loi uniforme ([-a,a])

