generate <- function(mu,sigma,pii,a){
	normale <- rnorm(100*pii,mu,sigma)
	print(length(normale))
	r <- 100*(1-pii)
	uniforme <- runif(r,min=-a,max=a)
	print(length(uniforme))
	x <- sample(c(normale,uniforme))
	print(x)
	return(x)
}


myData <- generate(0,1,0.8,5)

boxplot(generate(0,1,0.8,5))

# Nous avons 2 loi dans ce melange de modeles  : 
#	-loi normale (mu,sigma)
#	-loi uniforme ([-a,a])

