x1 = runif(300,0,10)
x2 = runif(300,0,100)
X = data.frame(x1=x1,x2=x2)

# centre reduire nos valeurs
X = scale(X)

meanSd = (sd(X[,1]) + sd(X[,2])) /2
epsilon = rnorm(1,mean=0,sd=meanSd)
intercept = runif(1,0,50)

y = intercept + 4*x1^2 + 0.5*x2^2  

reg.smsa <- lm(y~x1+x2)
print(summary(reg.smsa))
plot(reg.smsa)

x0 =  data.frame(x1 = runif(1,0,10), x2 = runif(1,0,100))
print(x0)

print(predict(reg.smsa, int="c", newdata = x0))
