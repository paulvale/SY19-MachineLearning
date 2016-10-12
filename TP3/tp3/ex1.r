# LOAD DATA
spam = read.table("data/spambase.dat.txt")
library(MASS)
library(pROC)

# Dimension des donnees spam
dimSpam = dim(spam)
# label des donnees ( 1 : Spam , 0: non spam)
label = spam[,58]
# centre reduire les valeurs car les dernieres features sont differentes que les premieres
spam = scale(spam[,1:dimSpam[2]-1])

# separation des datas
# ========
napp <- round(2/3*dimSpam[1])
indice <- sample(1:dimSpam[1], napp)

spam.train <- spam[indice,]
spam.test <- spam[-indice,]

label.train <- label[indice]
label.test <- label[-indice]

# ADL
lda <- lda(label.train~., data=as.data.frame(spam.train))
pred.lda <- predict(lda, newdata=as.data.frame(spam.test))
print(summary(pred.lda))
lda.perf <- table(label.test,pred.lda$class)
print(lda.perf)
lda.err <- 1 - sum(diag(lda.perf))/(dimSpam[1]-napp)
print(lda.err)
lda.curve <- roc(label.test,as.vector(pred.lda$x))
plot(lda.curve)

# Logistic Regression
print("=====")
glm <- glm(label.train~., data=as.data.frame(spam.train),family=binomial)
pred.glm <- predict(glm, newdata=as.data.frame(spam.test), type='response')
pred.glm.link <- predict(glm, newdata=as.data.frame(spam.test), type='link')
print(summary(pred.glm))
glm.perf <- table(label.test,pred.glm > 0.5)
print(glm.perf)
glm.err <- 1 - sum(diag(glm.perf))/(dimSpam[1]-napp)
print(glm.err)
glm.curve <- roc(label.test,pred.glm.link)

plot(lda.curve)
plot(glm.curve,add=TRUE, col='red')

# ===
# Derniere question
# ===

print(summary(glm))


