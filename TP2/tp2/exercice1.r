# LOAD DATA
prostate.data = read.table("data/prostate.data.txt")

# Valeur de train sans etiquette + Normalization of their features
train = subset(prostate.data,train == T, select=c(lcavol,lweight,lbph,age))
train = data.frame(lapply(train,function(x) scale(x)))
# Valeur de test sans etiquette + Normalization of their features
test = subset(prostate.data,train == F, select=c(lcavol,lweight,lbph,age))
test = data.frame(lapply(test,function(x) scale(x)))

reg.smsa <- lm(prostate.data$lpsa~prostate.data$lcavol+prostate.data$lweight+prostate.data$age+prostate.data$lbph+prostate.data$svi+prostate.data$lcp+prostate.data$gleason+prostate.data$pgg45)