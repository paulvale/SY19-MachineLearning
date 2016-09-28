# LOAD DATA
prostate.data = read.table("data/prostate.data.txt")


#Linear regression method
reg.smsa <- lm(prostate.data$lpsa~prostate.data$lcavol+prostate.data$lweight+prostate.data$age+prostate.data$lbph+prostate.data$svi+prostate.data$lcp+prostate.data$gleason+prostate.data$pgg45)

#Coefficients:
#          (Intercept)   prostate.data$lcavol  prostate.data$lweight  
#             0.181561               0.564341               0.622020  
#   prostate.data$age     prostate.data$lbph      prostate.data$svi  
#            -0.021248               0.096713               0.761673  
#    prostate.data$lcp  prostate.data$gleason    prostate.data$pgg45  
#            -0.106051               0.049228               0.004458 

#Les coefficients significativment non nuls sont lcavol, lweight, svi
#		En notation anglo-saxonne, « Intercept » correspond ici à l’ordonnée 
#		à l’origine le « b » de notre droite et le « x » est la pente de la droite 
#		ce qui correspond au « b » dans notre notation

#f(x)= 0.181561  + 0.564341x1 + 0.622020x2 -0.021248x3 + 0.096713x4 +0.761673x5 -0.106051x6 +0.049228x7 + 0.004458x8