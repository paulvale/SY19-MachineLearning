

# ============================
# Explications de chaque terme:
# ============================
# - E[(f^k - y)^2)]: formule de la d??composition bias-variance. Cette esp??rance correspond au MSE (Mean Square Error).
#                    Pour un algorithme donn??, nous allons ainsi essayer d'avoir cette erreur la plus faible possible.
#                    Plus cette esp??rance sera faible, plus notre estimation s'averera correcte. 
#                    L'estimation parfaite n'existe pas, une erreur irr??ductible est toujours pr??sente !  

# - Var[Epsilon]: C'est l'erreur irr??ductible, on ne peut donc pas la diminuer. Cette erreur est une constante   

# - Var[f^k]: Variance de notre mod??le. Plus cette variable est importante, plus on dit que notre mod??le est flexible.
#             En effet, si notre mod??le a une grande variance, alors il a donc de grandes diff??rences entre ses estimations.
#             Lorsque l'on a une variance trop importante, notre mod??le suit alors trop notre ensemble de training et 
#             on parle alors d'overfitting.

# - (E[f^k]-f)^2: Biais de notre mod??le, il correspond ?? la diff??rence entre la moyenne des predictions du mod??le et 
#                  les veritables ??tiquettes (ou "valeur") de nos donn??es d'apprentissage. Contrairement ?? la variance, 
#                  plus un mod??le est flexible, plus son biais sera faible. En effet, plus notre mod??le est flexible, 
#                  plus il se rapproche de nos donn??es d'apprentissage et donc plus nos estimation sont proche de leur 
#                  "veritable" valeur".
#                  Si notre mod??le est trop simple, on va alors parler d'underfitting. On aura alors un biais ??l??v??
#                  et une variable faible car le mod??le sera tres rigide.


n_values = c(50,100,150,200)

for(n in n_values){
    rep = 100
    nbK = round(n-n/10)                                        # Max de Kppv a prendre en compte
    reg_f_evol = matrix(nrow=rep,ncol=nbK)          # Matrix pour stocker reg_f        sur les 100 iterations
    var_reg_f_evol = matrix(nrow=rep,ncol=nbK)      # "                 " var_reg_f    ... 
    biais_reg_f_evol = matrix(nrow=rep,ncol=nbK)    # "                 " biais_reg_f  ...
    epsilon_evol = c()                              # "                 " epsilon_evol ...

    # Prise aleatoire d'un x compris entre 0 et 1 comme donn??e de test
    x = runif(1,0,1)
    f_x = 1 + 5*x*x

    for(i in 1:rep){
        # Initialisation des datas X,epsilon, et Y
        X = runif(n,0,1)
        epsilon_tmp = rnorm(n,0,0.5)
        Y = 1+5*X*X + epsilon_tmp

        # Calcule des distances entre x et les donn??es d'apprentissage
        dist_x_to_training = abs(X - x)
        
        # Reordonne par ordre croissant afin d'avoir les Kppv en 1er
        kppv_x = cbind(Y, dist_x_to_training)
        kppv_x = kppv_x[order(dist_x_to_training) , ]

        reg_f_tmp = c()
        for (K in 1:nbK) {
            reg_f_for_K = mean(kppv_x[1:K,1])       # Selectionne les K_1ers Y pour en faire une moyenne
            reg_f_tmp  = c(reg_f_tmp,reg_f_for_K)   # Ajoute le reg_f a notre reg_f_tmp
        }
        reg_f_evol[i,] = reg_f_tmp
        epsilon_evol = c(epsilon_evol,epsilon_tmp)
    }

    var_reg_f_evol = apply(reg_f_evol,2,var)
    biais_reg_f_evol = (apply(reg_f_evol,2,mean)-f_x)^2
    var_epsilon = rep(var(epsilon_evol),nbK)
    erreur = var_reg_f_evol + biais_reg_f_evol + var_epsilon

    pdf(file = paste(getwd(),"/graphs/graphs",n,".pdf",sep=""))

    plot(biais_reg_f_evol,col="black",xlab="K",ylab="Biais", main="Biais en fonction de K")
    plot(var_reg_f_evol,col="black",xlab="K",ylab="Variance", main="Variance en fonction de K")	
    plot(var_epsilon,col="black",xlab="K",ylab="Var epsilon", main="Variance d'Epsilon en fonction de K")	
    plot(erreur,col="black",xlab="K",ylab="MSE", main="MSE en fonction de K")	

    dev.off()
}

# ===============
# Observation des graphiques:
# ===============
# Nous avons choisi de r??aliser 100 simulations et g??n??rations al??atoires afin de pouvoir valider nos hypotheses. De plus, 
# nous avons choisi de faire varier K entre 1 et 40 comme demand?? dans l'??nonc??.
# Les graphiques obtenues nous montre que plus le nombre de voisins K augmente:
# - plus la variance diminue. Var[f^k] est une fonction d??croissante monotone de K.
# - plus le biais augmente. Le biais est une fonction croissante monotone de K.
# - Var[Epsilon] reste constant. La variance d'epsilon est une fonction constante.
# - la MSE d??cro??t tout d'abord vers son minimum global avant de cro??tre.
#
# L'observation de ces graphiques valident donc nos hypotheses de d??part.
# En effet, K petit correspond ?? un mod??le KPPV tr??s flexible  qui suit ainsi "parfaitement" ou presque les points de l'ensemble d'apprentissage.
# Expliquant donc cette grande variance et un biais qui est tres faible. 
# Plus on augmente K, plus ces fronti??res vont devenir "lisse", le mod??le perd donc en flexibilit??, sa variance diminue ainsi petit a petit
# contrairement a son biais qui sera alors de plus en plus eleve.
#
# => On peut donc conclure avec ces relations : 
#  - K petit: variance importante, biais faible.
#  - K grand: variance faible et fort biais.
# 
# La MSE correspond a la somme de la variance, le biais et epsilon. On peut observer qu'elle contient un minimum global 
# correspondant au bon compromis entre la variance et le biais, entre un modele qui est trop simple et un modele qui est trop flexible.

# ===============
# Observation sur le changement du nombre de variables d'apprentissage:
# ===============

# Tout d'abord, on observe que qque soit le nombre de variables d'apprentissage, nous avons des courbes qui ont toujours la meme allure
# Cependant, lorsque l'on regarde de plus pret, les valeurs de chacune des composantes, on observe que :

#            | Variance | Biais | erreur |
# - n = 50   |          |       |        |
# - n = 100  |          |       |        |
# - n = 150  |          |       |        |
# - n = 200  |          |       |        |




# ======
# Solution donn??e par le prof
# =====

#n <- 50
#x0 <- 0.5
#Ey0 <- 1+5*x0^2
#Kmax <- 40
#N <- 10000
#y0 <- rep(0,N)
#yhet <- matrix(0,N,Kmax)
#for(i in 1:N) {
#  x <- runif(n)
#  y <- 1+5*x^2 + sig*rnorm(n)
#  d <- abs(x-xo)
#  tri <- sort(d, index.return=TRUE)
#  #tirage du Y0
#  y0[i] <- Ey0 + sig*rnorm(i)
#  for(k in 1:Kmax){
#    yhet[i,k]<-mean(y[tri$ix[1:k]])
#  }
#}

#error <-rep(0,Kmax)
#biais2 <- rep(0,Kmax)
#variance <- rep(0,Kmax)
#for(k in 1:Kmax){
#    error[k] <- mean((yo-yhet[,k])^2)
#    biais2 <- (mean(yhet[,k])-Ey0)^2
#    variance[k]<-var(yhet[,k])
#}

# Plot les resultats
#plot(1:Kmax, error, type='l', ylim=range(error, biais2,variance))
#lines(1:Kmax, biais2, lty=2)
#lines(1:Kmax, variance, lty=2)
#lines(1:Kmax, variance+biais2+sig^2, lty=3)

# error = difference entre le y et la prediction que l'on en a fait
# biais = difference entre la valeur moyenne de notre prediction et la moyenne des y 

  
#}