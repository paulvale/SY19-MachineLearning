

# ============================
# Explications de chaque terme:
# ============================
# - E[(f^k - y)^2)]: formule de la décomposition bias-variance. Cette espérance correspond au MSE (Mean Square Error).
#                    Pour un algorithme donné, nous allons ainsi essayer d'avoir cette erreur la plus faible possible.
#                    Plus cette espérance sera faible, plus notre estimation s'averera correcte. 
#                    L'estimation parfaite n'existe pas, une erreur irréductible est toujours présente !  

# - Var[Epsilon]: C'est l'erreur irréductible, on ne peut donc pas la diminuer. Cette erreur est une constante   

# - Var[f^k]: Variance de notre modèle. Plus cette variable est importante, plus on dit que notre modèle est flexible.
#             En effet, si notre modèle a une grande variance, alors il a donc de grandes différences entre ses estimations.
#             Lorsque l'on a une variance trop importante, notre modèle suit alors trop notre ensemble de training et 
#             on parle alors d'overfitting.

# - (E[f^k]-f)^2: Biais de notre modèle, il correspond à la différence entre la moyenne des predictions du modèle et 
#                  les veritables étiquettes (ou "valeur") de nos données d'apprentissage. Contrairement à la variance, 
#                  plus un modèle est flexible, plus son biais sera faible. En effet, plus notre modèle est flexible, 
#                  plus il se rapproche de nos données d'apprentissage et donc plus nos estimation sont proche de leur 
#                  "veritable" valeur".
#                  Si notre modèle est trop simple, on va alors parler d'underfitting. On aura alors un biais élévé
#                  et une variable faible car le modèle sera tres rigide.


n_values = c(50,100,150,200)

for(n in n_values){
    rep = 100
    nbK = round(n-n/10)                                        # Max de Kppv a prendre en compte
    reg_f_evol = matrix(nrow=rep,ncol=nbK)          # Matrix pour stocker reg_f        sur les 100 iterations
    var_reg_f_evol = matrix(nrow=rep,ncol=nbK)      # "                 " var_reg_f    ... 
    biais_reg_f_evol = matrix(nrow=rep,ncol=nbK)    # "                 " biais_reg_f  ...
    epsilon_evol = c()                              # "                 " epsilon_evol ...

    # Prise aleatoire d'un x compris entre 0 et 1 comme donnée de test
    x = runif(1,0,1)
    f_x = 1 + 5*x*x

    for(i in 1:rep){
        # Initialisation des datas X,epsilon, et Y
        X = runif(n,0,1)
        epsilon_tmp = rnorm(n,0,0.5)
        Y = 1+5*X*X + epsilon_tmp

        # Calcule des distances entre x et les données d'apprentissage
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
# Nous avons choisi de réaliser 100 simulations et générations aléatoires afin de pouvoir valider nos hypotheses. De plus, 
# nous avons choisi de faire varier K entre 1 et 40 comme demandé dans l'énoncé.
# Les graphiques obtenues nous montre que plus le nombre de voisins K augmente:
# - plus la variance diminue. Var[f^k] est une fonction décroissante monotone de K.
# - plus le biais augmente. Le biais est une fonction croissante monotone de K.
# - Var[Epsilon] reste constant. La variance d'epsilon est une fonction constante.
# - la MSE décroît tout d'abord vers son minimum global avant de croître.
#
# L'observation de ces graphiques valident donc nos hypotheses de départ.
# En effet, K petit correspond à un modèle KPPV très flexible  qui suit ainsi "parfaitement" ou presque les points de l'ensemble d'apprentissage.
# Expliquant donc cette grande variance et un biais qui est tres faible. 
# Plus on augmente K, plus ces frontières vont devenir "lisse", le modèle perd donc en flexibilité, sa variance diminue ainsi petit a petit
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
