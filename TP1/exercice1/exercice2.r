#
# INSTALLATION PACKAGES
#install.packages("BioPhysConnectoR")

# LOAD PACKAGES
library("BioPhysConnectoR", character.only = TRUE)



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

n = 50                                          # Nombre de données d'apprentissage
nbK = 40                                        # Max de Kppv a prendre en compte
reg_f_evol = matrix(nrow=100,ncol=nbK)          # Matrix pour stocker reg_f        sur les 100 iterations
var_reg_f_evol = matrix(nrow=100,ncol=nbK)      # "                 " var_reg_f    ... 
biais_reg_f_evol = matrix(nrow=100,ncol=nbK)    # "                 " biais_reg_f  ...
epsilon_evol = c()                              # "                 " epsilon_evol ...

# Prise aleatoire d'un x compris entre 0 et 1 comme donnée de test
x = runif(1,0,1)
f_x = 1 + 5*x*x

for(i in 1:100){
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
        reg_f_for_K = mean(kppv_x[1:K,1])
        reg_f_tmp  = c(reg_f_tmp,reg_f_for_K)
    }
    reg_f_evol[i,] = reg_f_for_K
    epsilon_evol = c(epsilon_evol,epsilon_tmp)
}



var_reg_f_evol = apply(reg_f_evol,2,var)
biais_reg_f_evol = (apply(reg_f_evol,2,mean)-f_x)^2
var_epsilon = rep(var(epsilon_evol),nbK)
erreur = var_reg_f_evol + biais_reg_f_evol + var_epsilon

print(biais_reg_f_evol)
print("===========")
print(var_reg_f_evol)
print("===========")
print(var_epsilon)
print("===========")
print(erreur)
print("===========")
pdf(file = paste(getwd(),"/graphs/graphs",1,".pdf",sep=""))

plot(biais_reg_f_evol,col="black",xlab="K",ylab="Biais", main="Biais en fonction de K")
plot(var_reg_f_evol,col="black",xlab="K",ylab="Variance", main="Variance en fonction de K")	
plot(var_epsilon,col="black",xlab="K",ylab="Var epsilon", main="Variance d'Epsilon en fonction de K")	
plot(erreur,col="black",xlab="K",ylab="MSE", main="MSE en fonction de K")	

dev.off()