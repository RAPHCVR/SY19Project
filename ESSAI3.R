library(glmnet)
library(klaR)
library(MASS)

set.seed(2025)

# 1. Chargement des données
read_data <- function(filename) {
  if (file.exists(filename)) return(read.table(filename, header = TRUE, row.names = 1))
  f_tp <- paste0("TP5_", filename)
  if (file.exists(f_tp)) return(read.table(f_tp, header = TRUE, row.names = 1))
  stop("Erreur : Fichiers de données introuvables.")
}

df_reg <- read_data("a25_reg_app.txt")
df_cla <- read_data("a25_clas_app.txt")

cat("Données chargées.\n")

# 2. REGRESSION : Elastic Net avec sélection automatique de Alpha
cat("\n--- Optimisation Régression (Elastic Net) ---\n")

y_reg <- df_reg$y
X_reg <- as.matrix(df_reg[, -which(names(df_reg) == "y")])

# On teste différentes valeurs de alpha entre 0 (Ridge) et 1 (Lasso)
alphas <- seq(0, 1, by = 0.1)
best_mse <- Inf
best_alpha <- 0
best_model_cv <- NULL

for (a in alphas) {
  # Validation croisée pour trouver le meilleur lambda pour cet alpha
  cv <- cv.glmnet(X_reg, y_reg, alpha = a, type.measure = "mse", nfolds = 10)
  
  min_mse <- min(cv$cvm)
  
  # Si on trouve une meilleure erreur, on garde cet alpha
  if (min_mse < best_mse) {
    best_mse <- min_mse
    best_alpha <- a
    best_model_cv <- cv
  }
}

cat(sprintf("Meilleur Alpha trouvé : %.1f (MSE CV : %.2f)\n", best_alpha, best_mse))
cat("Construction du modèle final de régression...\n")

# On construit le modèle final avec le meilleur alpha et son meilleur lambda
final_model_reg <- glmnet(X_reg, y_reg, alpha = best_alpha, lambda = best_model_cv$lambda.min)


# 3. CLASSIFICATION : RDA avec Grid Search Standard
cat("\n--- Optimisation Classification (RDA) ---\n")

y_cla <- factor(df_cla$y)
X_cla <- df_cla[, -which(names(df_cla) == "y")]

# Grille de recherche
grid_rda <- expand.grid(
  gamma = seq(0, 1, by = 0.1), 
  lambda = seq(0, 1, by = 0.1)
)
grid_rda$accuracy <- NA

# Validation croisée K-Fold
K <- 5
folds <- sample(rep(1:K, length.out = nrow(X_cla)))

cat("Recherche des meilleurs paramètres RDA (Grid Search)...\n")

for (i in 1:nrow(grid_rda)) {
  g <- grid_rda$gamma[i]
  l <- grid_rda$lambda[i]
  scores <- numeric(K)
  
  for (k in 1:K) {
    # Séparation app/val
    idx_val <- which(folds == k)
    X_train <- X_cla[-idx_val, ]
    y_train <- y_cla[-idx_val]
    X_val <- X_cla[idx_val, ]
    y_val <- y_cla[idx_val]
    
    # Entraînement et Prédiction
    model <- klaR::rda(x = X_train, grouping = y_train, gamma = g, lambda = l)
    pred <- predict(model, X_val)$class
    
    # Calcul du taux de bon classement
    scores[k] <- mean(pred == y_val)
  }
  grid_rda$accuracy[i] <- mean(scores)
}

# Sélection
best_idx <- which.max(grid_rda$accuracy)
best_params <- grid_rda[best_idx, ]

cat(sprintf("Meilleurs paramètres : Gamma = %.1f, Lambda = %.1f (Accuracy : %.4f)\n", 
            best_params$gamma, best_params$lambda, best_params$accuracy))

# Modèle final
final_model_cla <- klaR::rda(y ~ ., data = data.frame(X_cla, y = y_cla),
                             gamma = best_params$gamma, 
                             lambda = best_params$lambda)


# 4. SAUVEGARDE

regresseur <- function(test_set) {
  library(glmnet)
  # Conversion en matrice nécessaire pour glmnet
  X_mat <- as.matrix(test_set)
  pred <- predict(final_model_reg, newx = X_mat)
  return(as.numeric(pred))
}

classifieur <- function(test_set) {
  library(klaR)
  library(MASS)
  pred <- predict(final_model_cla, newdata = test_set)
  return(pred$class)
}

save("regresseur", "classifieur", "final_model_reg", "final_model_cla", file = "env.Rdata")
cat("\nFichier env.Rdata généré avec succès.\n")