library(glmnet)
library(klaR)

set.seed(2025)

df_reg <- read.table("TP5_a25_reg_app.txt", header = TRUE, row.names = 1)
df_cla <- read.table("TP5_a25_clas_app.txt", header = TRUE, row.names = 1)

y_reg <- df_reg$y
X_reg <- as.matrix(df_reg[, -which(names(df_reg) == "y")])

cv_model <- cv.glmnet(X_reg, y_reg, alpha = 0.5)
best_lambda <- cv_model$lambda.min

final_model_reg <- glmnet(X_reg, y_reg, alpha = 0.5, lambda = best_lambda)

y_cla <- factor(df_cla$y)
X_cla <- df_cla[, -which(names(df_cla) == "y")]

final_model_cla <- klaR::rda(y ~ ., data = data.frame(X_cla, y = y_cla))

regresseur <- function(test_set) {
  library(glmnet)
  
  X_matrix <- as.matrix(test_set)
  
  pred <- predict(final_model_reg, newx = X_matrix)
  
  return(as.numeric(pred))
}

classifieur <- function(test_set) {
  library(klaR)
  
  pred <- predict(final_model_cla, newdata = test_set)
  
  return(pred$class)
}

save(
  "regresseur",       # La fonction de régression
  "classifieur",      # La fonction de classification
  "final_model_reg",  # L'objet modèle Elastic Net
  "final_model_cla",  # L'objet modèle RDA
  file = "env.Rdata"
)

cat("Fichier env.Rdata généré avec succès !")