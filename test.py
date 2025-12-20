# ==========================
# 1. Imports
# ==========================
import time
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================
# 2. Données : Iris
# ==========================
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# 3. Modèle de base : Random Forest
# ==========================
rf = RandomForestClassifier(random_state=42)

# ==========================
# 4. GRID SEARCH
#    - grille “petite” mais exhaustive
# ==========================
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5]
}
# Nombre de combinaisons = 3 * 3 * 2 * 2 = 36

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
grid_test_acc = accuracy_score(y_test, y_pred_grid)

print("========== GRID SEARCH ==========")
print("Meilleurs hyperparamètres (Grid) :")
print(grid_search.best_params_)
print("Meilleur score CV (Grid) :", grid_search.best_score_)
print("Accuracy test (Grid)      :", grid_test_acc)
print("Temps d'exécution (Grid)  : {:.3f} s".format(grid_time))


# ==========================
# 5. RANDOM SEARCH
#    - tirage aléatoire dans des distributions
# ==========================
# Ici on donne des distributions au lieu de listes exhaustives
param_dist = {
    "n_estimators": np.arange(50, 301, 10),      # 50 à 300
    "max_depth": [None] + list(range(3, 16)),    # None ou 3 à 15
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,              # 20 tirages aléatoires
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
random_test_acc = accuracy_score(y_test, y_pred_random)

print("\n========== RANDOM SEARCH ==========")
print("Meilleurs hyperparamètres (Random) :")
print(random_search.best_params_)
print("Meilleur score CV (Random) :", random_search.best_score_)
print("Accuracy test (Random)      :", random_test_acc)
print("Temps d'exécution (Random)  : {:.3f} s".format(random_time))


# ==========================
# 6. COMPARAISON RÉSUMÉE
# ==========================
print("\n========== COMPARAISON GRID vs RANDOM ==========")
print("GridSearchCV  -> meilleur score CV : {:.4f}, test acc : {:.4f}, temps : {:.3f} s"
      .format(grid_search.best_score_, grid_test_acc, grid_time))
print("RandomizedSearchCV -> meilleur score CV : {:.4f}, test acc : {:.4f}, temps : {:.3f} s"
      .format(random_search.best_score_, random_test_acc, random_time))
