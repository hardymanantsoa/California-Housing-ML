# ============================================================
# 1. Importation des bibliothèques nécessaires
# ============================================================
# pandas & numpy : manipulation et calculs sur les données
# matplotlib & seaborn : visualisation des données
# scikit-learn : outils de Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Configuration globale des graphiques pour une meilleure lisibilité
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Bibliothèques importées avec succès")


# ============================================================
# 2. Chargement et inspection du jeu de données
# ============================================================

# Chargement du dataset California Housing sous forme de DataFrame
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# Affichage des dimensions du dataset
print(f"Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# Définition de la variable cible
print("\nVariable cible : MedHouseVal (valeur médiane du logement en centaines de milliers de $)")

# Description des variables explicatives
print("\nDescription des variables :")
print("-" * 60)

descriptions = {
    'MedInc': 'Revenu médian du quartier',
    'HouseAge': 'Âge médian des logements',
    'AveRooms': 'Nombre moyen de pièces par logement',
    'AveBedrms': 'Nombre moyen de chambres par logement',
    'Population': 'Population du quartier',
    'AveOccup': 'Nombre moyen d\'occupants par logement',
    'Latitude': 'Latitude géographique',
    'Longitude': 'Longitude géographique',
    'MedHouseVal': 'Prix médian du logement (variable cible)'
}

for col, desc in descriptions.items():
    print(f"{col:15s} → {desc}")


# ============================================================
# 3. Aperçu et compréhension des données
# ============================================================

# Affichage des premières lignes pour avoir un aperçu rapide
print("\nAperçu des premières lignes du dataset :")
print(df.head())

# Informations générales : types des variables, valeurs non nulles
print("\nInformations générales sur le dataset :")
df.info()

# Statistiques descriptives : moyenne, écart-type, min, max, quartiles
print("\nStatistiques descriptives du dataset :")
print(df.describe())


# ============================================================
# 4. Nettoyage des données
# ============================================================

# 4.1 Vérification des valeurs manquantes
# Un dataset propre ne doit pas contenir de NaN
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# 4.2 Vérification et suppression des doublons
nb_doublons = df.duplicated().sum()
print(f"\nNombre de lignes dupliquées : {nb_doublons}")
df.drop_duplicates(inplace=True)

# 4.3 Détection visuelle des valeurs aberrantes (outliers)
# Les boxplots permettent d'identifier les valeurs extrêmes
colonnes_outliers = ['AveRooms', 'AveBedrms', 'AveOccup']

plt.figure(figsize=(12, 4))
for i, col in enumerate(colonnes_outliers, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# 4.4 Suppression des outliers extrêmes
# Ces valeurs peuvent fausser l'entraînement du modèle
print("\nDimensions avant suppression des outliers :", df.shape)

df = df[
    (df['AveRooms'] < 50) &
    (df['AveBedrms'] < 20) &
    (df['AveOccup'] < 20)
]

print("Dimensions après suppression des outliers :", df.shape)


# ============================================================
# 5. Analyse Exploratoire des Données (EDA)
# ============================================================

# 5.1 Distribution de la variable cible
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title("Distribution du prix médian des logements (MedHouseVal)")
plt.xlabel("Prix médian (en centaines de milliers de $)")
plt.ylabel("Fréquence")
plt.show()

sns.boxplot(x=df['MedHouseVal'])
plt.title("Boxplot du prix médian des logements")
plt.show()

# 5.2 Matrice de corrélation
# Permet d'identifier les relations linéaires entre les variables
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation des variables")
plt.show()

# 5.3 Relation entre le revenu médian et le prix des logements
sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'], alpha=0.5)
plt.title("Relation entre le revenu médian et le prix des logements")
plt.xlabel("Revenu médian (MedInc)")
plt.ylabel("Prix médian (MedHouseVal)")
plt.show()

# 5.4 Visualisation géographique des prix
plt.scatter(
    df['Longitude'],
    df['Latitude'],
    c=df['MedHouseVal'],
    cmap='viridis',
    s=10
)
plt.colorbar(label="Prix médian des logements")
plt.title("Répartition géographique des prix des logements en Californie")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# ============================================================
# 6. Ingénierie des variables
# ============================================================

# Création de nouvelles variables explicatives
# Ces variables peuvent améliorer les performances du modèle

df['PiecesParChambre'] = df['AveRooms'] / df['AveBedrms']
df['PopParLogement'] = df['Population'] / df['AveOccup']

print("\nAperçu des nouvelles variables créées :")
print(df[['PiecesParChambre', 'PopParLogement']].head())


# ============================================================
# 7. Séparation des données (Train / Test)
# ============================================================

# y : variable cible
# X : variables explicatives
y = df['MedHouseVal']
X = df.drop(columns=['MedHouseVal'])

# 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDimensions des ensembles :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)


# ============================================================
# 8. Construction du modèle de régression linéaire
# ============================================================

# Le modèle cherche à apprendre la relation :
# ŷ = β0 + β1x1 + β2x2 + ... + βnxn

model = LinearRegression()
model.fit(X_train, y_train)

print("Ordonnée à l'origine (β0) :", model.intercept_)

# Analyse des coefficients
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nCoefficients des variables :")
print(coefficients)

sns.barplot(x='Coefficient', y='Variable', data=coefficients)
plt.title("Impact des variables sur le prix médian des logements")
plt.show()


# ============================================================
# 9. Prédiction et évaluation du modèle
# ============================================================

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul des métriques d'évaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Évaluation du modèle :")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# Comparaison valeurs réelles vs prédites
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Valeurs réelles vs Valeurs prédites")
plt.show()

# Analyse des résidus
residus = y_test - y_pred

sns.histplot(residus, kde=True)
plt.title("Distribution des résidus")
plt.show()

plt.scatter(y_pred, residus, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Résidus vs Valeurs prédites")
plt.show()


# ============================================================
# 10. Export des résultats
# ============================================================

# Création d'un tableau récapitulatif des prédictions
resultats = X_test.copy()
resultats['Prix_reel'] = y_test
resultats['Prix_pred'] = y_pred
resultats['Residus'] = residus
resultats['Erreur_pct'] = (residus / y_test) * 100

print(resultats.head())

# Exporter le DataFrame en fichier CSV
resultats.to_csv("resultats_regression.csv", index=False)
print("\nRésultats exportés avec succès dans 'resultats_regression.csv'")
