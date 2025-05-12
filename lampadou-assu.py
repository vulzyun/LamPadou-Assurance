import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import pickle

# 1. Importation des données
print("1. IMPORTATION DES DONNÉES")
df = pd.read_csv('car_insurance.csv')

print("\nDimensions du jeu de données:")
print(df.shape)

print("\nAffichage des 5 premières lignes:")
print(df.head())

print("\nInformations sur les types de données:")
print(df.dtypes)

print("\nStatistiques descriptives:")
print(df.describe())

print("\nDonnées manquantes par colonne:")
print(df.isna().sum())

# Visualisation des distributions
plt.figure(figsize=(15, 10))
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig('histograms.png')
plt.close()
print("\nHistogrammes des variables numériques enregistrés dans 'histograms.png'")

# Identifier la variable cible
print("\nVariable cible (outcome):")
print(df['outcome'].value_counts())
print("\nObjectif de classification: Prédire si un client va souscrire à une assurance automobile (1) ou non (0)")

# 2. Préparation des données
print("\n2. PRÉPARATION DES DONNÉES")

def changedf(df):
    df_copy = df.copy()
    children_median = df_copy[' children'].median()
    mileage_median = df_copy[' annual_mileage'].median()
    speeding_median = df_copy['speeding_violations'].median()
    accidents_median = df_copy['past_accidents'].median()
    
    for column in df_copy.columns:
        if df_copy[column].dtype in ['int64', 'float64']:
            median_value = df_copy[column].median()
            df_copy[column] = df_copy[column].fillna(median_value)
        else:
            mode_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else None
            df_copy[column] = df_copy[column].fillna(mode_value)
        
        if column == ' children':
            df_copy[column] = df_copy[column].fillna(0)
            df_copy.loc[df_copy[column] > 12, column] = children_median
        
        elif column == ' annual_mileage':
            df_copy.loc[df_copy[column] > 20000, column] = mileage_median
        
        elif column == 'speeding_violations':
            df_copy.loc[df_copy[column] > 12, column] = speeding_median
        
        elif column == 'past_accidents':
            df_copy.loc[df_copy[column] > 12, column] = accidents_median
    
    return df_copy

# Nettoyer les données avec la fonction existante
df_clean = changedf(df)

# Convertir les variables catégorielles
print("\nConversion des variables catégorielles en variables numériques:")
categorical_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'object']
print(f"Variables catégorielles détectées: {categorical_cols}")

le = LabelEncoder()
for col in categorical_cols:
    df_clean[col] = le.fit_transform(df_clean[col])

print("\nVérification des valeurs manquantes après traitement:")
print(df_clean.isna().sum())

# 3. Normalisation des données
print("\n3. NORMALISATION DES DONNÉES")

# Séparation des entrées (X) et sorties (y)
X = df_clean.drop('outcome', axis=1)
y = df_clean['outcome']

# Normalisation avec StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFormat des données après normalisation: X_scaled shape: {X_scaled.shape}, y shape: {y.shape}")

# 4. Recherche de corrélations
print("\n4. RECHERCHE DE CORRÉLATIONS")

# Calcul de la matrice de corrélation
corr_matrix = df_clean.corr()

print("\nCorrélations avec la variable cible (outcome):")
target_correlations = corr_matrix['outcome'].sort_values(ascending=False)
print(target_correlations)

# Visualisation des corrélations
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.savefig('correlation_matrix.png')
plt.close()
print("\nMatrice de corrélation enregistrée dans 'correlation_matrix.png'")

# Sélection des variables les plus corrélées avec la cible
most_correlated = target_correlations[1:6]  # Top 5 variables (en excluant la cible elle-même)
print("\nTop 5 variables les plus corrélées avec la cible:")
print(most_correlated)

# Visualisation des nuages de points pour les variables les plus corrélées
pd.plotting.scatter_matrix(df_clean[list(most_correlated.index) + ['outcome']], figsize=(15, 15))
plt.savefig('scatter_matrix.png')
plt.close()
print("\nNuage de points enregistré dans 'scatter_matrix.png'")

# 5. Extraction des jeux d'apprentissage et de test
print("\n5. EXTRACTION DES JEUX D'APPRENTISSAGE ET DE TEST")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"\nTaille du jeu d'apprentissage: {X_train.shape[0]} échantillons ({X_train.shape[0]/X_scaled.shape[0]:.1%})")
print(f"Taille du jeu de test: {X_test.shape[0]} échantillons ({X_test.shape[0]/X_scaled.shape[0]:.1%})")

# 6. Entraînement du modèle de régression logistique
print("\n6. ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION LOGISTIQUE")

# Création et entraînement du modèle
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

print("\nModèle de régression logistique entraîné.")
print(f"Coefficients: {log_reg.coef_}")
print(f"Ordonnée à l'origine: {log_reg.intercept_}")

# 7. Évaluation du modèle
print("\n7. ÉVALUATION DU MODÈLE")

# Prédictions sur le jeu de test
y_pred = log_reg.predict(X_test)

# Calcul des métriques d'évaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMétriques d'évaluation:")
print(f"Précision (accuracy): {accuracy:.4f}")
print("\nMatrice de confusion:")
print(conf_matrix)
print("\nPrécision (precision): {:.4f}".format(precision))
print("Rappel (recall): {:.4f}".format(recall))
print("Score F1: {:.4f}".format(f1))

print("\nRapport de classification détaillé:")
print(classification_report(y_test, y_pred))

# 8. Amélioration de l'évaluation avec validation croisée
print("\n8. AMÉLIORATION DE L'ÉVALUATION AVEC VALIDATION CROISÉE")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=kfold)

print("\nScores de validation croisée:")
print(f"Scores individuels: {cv_scores}")
print(f"Score moyen: {cv_scores.mean():.4f}")
print(f"Écart-type: {cv_scores.std():.4f}")

# 9. Comparaison avec d'autres algorithmes
print("\n9. COMPARAISON AVEC D'AUTRES ALGORITHMES")

# Liste des classifieurs à comparer
classifiers = {
    'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
    'Perceptron': Perceptron(random_state=42),
    'K plus proches voisins (K=5)': KNeighborsClassifier(n_neighbors=5),
    'K plus proches voisins (K=10)': KNeighborsClassifier(n_neighbors=10)
}

# Évaluation des classifieurs
results = {}

for name, clf in classifiers.items():
    print(f"\nÉvaluation de {name}:")
    
    # Entraînement sur le jeu d'apprentissage
    clf.fit(X_train, y_train)
    
    # Prédictions et calcul de l'accuracy sur le jeu de test
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy sur le jeu de test: {accuracy:.4f}")
    
    # Validation croisée
    cv_scores = cross_val_score(clf, X_scaled, y, cv=kfold)
    print(f"Score moyen de validation croisée: {cv_scores.mean():.4f}")
    
    # Stockage des résultats
    results[name] = {
        'test_accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# Affichage des résultats comparatifs
print("\nRésultats comparatifs:")
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)

# Identification du meilleur modèle selon la validation croisée
best_model_name = results_df['cv_mean'].idxmax()
best_model = classifiers[best_model_name]
print(f"\nMeilleur modèle selon la validation croisée: {best_model_name}")

# Entraînement du meilleur modèle sur l'ensemble des données
best_model.fit(X_scaled, y)

# 10. Sauvegarde du modèle entraîné
print("\n10. SAUVEGARDE DU MODÈLE ENTRAÎNÉ")

# Sauvegarde du modèle avec pickle
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Sauvegarde du scaler pour prétraiter de nouvelles données
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("\nModèle et scaler sauvegardés avec succès!")

# Test de chargement et d'utilisation du modèle
print("\nTest de chargement et d'utilisation du modèle:")

with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Test avec un échantillon du jeu de test
sample_index = 0
sample_X = X.iloc[sample_index:sample_index+1]
sample_y = y.iloc[sample_index]

# Normalisation
sample_X_scaled = loaded_scaler.transform(sample_X)

# Prédiction
sample_pred = loaded_model.predict(sample_X_scaled)

print(f"\nTest sur l'échantillon {sample_index}:")
print(f"Caractéristiques: {sample_X.values}")
print(f"Classe réelle: {sample_y}")
print(f"Classe prédite: {sample_pred[0]}")

# Réponses aux questions sur la régression logistique
print("\nRéponses aux questions théoriques sur la régression logistique:")
print("1. Hypothèse sur la fonction logit: On suppose que le logarithme du rapport des vraisemblances (logit) est une fonction linéaire des variables d'entrée.")
print("2. Technique pour minimiser la fonction de coût: L'algorithme utilise l'optimisation par descente de gradient (ou ses variantes).")
print("3. Paramètres calculés pendant l'apprentissage: Les coefficients (poids) de chaque variable et l'ordonnée à l'origine (biais).")

print("\nAnalyse terminée avec succès!")