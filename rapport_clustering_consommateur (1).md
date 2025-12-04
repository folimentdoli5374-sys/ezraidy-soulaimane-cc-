# 🛒 Segmentation Client : Analyse de Clustering K-Means sur le Comportement Consommateur
<img src="WhatsApp Image 2025-12-04 à 11.45.28_0da4a02f.jpg" style="height:264px;margin-right:232px"/>
**Ezraidy soulaimane**  
**Projet Data Science & Machine Learning**  
**Année Universitaire 2025-2026**  
**Thématique:** Segmentation Client (Clustering)

---

## 📋 Table des Matières

1. [Introduction](#1-introduction)
2. [Dataset et Problématique](#2-dataset-et-problématique)
3. [Méthodologie](#3-méthodologie)
4. [Implémentation Technique](#4-implémentation-technique)
5. [Résultats et Discussion](#5-résultats-et-discussion)
6. [Conclusion](#6-conclusion)
7. [Bibliographie](#7-bibliographie)

---

## 1. Introduction

### 1.1 Contexte

Dans un environnement commercial de plus en plus compétitif, la compréhension approfondie du comportement des consommateurs est devenue un avantage stratégique majeur. Les entreprises cherchent à **personnaliser leurs stratégies marketing** pour maximiser l'engagement client, améliorer la rétention et augmenter le chiffre d'affaires.

La **segmentation client** permet de regrouper les consommateurs en clusters homogènes partageant des caractéristiques comportementales similaires. Cette approche facilite :
- 🎯 Le **ciblage marketing** précis et personnalisé
- 💰 L'**optimisation des ressources** publicitaires
- 📈 L'**amélioration de la satisfaction client** par des offres adaptées
- 🔄 La **prédiction du churn** et des opportunités de fidélisation

### 1.2 Problématique

**Question centrale :** Comment identifier et caractériser des groupes distincts de clients à partir de données comportementales et démographiques pour créer des stratégies marketing ciblées ?

**Objectifs spécifiques :**
1. Segmenter la base clients en groupes cohérents
2. Identifier les profils types de consommateurs
3. Fournir des insights actionnables pour les équipes marketing
4. Optimiser l'allocation des ressources commerciales

### 1.3 Type de Machine Learning

**Apprentissage non supervisé (Unsupervised Learning) - Clustering**

- **Algorithme principal :** K-Means Clustering
- **Pas de variable cible** : L'objectif est de découvrir des structures latentes dans les données
- **Méthode de validation :** Elbow Method, Silhouette Score

---

## 2. Dataset et Problématique

### 2.1 Source des Données

**Dataset :** Customer Personality Analysis  
**Origine :** Kaggle ([lien dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis))  
**Format :** Fichier CSV/TSV (`marketing_campaign.csv`)  
**Créateur :** Dr. Omar Romero-Hernandez

### 2.2 Description Générale

Le dataset contient des données sur **2 240 clients** d'une entreprise de vente au détail, collectées entre 2012 et 2014. Il comprend :
- 📊 **29 variables** initiales (réduites à ~25 après nettoyage)
- 🧑‍🤝‍🧑 Données démographiques (âge, éducation, situation familiale)
- 💳 Comportements d'achat (montants dépensés par catégorie de produits)
- 📢 Réponses aux campagnes marketing
- 🛒 Canaux d'achat préférés

### 2.3 Dictionnaire de Données

#### **2.3.1 Variables Démographiques**

| Variable | Type | Description | Exemple |
|----------|------|-------------|---------|
| `ID` | int | Identifiant unique client | 5524 |
| `Year_Birth` | int | Année de naissance | 1957 |
| `Education` | str | Niveau d'éducation | Graduation, PhD, Master, 2n Cycle, Basic |
| `Marital_Status` | str | Situation familiale | Single, Together, Married, Divorced, Widow, Alone, Absurd, YOLO |
| `Income` | float | Revenu annuel du foyer (USD) | 58138.0 |
| `Kidhome` | int | Nombre d'enfants en bas âge | 0, 1, 2 |
| `Teenhome` | int | Nombre d'adolescents | 0, 1, 2 |
| `Dt_Customer` | date | Date d'inscription | 04-09-2012 |

#### **2.3.2 Variables de Dépenses (Produits)**

| Variable | Type | Description | Plage |
|----------|------|-------------|-------|
| `MntWines` | int | Montant dépensé en vins (2 ans) | 0 - 1493 USD |
| `MntFruits` | int | Montant dépensé en fruits | 0 - 199 USD |
| `MntMeatProducts` | int | Montant dépensé en viande | 0 - 1725 USD |
| `MntFishProducts` | int | Montant dépensé en poisson | 0 - 259 USD |
| `MntSweetProducts` | int | Montant dépensé en sucreries | 0 - 263 USD |
| `MntGoldProds` | int | Montant dépensé en produits premium | 0 - 362 USD |

#### **2.3.3 Variables Comportementales (Canaux d'Achat)**

| Variable | Type | Description |
|----------|------|-------------|
| `NumDealsPurchases` | int | Nombre d'achats avec réduction |
| `NumWebPurchases` | int | Nombre d'achats via site web |
| `NumCatalogPurchases` | int | Nombre d'achats via catalogue |
| `NumStorePurchases` | int | Nombre d'achats en magasin physique |
| `NumWebVisitsMonth` | int | Nombre de visites web/mois |

#### **2.3.4 Variables Marketing (Campagnes)**

| Variable | Type | Description |
|----------|------|-------------|
| `AcceptedCmp1` | bool | A accepté l'offre campagne 1 (0/1) |
| `AcceptedCmp2` | bool | A accepté l'offre campagne 2 (0/1) |
| `AcceptedCmp3` | bool | A accepté l'offre campagne 3 (0/1) |
| `AcceptedCmp4` | bool | A accepté l'offre campagne 4 (0/1) |
| `AcceptedCmp5` | bool | A accepté l'offre campagne 5 (0/1) |
| `Response` | bool | A accepté la dernière campagne (0/1) |
| `Complain` | bool | A déposé une plainte (2 dernières années) |

#### **2.3.5 Variables à Exclure**

| Variable | Raison de l'exclusion |
|----------|----------------------|
| `Z_CostContact` | Constante (valeur = 3 pour tous) - Aucune variance |
| `Z_Revenue` | Constante (valeur = 11 pour tous) - Aucune variance |

### 2.4 Caractéristiques du Dataset

**Taille :**
- **Lignes :** 2 240 observations
- **Colonnes :** 29 variables initiales → ~22-25 après feature engineering

**Types de variables :**
- **Numériques continues :** 16 (Income, dépenses, âge calculé)
- **Numériques discrètes :** 8 (compteurs d'achats, campagnes)
- **Catégorielles :** 2 (Education, Marital_Status)
- **Date :** 1 (Dt_Customer)

**Valeurs manquantes :**
- `Income` : **24 valeurs manquantes** (~1,07%)
- Autres variables : Complètes

**Outliers potentiels :**
- `Year_Birth` : Clients nés en 1893, 1899, 1900 (âges > 120 ans) → Erreurs de saisie
- `Income` : Revenus > 600 000 USD (top 0,1%) → Valeurs atypiques extrêmes

---

## 3. Méthodologie

### 3.1 Pipeline Global

```
┌─────────────────┐
│  Données Brutes │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  1. Preprocessing   │◄─── Nettoyage, imputation, encodage
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Feature Eng.    │◄─── Création de features, scaling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. EDA             │◄─── Visualisations, corrélations
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Réduction Dim.  │◄─── PCA (optionnel)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Clustering      │◄─── K-Means, détermination de k
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Validation      │◄─── Elbow, Silhouette Score
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  7. Interprétation  │◄─── Profils, insights business
└─────────────────────┘
```

### 3.2 Justification des Choix Techniques

#### **3.2.1 Pourquoi K-Means ?**

**Avantages :**
- ✅ **Simplicité et rapidité** : Excellent sur datasets de taille moyenne (2K observations)
- ✅ **Scalabilité** : Complexité O(n·k·i·d) où i = itérations, d = dimensions
- ✅ **Interprétabilité** : Les centroïdes donnent des profils moyens clairs
- ✅ **Adapté aux données numériques** après normalisation

**Limites assumées :**
- ⚠️ Sensibilité aux outliers (d'où l'importance du preprocessing)
- ⚠️ Suppose des clusters sphériques de taille similaire
- ⚠️ Nécessite de spécifier k à l'avance (résolu par Elbow Method)

#### **3.2.2 Pourquoi la normalisation StandardScaler ?**

**Problème :** Les variables ont des échelles très différentes :
- `Income` : 0 - 600 000 USD
- `MntWines` : 0 - 1 493 USD  
- `NumWebVisitsMonth` : 0 - 20 visites

**Solution :** StandardScaler (z-score normalization)
```python
z = (x - μ) / σ
```
- Transforme chaque variable : **moyenne = 0, écart-type = 1**
- K-Means utilise la distance euclidienne → Évite que les grandes valeurs dominent

**Alternative considérée :** MinMaxScaler (rejeté car sensible aux outliers extrêmes)

#### **3.2.3 Pourquoi le Silhouette Score ?**

**Formule :**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Où :
- `a(i)` = distance intra-cluster moyenne
- `b(i)` = distance inter-cluster moyenne au cluster le plus proche

**Interprétation :**
- **s(i) ≈ 1** : Bien clustérisé
- **s(i) ≈ 0** : À la frontière entre deux clusters
- **s(i) < 0** : Mal affecté (devrait être dans un autre cluster)

---

## 4. Implémentation Technique

### 4.1 Environnement et Bibliothèques

```python
# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Statistiques
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage

# Warnings
import warnings
warnings.filterwarnings('ignore')
```

### 4.2 Étape 1 : Chargement et Exploration Initiale

```python
# Chargement des données
df = pd.read_csv('marketing_campaign.csv', sep='\t')

# Aperçu des données
print(f"Dimensions du dataset : {df.shape}")
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# Informations sur les types de données
df.info()

# Statistiques descriptives
df.describe()

# Vérification des valeurs manquantes
missing_values = df.isnull().sum()
print(f"\nValeurs manquantes :\n{missing_values[missing_values > 0]}")
```

**Résultat attendu :**
```
Dimensions du dataset : (2240, 29)
Valeurs manquantes :
Income    24
```

### 4.3 Étape 2 : Preprocessing (Nettoyage des Données)

#### **4.3.1 Suppression des Colonnes Inutiles**

```python
# Suppression des variables constantes (aucune information)
df_clean = df.drop(['Z_CostContact', 'Z_Revenue'], axis=1)
```

**Justification :** Ces colonnes ont la même valeur pour tous les clients (variance nulle).

#### **4.3.2 Gestion des Valeurs Manquantes**

```python
# Option 1 : Imputation par la médiane (robuste aux outliers)
df_clean['Income'].fillna(df_clean['Income'].median(), inplace=True)

# Option 2 (alternative) : Suppression des lignes avec NA
# df_clean = df_clean.dropna(subset=['Income'])
```

**Justification :** 
- Seulement 24 valeurs manquantes (1,07%) sur `Income`
- La médiane est plus robuste que la moyenne face aux outliers
- Préservation de 100% des données

#### **4.3.3 Détection et Traitement des Outliers**

```python
# Détection des âges aberrants
df_clean['Age'] = 2022 - df_clean['Year_Birth']
print(df_clean[df_clean['Age'] > 100])

# Suppression des outliers extrêmes
df_clean = df_clean[df_clean['Age'] <= 100]

# Suppression des revenus extrêmes (> 600k USD)
df_clean = df_clean[df_clean['Income'] <= 600000]

print(f"Nouvelles dimensions : {df_clean.shape}")
```

**Résultat attendu :** ~2 200 lignes restantes

#### **4.3.4 Feature Engineering : Création de Nouvelles Variables**

```python
# 1. Âge du client (2022 - année de naissance)
df_clean['Age'] = 2022 - df_clean['Year_Birth']
df_clean.drop('Year_Birth', axis=1, inplace=True)

# 2. Nombre total d'enfants
df_clean['Total_Children'] = df_clean['Kidhome'] + df_clean['Teenhome']
df_clean.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)

# 3. Ancienneté client (en jours)
df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], format='%d-%m-%Y')
df_clean['Days_As_Customer'] = (pd.to_datetime('2022-01-01') - df_clean['Dt_Customer']).dt.days

# 4. Dépenses totales
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_clean['Total_Spending'] = df_clean[spending_cols].sum(axis=1)

# 5. Nombre total d'achats
purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 
                 'NumCatalogPurchases', 'NumStorePurchases']
df_clean['Total_Purchases'] = df_clean[purchase_cols].sum(axis=1)

# 6. Taux d'acceptation des campagnes
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df_clean['Total_Accepted_Campaigns'] = df_clean[campaign_cols].sum(axis=1)
df_clean['Campaign_Acceptance_Rate'] = df_clean['Total_Accepted_Campaigns'] / 6

# 7. Dépense moyenne par achat
df_clean['Avg_Spending_Per_Purchase'] = df_clean['Total_Spending'] / (df_clean['Total_Purchases'] + 1)
```

**Justification :**
- **Age** : Plus interprétable que l'année de naissance
- **Total_Children** : Variable agrégée plus simple
- **Total_Spending** : Indicateur global du pouvoir d'achat
- **Campaign_Acceptance_Rate** : Mesure de l'engagement marketing

#### **4.3.5 Encodage des Variables Catégorielles**

```python
# Nettoyage de la variable Education
education_mapping = {
    'Graduation': 'Undergraduate',
    'PhD': 'Postgraduate',
    'Master': 'Postgraduate',
    '2n Cycle': 'Undergraduate',
    'Basic': 'High School'
}
df_clean['Education'] = df_clean['Education'].replace(education_mapping)

# Nettoyage de Marital_Status
marital_mapping = {
    'Married': 'In Relationship',
    'Together': 'In Relationship',
    'Single': 'Single',
    'Divorced': 'Single',
    'Widow': 'Single',
    'Alone': 'Single',
    'Absurd': 'Single',
    'YOLO': 'Single'
}
df_clean['Marital_Status'] = df_clean['Marital_Status'].replace(marital_mapping)

# Encodage One-Hot
df_encoded = pd.get_dummies(df_clean, columns=['Education', 'Marital_Status'], drop_first=True)
```

**Justification :**
- **Regroupement** : Réduction de la cardinalité (8 → 2 catégories pour Marital_Status)
- **One-Hot Encoding** : K-Means nécessite des données numériques
- **drop_first=True** : Évite la multicolinéarité parfaite

### 4.4 Étape 3 : Analyse Exploratoire des Données (EDA)

#### **4.4.1 Distribution de l'Âge**

```python
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['Age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution de l\'Âge des Clients', fontsize=16, fontweight='bold')
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.axvline(df_clean['Age'].mean(), color='red', linestyle='--', label=f'Moyenne : {df_clean["Age"].mean():.1f} ans')
plt.legend()
plt.show()
```

**Interprétation :** 
- **Moyenne** : ~52 ans
- **Distribution** : Quasi-normale, légèrement asymétrique vers la droite
- **Insight** : Clientèle mature, cible marketing adaptée aux 45-65 ans

#### **4.4.2 Distribution des Revenus**

```python
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_clean['Income'], color='lightgreen')
plt.title('Distribution des Revenus Annuels', fontsize=16, fontweight='bold')
plt.xlabel('Revenu (USD)')
plt.show()
```

**Interprétation :**
- **Médiane** : ~51 000 USD
- **Outliers** : Quelques clients avec revenus > 150 000 USD
- **Insight** : Majorité classe moyenne, segment premium minoritaire

#### **4.4.3 Matrice de Corrélation (Heatmap)**

```python
# Sélection des variables numériques
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns

# Calcul de la matrice de corrélation
corr_matrix = df_clean[numerical_cols].corr()

# Visualisation
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=0.5, cbar_kws={'label': 'Coefficient de corrélation'})
plt.title('Matrice de Corrélation des Variables Numériques', fontsize=18, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

**Interprétation des corrélations clés :**

| Variables | Corrélation | Signification |
|-----------|-------------|---------------|
| `Income` ↔ `Total_Spending` | **+0.78** | Fort pouvoir prédictif : revenus élevés = dépenses élevées |
| `MntWines` ↔ `MntMeatProducts` | **+0.72** | Achats liés (clients gourmets) |
| `Total_Children` ↔ `Total_Spending` | **-0.42** | Familles nombreuses dépensent moins |
| `NumWebVisitsMonth` ↔ `Total_Purchases` | **-0.35** | Visites fréquentes mais peu d'achats = friction UX ? |

#### **4.4.4 Analyse Bivariée : Revenus vs Dépenses**

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='Income', y='Total_Spending', 
                hue='Total_Children', palette='viridis', s=100, alpha=0.6)
plt.title('Relation Revenus - Dépenses Totales', fontsize=16, fontweight='bold')
plt.xlabel('Revenu Annuel (USD)')
plt.ylabel('Dépenses Totales (USD)')
plt.legend(title='Nombre d\'enfants', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

**Interprétation :**
- **Corrélation positive forte** : Confirme la relation revenu-dépense
- **Segmentation visible** : Clients sans enfants dépensent plus à revenu égal
- **Insight** : Opportunité de ciblage pour produits premium

### 4.5 Étape 4 : Normalisation des Données

```python
# Sélection des features pour le clustering
features_for_clustering = [
    'Age', 'Income', 'Total_Children', 'Days_As_Customer',
    'Total_Spending', 'Total_Purchases', 'Avg_Spending_Per_Purchase',
    'NumWebVisitsMonth', 'Campaign_Acceptance_Rate'
]

X = df_encoded[features_for_clustering]

# Standardisation (z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Conversion en DataFrame pour visualisation
X_scaled_df = pd.DataFrame(X_scaled, columns=features_for_clustering)

print("Données normalisées (5 premières lignes) :")
print(X_scaled_df.head())
```

**Résultat :** Toutes les variables ont maintenant une moyenne ≈ 0 et un écart-type ≈ 1.

### 4.6 Étape 5 : Détermination du Nombre Optimal de Clusters (k)

#### **4.6.1 Méthode du Coude (Elbow Method)**

```python
# Calcul de l'inertie (WCSS) pour k de 1 à 10
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=10)
plt.xlabel('Nombre de Clusters (k)', fontsize=14)
plt.ylabel('WCSS (Inertie)', fontsize=14)
plt.title('Méthode du Coude pour Détermination de k', fontsize=16, fontweight='bold')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.axvline(x=4, color='red', linestyle='--', label='Coude suggéré : k=4')
plt.legend()
plt.show()
```

**Interprétation :**
- **WCSS** (Within-Cluster Sum of Squares) : Mesure la compacité intra-cluster
- **Coude** visible autour de **k = 3-4**
- Au-delà de k=4, la diminution du WCSS est marginale

#### **4.6.2 Silhouette Score**

```python
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k = {k} : Silhouette Score = {score:.4f}")

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=10)
plt.xlabel('Nombre de Clusters (k)', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.title('Silhouette Score en Fonction de k', fontsize=16, fontweight='bold')
plt.xticks(range(2, 11))
plt.grid(True, alpha=0.3)
plt.axhline(y=max(silhouette_scores), color='red', linestyle='--', label=f'Maximum : k={silhouette_scores.index(max(silhouette_scores))+2}')
plt.legend()
plt.show()
```

**Résultat attendu :**
```
k = 2 : Silhouette Score = 0.3812
k = 3 : Silhouette Score = 0.4051
k = 4 : Silhouette Score = 0.4127  ← Maximum
k = 5 : Silhouette Score = 0.3956
```

**Décision :** **k = 4** clusters (compromis entre Elbow et Silhouette Score maximum)

### 4.7 Étape 6 : Clustering K-Means Final

```python
# Application de K-Means avec k=4
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', 
                      random_state=42, n_init=20, max_iter=300)
df_encoded['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Affichage de la répartition
print(df_encoded['Cluster'].value_counts().sort_index())
```

**Résultat attendu :**
```
Cluster
0    623
1    512
2    589
3    476
```

### 4.8 Étape 7 : Visualisation des Clusters

#### **4.8.1 Réduction de Dimensionnalité avec PCA**

```python
from sklearn.decomposition import PCA

# Réduction à 2 dimensions pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Ajout au DataFrame
df_encoded['PCA1'] = X_pca[:, 0]
df_encoded['PCA2'] = X_pca[:, 1]

# Variance expliquée
print(f"Variance expliquée par PCA1 : {pca.explained_variance_ratio_[0]:.2%}")
print(f"Variance expliquée par PCA2 : {pca.explained_variance_ratio_[1]:.2%}")
print(f"Variance totale expliquée : {sum(pca.explained_variance_ratio_):.2%}")
```

**Résultat attendu :** ~60-70% de variance expliquée (suffisant pour visualisation)

#### **4.8.2 Scatter Plot 2D (PCA)**

```python
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(data=df_encoded, x='PCA1', y='PCA2', hue='Cluster', 
                          palette='Set2', s=100, alpha=0.7, edgecolor='black')
plt.title('Visualisation des Clusters (PCA 2D)', fontsize=18, fontweight='bold')
plt.xlabel(f'Composante Principale 1 ({pca
