# Projet scoring PD 

## Contexte

Objectif : construire un modèle de scoring PD complet, de la segmentation amont
à la risk quantification, sur la clientèle Retail.


## Règle absolue — confidentialité des données

**Interdiction formelle de transmettre les données à un outil IA**,
y compris à toi . Cela signifie concrètement :
- Ne jamais lire, ouvrir, inspecter ou résumer le contenu des fichiers sous
  `data/raw/` ou `data/processed/` s'ils contiennent des données réelles.
- Ne jamais mentionner explicitement qu'il s'agit de données dans le code,
  les commentaires, ou les échanges.
- Le code que tu écris doit être **générique et testable sur des données
  simulées/fictives** (simulation  d'un
  `df` avec `numpy.random`), jamais en lisant les vraies données pour en
  déduire la structure.
- Si une tâche semble nécessiter d'inspecter les données réelles, demande
  confirmation explicite avant toute lecture — par défaut, refuse.

## Méthode de travail

Le projet sera repris **étape par étape** avec moi l'utilisateur, pas construit
d'un bloc. Ce fichier sert de mémoire du plan global, pas de spécification à exécuter automatiquement. Attendre une demande
explicite avant d'implémenter une étape donnée. Utiliser la terminologie du
cours telle quelle (WOE, IV, CHR, LRA, PDO, MoC...) plutôt que d'improviser
un vocabulaire différent.

## Architecture du projet

```
Projet-scoring/
├── CLAUDE.md
├── config.json                    # cibles, seuils, colonnes id, params modèles
├── data/
│   ├── raw/                       # jamais lu par l'agent (données réelles)
│   └── processed/                 # jamais lu par l'agent (données réelles)
├── utils/
│   ├── segmentation.py            # découpage amont de la population
│   ├── exploration.py             # associations et redondances (ch. 3)
│   ├── binning.py                 # discrétisation, WOE/IV (ch. 4)
│   ├── modele.py                  # régression logistique, modèle concurrent, sélection (ch. 5)
│   ├── scorecard.py               # score → points, PDO (ch. 6)
│   ├── chr.py                     # mise en classe, CHR (ch. 7)
│   └── calibrage.py               # calibrage, LRA, MoC (ch. 8)
├── outputs/
│   ├── figures/
│   └── scorecard/
├── 00_segmentation_exploration.ipynb
├── 01_differenciation.ipynb       
└── 02_calibration.ipynb           
```

Principe : les notebooks sont des rapports lisibles (texte, graphiques,
justification des décisions) qui appellent les fonctions des `.py` — pas de
logique lourde directement dans les cellules.

## Plan complet du cours (référence exhaustive)

### Introduction — Usages et finalités du score
- Qu'est-ce qu'un score, à quoi ça sert
- Données disponibles selon le contexte
- Panorama des usages par typologie
- Distinction structurante : **risk differentiation** (ordonnancement, mise
  en classe) vs **risk quantification** (calibrage en probabilité)

### 1. Fondements théoriques du scoring
- 1.1 Formalisation : classification binaire, probabilité vs classification pure
- 1.2 Notions de risque : population, target, fenêtres temporelles
- 1.3 Échantillonnage et volumétrie (rééquilibrage, correction de l'intercept)
- 1.4 Segmentation de la population : modèle unique vs par segment
- 1.5 Découpage des données : apprentissage, validation, test et Out-of-Time

### 2. Préparation des données
- 2.1 Sources de données : production, données payantes externes, open banking
- 2.2 Valeurs manquantes (missing conservé comme modalité à part entière)
- 2.3 Outliers : détection, traitement
- 2.4 Feature engineering au-delà du binning
- 2.5 Variables interdites, RGPD et AI Act

### 3. Analyse exploratoire : associations et redondances entre variables
- 3.0 Introduction
- 3.1 Quanti/quanti : Pearson, Spearman, Kendall
- 3.2 Quali/quali : Chi², V de Cramer
- 3.3 Quanti/quali (croisé) : ANOVA, eta², Kruskal-Wallis
- 3.4 Multicolinéarité : VIF, matrice de corrélation, ACP exploratoire
- 3.5 Détecter les relations non monotones
- 3.6 Limites pour la sélection de variables prédictives

### 4. Discrétisation et regroupement de modalités des variables explicatives
- 4.1 Objectifs de la discrétisation
- 4.2 Méthodes de découpage (manuel/expert, quantiles, arbre univarié, ChiMerge)
- 4.3 Weight of Evidence (WoE) et Information Value (IV) — seuils Siddiqi
- 4.4 Monotonicité et contraintes métier
- 4.5 Gestion des modalités rares et à forte cardinalité (traitement du missing)
- 4.6 Règles de stabilité dans le temps (écart de risque ≥30% entre classes adjacentes)
- 4.7 Processus recommandé : du découpage fin au découpage final

### 5. Modèles de scoring (risk differentiation)
- 5.1 Régression logistique et scorecards classiques (encodage dummies,
  modalité 1 = référence, tests globaux/par variable/par modalité, monotonie
  des coefficients, IC95% de l'odds ratio)
- 5.2 Arbres, forêts, gradient boosting (modèle concurrent obligatoire, jamais
  retenu comme final en scoring réglementaire)
- 5.3 Métriques d'ordonnancement d'un modèle (AUC, Gini, KS)
- 5.4 Sélection du modèle le plus parcimonieux
- 5.5 Une alternative orientée métier : le stepwise par ordre de liaison

### 6. Le modèle : ordonnancement et traduction en points (risk differentiation)
- 6.1 Sortie du modèle : score continu et capacité d'ordonnancement
- 6.2 Construire une échelle de points : la méthode du PDO (Points to Double
  the Odds) — dérivation Offset/Factor, formule additive par modalité
- 6.3 Une alternative pratique : la grille par normalisation des coefficients
- 6.4 Comparaison des deux méthodes sur un exemple commun

### 7. Mise en classe du score (risk differentiation)
- 7.1 Objectif et construction des Classes Homogènes de Risque (CHR)
- 7.2 Règles de stabilité en risque et en volume des CHR dans le temps
- 7.3 Homogénéité intra-CHR : analyse selon une variable exogène
- 7.4 Hétérogénéité inter-CHR : vérifier la différenciation des niveaux de risque
- 7.5 Construction d'une échelle maître (si plusieurs segments)
- 7.6 Validation globale : perte d'information induite par la mise en classe

### 8. Du score à la probabilité : le calibrage (risk quantification)
- 8.1 De la discrimination à la quantification du risque
- 8.2 Pourquoi une fréquence observée ne suffit pas
- 8.3 Les différentes notions de probabilité de défaut (PIT, TTC, modèle ASRF)
- 8.4 Le calibrage Through-the-Cycle : le Long Run Average (LRA)
- 8.5 Les méthodes de calibrage (fréquence observée, LRA, régression
  logistique, régression isotone, approches bayésiennes/crédibilité)
- 8.6 Vérifier le calibrage (diagramme de calibration, Brier Score et
  décomposition de Murphy)
- 8.7 Du calibrage à la PD réglementaire (Margin of Conservatism, catégories A/B/C)

### 9. Backtesting et robustesse dans le temps *(non exigé, pour culture)*
- 9.1 Backtesting du pouvoir discriminant
- 9.2 Backtesting de la calibration
- 9.3 Stabilité de la population : PSI et CSI
- 9.4 Stress testing et matrices de migration

### 10. Enjeux avancés *(non exigé, encouragé)*
- 10.1 Interprétabilité (SHAP, LIME) vs scorecards natifs
- 10.2 Biais, fairness, gouvernance
- 10.3 Limites du machine learning face à l'économétrie classique


## Livrables attendus du projet

- Segmentation de la population en amont (2 à 4 segments, justifiée
  statistiquement et métier ; discuter aussi le nombre "optimal" théorique)
- Pour chaque segment : préparation des données, analyse exploratoire, maîtrise des données
- Pour chaque segment : risk differentiation (discrétisation, regroupement de
  modalités, modèle de scoring — régression logistique obligatoire + au moins
  un modèle concurrent, grille de score). Explicabilité (ch. 10) encouragée mais pas obligatoire.
- Risk quantification : par segment ou sur échelle maître unifiée (à justifier)
- Rapport professionnel niveau "dossier de validation" : toute décision
  documentée (variable retenue/écartée, choix de seuils, points volontairement
  écartés de l'étude)

## Critères de notation
- Qualité et complétude du rapport, respect du cours
- Performance et pertinence de la segmentation / data engineering 
- Performance du modèle, principalement sur le Gini 
