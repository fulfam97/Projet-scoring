# Corrections à apporter — `statistique_Mono_sansEngagement.ipynb` (segment SA)

Relecture du 14/10/2026. Les points 1 à 3 changent directement la liste des variables transmises à la modélisation : ils sont à corriger en priorité.

## Solution recommandée

Relancer le notebook **`0b - Stats_descriptives.ipynb`** avec `SEGMENT_ETUDIE = "SA_mono_sans_eng"` (cellule 1). Il applique déjà toutes les règles ci-dessous : variables à IV > 0,5 gardées avec alerte, valeurs rares discriminantes conservées, redondances traitées, variables de sondage exclues. On obtient ainsi la même méthode et le même export (`outputs/stat_desc_SA.xlsx`, onglet `SA_Fiche_variables`) pour les trois segments, ce qui permet d'utiliser directement le notebook `1 - Differenciation.ipynb` sur SA.

Si vous préférez garder ce notebook, voici les corrections une par une.

## 1. Ne pas exclure les variables à IV > 0,5 (bloquant)

**Constat** : la cellule finale exclut les 19 variables « suspectes » (`vars_iv_suspect`). Ce sont les plus discriminantes du segment : `MINSOLDE_PAR` (0,81), `anciennete_jour_sain_M0` (0,76), `CRTAD_AG_NBJDE_BA` (0,75), `solde_DAV_moyen` (0,73), `solde_crediteur` (0,68), `Dfo_max_*` (≈ 0,5)…

**Pourquoi c'est un problème** : selon le cours (seuils de Siddiqi), une IV > 0,5 impose de **vérifier l'absence de fuite** (variable bien mesurée avant la date d'observation), pas d'exclure. Le texte de la section 3.6 du notebook le dit lui-même. En l'état, le modèle perdrait l'essentiel de son pouvoir discriminant.

**Correction** : retirer `vars_iv_suspect` de `vars_a_exclure` ; garder la liste pour documenter le contrôle de fuite.

## 2. Exclure `SamplingWeight` et `DATE_SAIN` (bloquant)

**Constat** : ces deux colonnes sont encore dans `base_filtre` (sortie de la cellule 37 : IV de 5,91 pour `SamplingWeight`, 1,06 pour `DATE_SAIN`). Elles ne figurent dans aucune liste d'exclusion de la cellule finale (`EXCLURE` n'est utilisé que pour construire `VARS`).

**Pourquoi c'est un problème** :
- `SamplingWeight` (et `SelectionProb`) encodent la cible via le plan de sondage : exclusion décidée au périmètre (notebook 0).
- `DATE_SAIN` est une date brute : son information est déjà portée par `anciennete_jour_sain_M0`.

**Correction** : partir de `VARS` (qui exclut déjà ces colonnes) plutôt que de `base.columns` pour construire `colonnes_a_garder`, ou les ajouter explicitement à `vars_a_exclure`. Retirer aussi les colonnes constantes de structure encore présentes (`code_marche_v2`, `type_PM_PP`, `has_engagement`).

## 3. Ne pas exclure automatiquement les quasi constantes discriminantes

**Constat** : les 15 quasi constantes sont toutes exclues, dont `Dfo_max_Mm1_Mm3` (défaut passé récent) et `CODITDBDF_PAR` (incident Banque de France).

**Pourquoi c'est un problème** : une modalité rare peut être très risquée (dans SB, ces variables sont parmi les plus risquées : 18 à 25 % de défaut). Le texte de la section 3.2 conclut d'ailleurs qu'elles « ne doivent pas être supprimées automatiquement ».

**Correction** : n'exclure une quasi constante que si son IV est faible (règle du 0b : quasi constante **et** IV < 0,1).

## 4. Utiliser le contrôle de redondance

**Constat** : 126 paires de variables corrélées (Spearman ≥ 0,8) sont identifiées (cellule 16, affichée deux fois : cellules 16 et 18 identiques), mais aucune variable n'est écartée.

**Correction** : pour chaque groupe de variables corrélées, garder celle de plus forte IV (fonction `exploration.preselection_variables`, utilisée dans le 0b).

## 5. Réparer la cellule VIF

**Constat** : erreur `AttributeError: module 'pandas' has no attribute 'baseFrame'` (cellule 29).

**Cause probable** : un rechercher-remplacer « data » → « base » a aussi modifié `pd.DataFrame`. Vérifier qu'aucun autre appel n'a été touché.

**Correction** : remplacer `pd.baseFrame` par `pd.DataFrame`. Calculer le VIF sur un nombre réduit de variables (après les points 1 à 4), sinon `dropna()` sur toutes les variables numériques supprime une grande partie des lignes et les variables constantes faussent le calcul.

## 6. Mettre les textes à jour

**Constat** : le markdown de la section 3.6 indique « 175 variables : 87 inutiles, 27 faibles, 26 moyennes, 16 fortes, 19 suspectes », alors que la sortie donne **161 variables : 80 inutiles, 25 faibles, 23 moyennes, 14 fortes, 19 suspectes**.

**Correction** : reprendre les chiffres après relance.

## 7. Points mineurs

- Les 64 variables strictement constantes sont **structurelles** (toutes les variables de crédit valent 0 dans un segment sans engagement) : à expliquer en une phrase dans le rapport.
- Le fichier `base_mono_sans_engagement_filtre.parquet` (cellule 38) sera à régénérer après correction, ou à abandonner si le 0b est utilisé (le notebook 1 lit les candidates dans la fiche Excel du 0b).
- La redéfinition de `TARGET` et les `import polars as pl` répétés dans plusieurs cellules peuvent être retirés.

---

# Corrections à apporter — `Analyse_Mono_Sans_Engagement_decoupe.ipynb` (discrétisation et modèle SA)

Relecture du 14/10/2026. Les bonnes pratiques sont déjà là (découpage par client, seuils appris sur le train seulement, missing en classe à part, référence = classe la moins risquée, erreurs standard par grappes, modèle concurrent). Mais le modèle actuel n'est **pas exploitable en l'état** : il est numériquement dégénéré et surapprend.

## A. Le modèle est numériquement dégénéré (bloquant)

**Constat** :
- matrice de conception de rang déficient (avertissements `SingularMatrixWarning`, nombre de conditionnement 1e16) ;
- inversion de la hessienne impossible : toutes les erreurs standard valent `NaN` ;
- le modèle par grappes et les tests par variable plantent (`LinAlgError: Singular matrix`, cellules 78 et 83) ;
- VIF jusqu'à 3·10⁸ ; intervalles de confiance des odds ratios de 0 à l'infini ; un odds ratio de 2 443 pour la classe manquante de `NBJDE_RATIO_A_vs_A1`.

**Causes probables** :
1. Les **classes manquantes (`-1`)** de plusieurs variables concernent exactement les mêmes lignes (même source de données absente) : leurs indicatrices sont identiques, d'où la colinéarité parfaite. Le contrôle des « partitions identiques » (cellule 68) compare les variables entières, pas les indicatrices une à une.
2. **Trop de paramètres pour trop peu de défauts** : 37 variables, 120 indicatrices, pour 359 défauts dans le train (environ 3 défauts par paramètre ; la règle usuelle est d'au moins 10).

**Correction** :
- réduire fortement le nombre de variables avant d'estimer (sélection pas à pas par ordre d'IV, cours 5.5, en ne gardant une variable que si elle est significative et améliore le Gini) ; viser environ 8 à 12 variables ;
- traiter la redondance entre variables découpées (V de Cramér, comme dans le notebook 1 de SB) et pas seulement les partitions identiques ;
- rattacher une classe manquante de petite taille à la classe de risque la plus proche (comme `regrouper` dans `utils/binning.py`) plutôt que de la garder systématiquement.

## B. Surapprentissage et résultats incohérents (bloquant)

**Constat** :
- Gini train 0,76 contre Gini test 0,49 (cellule 88) : écart de 27 points, signe de surapprentissage ;
- la cellule 93 donne un **autre** Gini test pour la même régression (0,59) : les cellules n'ont pas été exécutées dans l'ordre (compteurs d'exécution 163 à 170 non croissants, cellules 92-93 sans compteur), et la constante du modèle vaut −17,5 dans un affichage et −6,5 dans un autre.

**Correction** : redémarrer le noyau et tout relancer dans l'ordre (« Restart & Run All ») avant de citer un chiffre. Le surapprentissage devrait se réduire avec la correction A.

## C. Les variables à IV > 0,5 sont exclues (bloquant)

**Constat** : les cellules 56 et 62 ne gardent que les variables à IV < 0,5. Les plus discriminantes du segment sont donc écartées : `anciennete_jour_sain_M0` (0,91), `MINSOLDE_PAR` (0,87), soldes trimestriels (0,58 à 0,75), jours débiteurs (0,58 à 0,74), `solde_crediteur` (0,53).

**Correction** : comme pour le notebook de statistiques descriptives (point 1), une IV > 0,5 impose un **contrôle de fuite**, pas une exclusion. Les réintégrer : c'est probablement le plus gros gain de Gini possible.

## D. La fusion progressive n'est pas utilisée

**Constat** : `fusion_toutes_variables_rapide` produit `train_fusionne` (57 variables traitées), mais la régression est estimée sur `train_binned` (découpage brut de l'arbre, cellules 73 et 75). L'étape de fusion n'a donc aucun effet sur le modèle.

**Correction** : appliquer les classes fusionnées au train **et** au test, puis estimer le modèle dessus.

## E. Les critères de fusion sont trop faibles

**Constat** : `MIN_EFFECTIF = 3` lignes, `MIN_DEFAUTS = 3`, écart minimal de 0,1 point de taux de défaut.

**Correction** : reprendre les règles du cours (4.4 à 4.6) : effectif minimal en pourcentage de la population (5 %), au moins 20 défauts par classe (10 si le segment est petit), écart **relatif** de risque d'au moins 30 % entre classes voisines.

## F. La stabilité temporelle est mesurée par année

**Constat** : `stabilite_temporelle` regroupe par `_annee` (cellule 54). Toutes les observations sont en 2021 : il n'y a qu'une seule période, donc aucun contrôle réel. Elle n'est d'ailleurs appliquée qu'à `RATIO3_PRO`.

**Correction** : regrouper par **mois** (`datdelhis`, 4 mois) et l'appliquer à toutes les variables retenues.

## G. Harmonisation avec les autres segments

- Découpage 80/20 stratifié sur le défaut seul ; SB utilise 70/30 stratifié sur défaut × mois (`split_train_test` dans `utils/segmentation.py`). Utiliser la même fonction pour pouvoir comparer les segments.
- Le taux de défaut d'avril est presque le double des autres mois (0,82 % contre 0,44 à 0,55 %) : stratifier par mois est d'autant plus important pour ce segment.
- La CSP (`CODACVPRO_modif`) apparaît dans les variables fusionnées alors que l'introduction annonce que les qualitatives sont exclues : clarifier.

## H. Recommandation globale

Le plus rapide est de réutiliser le pipeline du notebook `1 - Differenciation.ipynb` (segment SB) en changeant `SEGMENT_ETUDIE = "SA_mono_sans_eng"`, après avoir relancé le 0b sur SA : il traite déjà les points C à G (trois méthodes de découpage, règles du cours, stabilité mensuelle, redondance). Les bonnes idées de ce notebook (erreurs standard par grappes, tests de Wald, forêt aléatoire concurrente) pourront être reprises dans le notebook de modélisation commun.
