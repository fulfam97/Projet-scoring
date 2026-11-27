# Travail d'Anis — segment comptes joints

Dossier autonome : les notebooks importent `Anis/utils/` et écrivent dans `Anis/outputs/`.
Les données restent à la racine du dépôt (`data/`, non versionné).

| Notebook | Contenu |
|---|---|
| `0b - Stats_descriptives.ipynb` | statistiques descriptives (lancer depuis `Anis/`) |
| `Segmentation_joint/0c` → `0d` | préparation des données (ch. 2), analyse exploratoire (ch. 3) |
| `Segmentation_joint/1a` | discrétisation, WOE / IV (ch. 4) |
| `Segmentation_joint/1b` → `1c` | régression logistique et concurrent (ch. 5), grille de score PDO (ch. 6) |
| `Segmentation_joint/1d` | CHR (ch. 7) |
| `Segmentation_joint/1e` | calibrage : LRA, MoC, échelle maître (ch. 8) |
| `Segmentation_joint/1f` | résultats sur l'échantillon Out-of-Time (ch. 9) |

Les suffixes `_v2` désignent la variante retenue.
