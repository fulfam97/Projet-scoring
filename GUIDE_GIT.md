# Guide de collaboration Git — Projet Scoring

Ce guide résume les bonnes pratiques et les commandes adaptées à notre organisation :
- **`main`** est la branche de référence : elle ne contient que du code testé et validé par l'équipe.
- **Chaque collaborateur** travaille sur sa propre branche.
- Le transfert vers `main` se fait au niveau des **fichiers validés** pour éviter d'écraser les travaux en cours des autres.

---

## 1. Routine quotidienne : Travailler sur sa branche

Tu travailles toujours sur ta branche personnelle pour faire des tests sans risque :

```bash
# 1. Aller sur sa branche
git switch <branche>

# 2. Sauvegarder régulièrement son travail sur sa branche
git add .
git commit -m "Description claire de ce qui a été fait"

# 3. (Optionnel) Sauvegarder sa branche sur GitHub pour ne rien perdre
git push origin <branche>
```

---

## 2. Pousser un ou plusieurs fichiers validés vers `main`

C'est la procédure à suivre dès qu'un fichier (`utils/segmentation.py`, un notebook nettoyé, etc.) est prêt à être partagé avec tout le monde.

```bash
# Étape 1 : S'assurer que le travail est commité sur sa branche
git add .
git commit -m "Mon travail prêt pour validation"

# Étape 2 : Basculer sur main et la mettre à jour depuis GitHub
git checkout main
git pull origin main

# Étape 3 : Importer UNIQUEMENT le(s) fichier(s) validé(s) depuis sa branche
# (séparer par des espaces, mettre des guillemets si le nom contient des espaces)
git checkout <branche> -- utils/segmentation.py "0 - Segmentation_exploration.ipynb"

# Étape 4 : Valider et pousser sur main
git commit -m "Mise à jour utils/segmentation.py par ..."
git push origin main

# Étape 5 : Revenir sur sa branche de travail
git checkout <branche>
```

---

## 3. Récupérer les nouveautés de `main` sur sa branche

Quand un collègue a poussé des fonctions validées sur `main` et que tu veux en profiter sur ta branche :

```bash
# 1. Aller sur sa branche
git checkout <branche>

# 2. Fusionner main dans sa branche
git merge main
```
*Si un conflit apparaît, VS Code l'affiche en surbrillance. Choisis ce que tu souhaites garder ("Accept Incoming", "Accept Current" ou "Accept Both"), sauvegarde (`Cmd + S`), puis termine avec `git commit`.*

---

## 4. Résolution des pièges fréquents

### A. "Your local changes to ... would be overwritten by merge"
> **Cause** : Un fichier (très souvent un notebook `.ipynb` ou un export) a été modifié ou auto-enregistré en arrière-plan par VS Code.

* **Si ce sont des modifs temporaires / inutiles** :
  ```bash
  git restore "chemin/du/fichier.ipynb"
  git pull origin main
  ```
* **Si tu veux tout mettre de côté sans rien supprimer** :
  ```bash
  git stash -u
  git pull origin main
  # (pour récupérer tes brouillons plus tard : git stash pop)
  ```

### B. "non-fast-forward / [rejected]" lors du `git push`
> **Cause** : GitHub a des commits d'avance que tu n'as pas encore en local.

* **Solution** :
  ```bash
  git pull origin main --no-rebase
  # ou git pull --rebase origin main
  git push origin main
  ```

### C. Faux conflit lors d'un merge
> **Exemple** : Git affiche `<<<<<<< HEAD` avec ton code et une section distante vide sous `=======`.

1. Ouvre le fichier dans VS Code.
2. Clique sur **"Accept Current Change"** (pour garder ton code).
3. Sauvegarde (`Cmd + S`).
4. Dans le terminal :
   ```bash
   git add <fichier>
   git commit -m "Résolution du conflit"
   git push origin main
   ```

---

## 5. Règle absolue de sécurité des données

- **Ne jamais commiter de données réelles** (`data/raw/`, `data/processed/`, fichiers `.csv`, `.parquet`, etc.).
- Avant chaque `git add .`, fais un rapide `git status` pour vérifier qu'aucun fichier de données n'est sur le point d'être indexé.
