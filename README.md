## Installation des dépendances

Ce projet utilise [`uv`](https://github.com/astral-sh/uv), un gestionnaire de paquets Python moderne qui remplace `pip` et `venv`. Il suffit de deux étapes pour récupérer l'environnement complet.

**Étape 1 — Installer `uv` :**

```bash
pip install uv
```

**Étape 2 — Installer toutes les dépendances du projet :**

```bash
uv sync
```

Cette commande crée automatiquement un environnement virtuel dans le dossier `.venv/` et installe toutes les bibliothèques nécessaires telles qu'elles ont été fixées par les développeurs du projet.

---

## Contexte du projet

Dans le cadre de notre enseignement en actuariat à l'ENSAE Paris, nous réalisons un projet dans le cours *"Generative AI for Insurance and Actuarial Studies"*.

## Objectif

L'objectif est d’appliquer des modèles génératifs à un indicateur physique clé du risque climatique assurantiel : le **Soil Wetness Index (SWI)**.

## Méthodologie

Pour cela, nous :
- décrivons d’abord les données mobilisées,
- explorons leurs propriétés statistiques et spatiales,
- effectuons une étape de prétraitement,
- entraînons et comparons plusieurs modèles génératifs,
- évaluons leur capacité à reproduire les caractéristiques physiques du SWI.

## Conclusion

Nous concluons par une discussion des résultats et des perspectives pour l’actuariat climatique.