# Ahcene - Preparation Des Donnees Et Evaluation Du Modele

## Perimetre

Cette zone de contribution couvre les parties du projet liees a la preparation du pipeline de donnees pour l'entrainement et a l'evaluation des modeles de classification du glaucome.

## Responsabilites Principales

- Definir et maintenir les choix de pretraitement des images avant l'entrainement
- Configurer les pipelines d'augmentation adaptes aux images du fond d'oeil
- Preparer de maniere coherente les jeux train, validation et test
- Gerer le desequilibre des classes pendant le chargement des donnees d'entrainement
- Suivre et interpreter les metriques d'evaluation du modele entraine

## Correspondance Avec Le Projet Actuel

L'implementation actuelle de cette partie se trouve principalement dans `notebooks/code_principale.ipynb`.

Sections deja presentes dans le notebook :

- Transformations et pretraitement avec Albumentations
- Amelioration du contraste avec CLAHE
- Chargement du dataset avec `ImageFolder`
- Preparation des jeux train, validation et test
- Echantillonnage pondere pour gerer le desequilibre
- Evaluation avec accuracy, precision, recall, F1-score, AUC et matrice de confusion

## Livrables Attendus Pour Cette Branche

- Une configuration claire du pretraitement et de l'augmentation
- Des etapes reproductibles de preparation des donnees
- Une procedure d'evaluation documentee
- Un reporting propre des principales metriques de validation et de test

## Notes

Cette branche est reservee au travail lie a la preparation des donnees et a l'evaluation.
Le code d'inference de l'API et la demonstration frontend restent hors du perimetre principal de cette contribution.
