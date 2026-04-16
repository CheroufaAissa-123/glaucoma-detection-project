# Détection du glaucome à partir d’images du fond d’œil

## Présentation du projet
Ce projet vise à développer un pipeline de deep learning pour la détection du glaucome à partir d’images du fond d’œil. L’objectif est de classifier les images en deux catégories : **normal** et **glaucome**, en utilisant des techniques de vision par ordinateur.

Le projet est réalisé en équipe, chaque membre étant responsable d’une partie spécifique du pipeline.

---

## Objectifs
- Construire un modèle robuste de classification d’images pour la détection du glaucome  
- Appliquer des techniques de prétraitement et d’augmentation des données  
- Gérer le déséquilibre des classes dans les données médicales  
- Évaluer les performances du modèle avec des métriques adaptées  
- Assurer une bonne capacité de généralisation du modèle  

---

## Structure du projet
Le projet suit un workflow modulaire :

- Prétraitement et augmentation des données  
- Conception de l’architecture du modèle (CNN / Transfer Learning)  
- Entraînement et optimisation  
- Évaluation et analyse des performances  

---

## Données
Le dataset est composé d’images du fond d’œil réparties en deux classes :
- Normal  
- Glaucome  

> Remarque : Le dataset n’est pas inclus dans ce dépôt en raison de sa taille.

---

## Technologies utilisées
- Python  
- PyTorch / TensorFlow  
- Albumentations (augmentation des données)  
- NumPy / Pandas  
- Matplotlib / Seaborn  

---

## Organisation du travail en équipe
Le projet est géré avec Git selon une approche collaborative :

- `main` : branche principale stable  
- `feature-*` : branches de travail individuelles pour chaque membre  

Chaque membre travaille sur une branche dédiée puis fusionne son travail dans la branche principale.

---

## Ma contribution
Je suis responsable du **prétraitement et de l’augmentation des données**, incluant :
- Normalisation des images  
- Techniques d’augmentation  
- Préparation des données d’entraînement  

---

## Améliorations futures
- Améliorer la généralisation sur des datasets externes  
- Tester des architectures avancées (EfficientNet, ConvNeXt)  
- Optimiser les hyperparamètres  
- Déployer le modèle sous forme d’application web  

---

## Licence
Projet réalisé dans un cadre académique.