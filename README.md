# Détection du glaucome à partir d’images du fond d’œil

## Présentation du projet
Ce projet vise à développer un pipeline de deep learning pour la détection du glaucome à partir d’images du fond d’œil.  
L’objectif est de classifier les images en deux catégories : normal et glaucome, en utilisant des techniques de vision par ordinateur.

Le projet est réalisé en équipe, chaque membre étant responsable d’une partie spécifique du pipeline.

---

## Objectifs
- Construire un modèle robuste de classification d’images pour la détection du glaucome  
- Appliquer des techniques de prétraitement et d’augmentation des données  
- Gérer le déséquilibre des classes dans les données médicales  
- Évaluer les performances du modèle avec des métriques adaptées  
- Assurer une bonne capacité de généralisation du modèle  

---

## Pipeline du projet
Le projet suit un workflow modulaire :

1. Chargement des données  
2. Prétraitement et augmentation des données  
3. Conception de l’architecture du modèle (CNN / Transfer Learning)  
4. Entraînement et optimisation  
5. Évaluation et analyse des performances  

---

## Données
Le dataset est composé d’images du fond d’œil réparties en deux classes :

- Normal  
- Glaucome  

Remarque : Le dataset n’est pas inclus dans ce dépôt en raison de sa taille.

---

## Technologies utilisées
- Python  
- PyTorch / TensorFlow  
- Albumentations  
- NumPy / Pandas  
- Matplotlib / Seaborn  

---

## Organisation du travail en équipe
Le projet est géré avec Git selon une approche collaborative :

- main : branche principale stable  
- feature-* : branches de travail individuelles pour chaque membre  

Chaque membre travaille sur une branche dédiée, effectue ses commits puis propose une fusion dans la branche principale.

---

## Contribution
Dans ce projet, la partie prétraitement des données inclut :

- Normalisation des images  
- Augmentation des données  
- Amélioration du contraste des images  
- Préparation des données pour l’entraînement  

---

## Limites du projet
- Dépendance à la qualité du dataset  
- Déséquilibre des classes  
- Difficulté de généralisation sur d’autres datasets  

---

## Améliorations futures
- Améliorer la généralisation sur des datasets externes  
- Tester des architectures avancées (EfficientNet, ConvNeXt)  
- Optimiser les hyperparamètres  
- Déployer le modèle sous forme d’application web  

---

## Licence
Projet réalisé dans un cadre académique par :

Cheroufa Aissa  
Hanafi Ahcene  
Saada Lounes  
Hanouti Yanis  
Kemiche Nassim  
Sirem Kaci