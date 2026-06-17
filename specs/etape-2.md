# Étape 2 – Prétraitez et extrayez les features

Préparez les images (redimensionnement, normalisation) et utilisez un modèle pré-entraîné (ex : ResNet) pour extraire des embeddings visuels.


## Prérequis

* Avoir nettoyé et formaté les données image.
* Avoir compris le fonctionnement des CNNs.

## Résultat attendu 

Vecteurs de features pour chaque image, sauvegardés dans un tableau exploitable.

## Recommandations

* Geler les couches convolutionnelles.
* Évaluer plusieurs couches d’extraction si besoin.

## Outils 
torchvision / tensorflow / transformers
numpy, 
pandas, 
matplotlib, 
opencv-python
