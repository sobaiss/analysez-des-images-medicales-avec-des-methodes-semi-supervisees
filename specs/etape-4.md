# Étape 4 – Appliquez une méthode semi supervisée

Entraîner un modèle de type CNN sur votre jeu de données “faiblement” labellisé dans un premier temps puis et évaluer ses performances. Poursuivez ensuite l’entraînement de ce même modèle sur le jeu de données “fortement” labellisé. Comparer ensuite la différence de performance entre entraînement supervisé (modèle entraîné sur le jeu de données “fortement” labellisé uniquement) et semi-supervisé (entraîné sur les 2 jeux de données).

## Prérequis 

Avoir préparé un ensemble labellisé et un ensemble non labellisé

## Résultat attendu 

Modèle entraîné et validé avec les métriques choisies (accuracy, F1-score ou autre ?)

## Recommandations 

* Réaliser un split train/test équilibré.
* Pour chaque évaluation des performances: attention à ce que le jeu de test soir bien des données jamais vues par le modèle évalué lors de son entraînement.
* Utiliser des visualisations bien lisibles pour analyser les performances.

## Outils 
torchvision / tensorflow / transformers
numpy,
pandas.