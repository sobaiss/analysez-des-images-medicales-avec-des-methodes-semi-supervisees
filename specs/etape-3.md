# Étape 3 – Réalisez une analyse non supervisée

Réduisez la dimension des features (PCA, t-SNE) et appliquez un clustering (K-Means, DBSCAN) pour identifier des regroupements d'images similaires (avec le même nombre de groupes que votre dataset labellisé ici 2). Le but est de labelliser de façon non-supervisée les images pour lesquelles vous n'avez pas de label. On parle alors de labellisation “faible”.

Attention ! Gardez bien ce jeu de données “faiblement” labélisé séparé du jeu de données “fortement” labellisé (pour lequel vous avez les labels de base). Il ne faut jamais les mélanger !

## Prérequis 

Avoir un tableau de features standardisé

## Instructions 

* Réduire la dimensionnalité avec PCA ou t-SNE pour visualisation.
* Appliquer un clustering (ex : K-Means, DBSCAN).
* Visualiser les résultats et interpréter les regroupements.
* Déterminer via le score ARI (avec les données “fortement” labellisées) si vous êtes sur le bon chemin ou non.

## Recommandations :

* Standardiser les features.
* Tester plusieurs méthodes de clustering.
* Ne pas s'attendre à un résultat parfait via le score ARI.

## Outils
scikit-learn, seaborn, matplotlib, plotly

## Résultat attendu 
Des graphes 2D illustrant les clusters, accompagnés de commentaires.
