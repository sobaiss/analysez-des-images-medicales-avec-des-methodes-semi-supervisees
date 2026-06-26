# Contexte général du projet BrainScanAI
Vous êtes Data Scientist junior spécialisé en Computer Vision au sein de CurelyticsIA, une startup innovante dans le domaine de la e-santé. L'entreprise développe des solutions basées sur l'intelligence artificielle pour assister les professionnels de santé dans l'analyse d'images médicales, en particulier des IRM.

Dans le cadre d'un nouveau projet R&D, CurelyticsIA souhaite explorer la possibilité d'automatiser la détection de tumeurs du cerveau. Un ensemble conséquent de radios a été collecté : la majorité de ces images ne dispose d'aucun étiquetage, tandis qu'un sous-ensemble limité a été annoté par des radiologues experts.

Vous êtes chargé de concevoir une première exploration analytique du jeu de données. Plus précisément, votre mission est de :
* Explorer les images et extraire des caractéristiques visuelles via un modèle pré-entraîné ;
* Appliquer des méthodes de clustering pour identifier des structures ou regroupements dans les données ;
* Mettre en œuvre une méthode d'apprentissage semi-supervisé à partir des quelques étiquettes disponibles ;
* Synthétiser vos résultats, formuler des recommandations, et les présenter à votre équipe projet.

 

Vous recevez un mail de votre responsable Data Science, Clara Martin, qui vous partage les détails du projet :

# Mail de Clara
Bonjour,

Comme discuté lors de notre dernière réunion, tu es assigné à la première phase du projet BrainScanAI. Tu trouveras dans le dossier `data` :

* Dans le dossier `sans_label` le jeu de données de radiographies (en format JPG + métadonnées anonymisées),
* Dans le fichier `Jeu de Données d'Images Cérébrales pour la Détection de Tumeurs.txt` Une documentation technique sur le format des images ;
* Dans le dossier `avec_labels` une liste restreinte de labels annotés par nos partenaires hospitaliers (normal/cancer). 

Tes objectifs :
* Extraire des caractéristiques visuelles pertinentes à l'aide d'un modèle pré-entraîné (type ResNet ou équivalent).
* Réaliser un clustering exploratoire pour identifier des regroupements naturels.
* Mettre en œuvre une méthode semi-supervisée en exploitant les labels partiels pour prédire les étiquettes manquantes.

Proposer des livrables au format Notebook contenant :
* l'extraction des features
* le preprocessing adapté au(x) modèle(s) utilisés
* l'analyse non supervisée (.ipynb)
* l'entraînement de modèles de clustering
* l'approche semi-supervisée (.ipynb)
