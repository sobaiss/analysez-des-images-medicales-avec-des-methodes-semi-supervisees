# BrainScanAI — Détection de tumeurs cérébrales par apprentissage semi-supervisé

Projet R&D pour **CurelyticsIA** : exploration d'un pipeline de classification d'IRM cérébrales (cancer / normal) dans un contexte semi-supervisé, où la grande majorité des images n'est pas étiquetée.

## Contexte

Le dataset contient **1506 IRM** au format JPEG (512×512, RGB) :
- `data/sans_label/` — 1406 images **non étiquetées**
- `data/avec_labels/cancer/` — 50 images annotées **cancer**
- `data/avec_labels/normal/` — 50 images annotées **normal**

Seulement ~6,6 % des images disposent d'un label → justifie une approche semi-supervisée.

## Pipeline en 2 notebooks

```
Notebook 1 : Préparation & Extraction      → Exploration dataset, qualité, ResNet50 features (avgpool 2048D)
Notebook 2 : Clustering & Semi-supervision → K-Means, DBSCAN, Transfer Learning CNN, Pseudo-labellisation
```

## Résultats

| Modèle | Accuracy | F1 macro | AUC-ROC |
|--------|:--------:|:--------:|:-------:|
| Supervisé pur (80 images) | 0.950 | 0.950 | 1.000 |
| Semi-supervisé (weak → fine-tune) | 1.000 | 1.000 | 1.000 |
| **Pseudo-labellisation** (50 labels forts) | **0.880** | **0.880** | **0.872** |

Le pré-entraînement sur les 1406 weak labels améliore le classificateur final, malgré la qualité limitée des pseudo-labels (ARI K-Means = 0.11). La pseudo-labellisation atteint 0.88 d'accuracy avec seulement 50 labels forts initiaux, en exploitant 1406 images non étiquetées sur 12 rounds itératifs.

## Structure du projet

```
.
├── data/
│   ├── sans_label/            # 1406 images non étiquetées
│   ├── avec_labels/
│   │   ├── cancer/            # 50 images annotées cancer
│   │   └── normal/            # 50 images annotées normal
│   ├── features.npz           # Features ResNet50 (layer3: 1506×1024, avgpool: 1506×2048)
│   ├── features_meta.csv      # Métadonnées (path, label, split)
│   └── weak_labels.csv        # Pseudo-labels K-Means (1406 images)
├── notebooks/
│   ├── preparation-et-extraction-de-features.ipynb                      # Exploration dataset + extraction ResNet50
│   └── clustering-et-semi-supervision-et-support-de-presentation.ipynb  # Clustering + CNN + Pseudo-labels
├── specs/
│   ├── context.md
│   ├── etape-1.md
│   ├── etape-2.md
│   ├── etape-3.md
│   └── etape-4.md
├── Detection-automatique-de-tumeurs-cerebrales-par-IRM.pptx
├── main.py
├── requirements.txt
├── pyproject.toml
└── uv.lock
```

## Installation

```bash
# Installer les dépendances avec uv
uv sync

# Enregistrer le kernel Jupyter
uv run ipython kernel install --user --name=analysez-des-images-medicales-avec-des-methodes-semi-supervisees
```

## Exécution des notebooks

```bash
# Exécuter un notebook
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=analysez-des-images-medicales-avec-des-methodes-semi-supervisees \
  --inplace notebooks/preparation-et-extraction-de-features.ipynb

# Lancer Jupyter Lab pour une exploration interactive
uv run jupyter lab
```

## Outils utilisés

| Catégorie | Bibliothèques |
|-----------|--------------|
| Deep Learning | `torch`, `torchvision` (ResNet50 IMAGENET1K_V2) |
| Machine Learning | `scikit-learn` (PCA, K-Means, DBSCAN, ARI, métriques) |
| Visualisation | `matplotlib`, `seaborn` |
| Données | `numpy`, `pandas`, `Pillow`, `tqdm` |
