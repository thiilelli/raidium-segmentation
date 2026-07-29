## Segmentation de coupes CT avec labels incomplets

Score leaderboard: 0.6089 (vs baseline nnU-Net 0.4399, +38%)

Le problème

54 structures anatomiques à segmenter sur 256×256 pixels. Seulement ~27 classes annotées par image, les 27 autres sont inconnues (pas du fond).

La solution

Modèle: U-Net, encodeur ResNet34 pré-entraîné ImageNet

Loss masquée: Gradient seulement sur les couples (image, classe) certains. Ignore les statuts inconnus. Exploite les 15% de supervision négative.

Sorties: 54 sigmoïdes indépendantes + tête auxiliaire qui gate les prédictions.

Résultats
Métrique	Valeur
Leaderboard	0.6089
Validation	0.6290
Training	250 epochs, T4 GPU
