 

Développement d'une application web intelligente pour la prédiction du délai de séjour des conteneurs au port à l'aide du Machine Learning

| Version | Description | MAE test | R² test |
|---|---|---|---|
| v1 | Encodage simple (target encoding moyenne) | 4.65 | 0.329 |
| v2 | Historique riche, seuil=5, sans régularisation | 4.71 | 0.229 |
| v3 | Historique riche, seuil=30, régularisé | 4.87 | 0.267 (RF) / 0.295 (GB) |
