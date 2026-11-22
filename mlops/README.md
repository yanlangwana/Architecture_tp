# `mlops/` — artefacts d'exploitation du modèle FraudLake

| Fichier / dossier | Produit par | Contenu |
|---|---|---|
| `MODEL_CARD.md` | notebook J5 | fiche du dernier modèle évalué : usage, données, performance, portes de qualité, équité, limites |
| `runs/J5_<date>.ipynb` | `task mlops:retrain` | rapport exécuté de chaque ré-entraînement (piste d'audit) |
| `mlflow.db`, `mlartifacts/` | J5, mode de repli | suivi MLflow local, utilisé uniquement si le serveur `mlflow` (:5050) est injoignable |

Le code MLOps réutilisable est dans `fraudlake/` (features, motifs d'alerte, PSI) et `serving/` (API FastAPI) ;
les tests unitaires sont dans `tests/` et tournent en CI (`.github/workflows/ci.yml`).
