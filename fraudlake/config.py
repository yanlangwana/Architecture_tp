"""Configuration centrale du projet FraudLake (chemins, features, seuils MLOps)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLOPS_DIR = PROJECT_ROOT / "mlops"

# ── Chemins Lakehouse (HDFS dans le cluster, surchargeables pour les tests locaux) ──────────────────
LAKEHOUSE = os.environ.get("LAKEHOUSE_ROOT", "/lakehouse").rstrip("/")
SILVER_PATH = f"{LAKEHOUSE}/silver/transactions_enriched"
GOLD_ROOT = f"{LAKEHOUSE}/gold"
MODELS_ROOT = f"{LAKEHOUSE}/models"
CHAMPION_PATH = f"{MODELS_ROOT}/fraudlake_scoring/champion"          # artefact de déploiement (PipelineModel Spark)
CHAMPION_META_PATH = f"{MODELS_ROOT}/fraudlake_scoring/champion_metadata"
MLFLOW_DFS_TMP = f"{LAKEHOUSE}/tmp/mlflow"

# ── MLflow ──────────────────────────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_FALLBACK_URI = f"sqlite:///{MLOPS_DIR}/mlflow.db"            # repli si le serveur MLflow est injoignable
MLFLOW_FALLBACK_ARTIFACTS = f"file://{MLOPS_DIR}/mlartifacts"
EXPERIMENT_NAME = "fraudlake-scoring"
REGISTERED_MODEL = "fraudlake-scoring"
API_URL = os.environ.get("FRAUDLAKE_API_URL", "http://fraud-api:8000")

# ── Features du modèle (identiques en entraînement, batch et API) ───────────────────────────────────
NUM_FEATURES = [
    "amt_log", "hour", "is_night", "is_weekend", "is_online", "age", "distance_km", "city_pop_log",
    "txn_count_24h", "amt_sum_24h", "secs_since_prev_txn", "amt_ratio_to_card_mean", "days_to_holiday",
]
CAT_FEATURES = ["category", "gender"]
MODEL_FEATURES = NUM_FEATURES + CAT_FEATURES

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness", "home",
    "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]
HIGH_RISK_CATEGORIES = ["shopping_net", "misc_net", "shopping_pos", "grocery_pos"]   # classe A du Pareto (J3)

# ── Découpage temporel (aucune transaction future ne sert à l'apprentissage) ───────────────────────
VALIDATION_START = "2020-05-01"     # train : 2019-01 → 2020-04 | validation : 2020-05 → 2020-06-21 | test : fraud_test.csv
LEGIT_SAMPLING_RATE = 0.10          # sous-échantillonnage des légitimes dans le train uniquement

# ── Portes de qualité avant promotion en production ─────────────────────────────────────────────────
QUALITY_GATES = {
    "test_pr_auc": 0.75,            # aire sous la courbe précision-rappel sur la période de test
    "test_recall": 0.70,            # part des fraudes détectées au seuil retenu
    "test_precision": 0.50,         # part d'alertes justifiées (charge de l'équipe d'investigation)
    "fairness_recall_ratio": 0.80,  # rappel min / rappel max entre groupes ≥ 4/5 (règle des quatre cinquièmes)
}

# ── Surveillance de la dérive (Population Stability Index) ──────────────────────────────────────────
PSI_STABLE = 0.10                   # < 0,10 : population stable
PSI_ALERT = 0.25                    # ≥ 0,25 : dérive significative → ré-entraînement
MONITORED_FEATURES = NUM_FEATURES + ["category"]
# Variables calendaires déterministes : leur distribution varie mécaniquement d'un mois à l'autre (août sans jour férié…).
# Elles restent affichées mais ne déclenchent pas de ré-entraînement (sinon fausses alertes systématiques).
DRIFT_EXCLUDED_FROM_TRIGGER = ["days_to_holiday", "is_weekend"]
