"""Construction des features modèle à partir d'une transaction « métier » (API temps réel).

Les formules reproduisent exactement celles de la couche Silver (notebook J2) :
log1p des montants / populations, nuit = 22 h → 4 h, week-end = samedi / dimanche (ISO 6-7),
e-commerce = catégorie `*_net`.
"""

from __future__ import annotations

import math
from typing import Any


def is_night(hour: int) -> int:
    return int(hour >= 22 or hour < 4)


def build_model_features(tx: dict[str, Any]) -> dict[str, Any]:
    """Transforme une transaction brute en vecteur de features attendu par le pipeline Spark."""
    category = str(tx["category"]).strip().lower()
    return {
        "amt_log": round(math.log1p(float(tx["amt"])), 5),
        "hour": int(tx["hour"]),
        "is_night": is_night(int(tx["hour"])),
        "is_weekend": int(int(tx["weekday_num"]) >= 6),
        "is_online": int(category.endswith("_net")),
        "age": int(tx["age"]),
        "distance_km": float(tx["distance_km"]),
        "city_pop_log": round(math.log1p(float(tx["city_pop"])), 5),
        "txn_count_24h": int(tx["txn_count_24h"]),
        "amt_sum_24h": float(tx["amt_sum_24h"]),
        "secs_since_prev_txn": int(tx["secs_since_prev_txn"]),
        "amt_ratio_to_card_mean": float(tx["amt_ratio_to_card_mean"]),
        "days_to_holiday": int(tx["days_to_holiday"]),
        "category": category,
        "gender": str(tx["gender"]).strip().upper(),
    }


def priority_for(score: float, threshold: float) -> str:
    """Niveau de priorité d'investigation d'une transaction scorée."""
    if score >= max(threshold, 0.97):
        return "critique"
    if score >= threshold:
        return "élevée"
    if score >= 0.5 * threshold:
        return "à surveiller"
    return "faible"
