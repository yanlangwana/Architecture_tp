"""Contrats d'interface de l'API de scoring (validation stricte des entrées)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudlake.config import CATEGORIES

EXAMPLE = {
    "transaction_id": "demo-0001",
    "category": "shopping_net",
    "amt": 962.4,
    "gender": "F",
    "hour": 23,
    "weekday_num": 6,
    "age": 58,
    "city_pop": 2380,
    "distance_km": 74.2,
    "txn_count_24h": 4,
    "amt_sum_24h": 2210.0,
    "secs_since_prev_txn": 1260,
    "amt_ratio_to_card_mean": 11.8,
    "days_to_holiday": 4,
}


class TransactionIn(BaseModel):
    """Transaction à scorer : attributs métier + features de vélocité calculées par le système de paiement."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE]})

    transaction_id: str | None = Field(default=None, max_length=64)
    category: str = Field(description="Catégorie du commerçant (14 valeurs)")
    amt: float = Field(gt=0, le=100_000, description="Montant en dollars")
    gender: Literal["F", "M"]
    hour: int = Field(ge=0, le=23)
    weekday_num: int = Field(ge=1, le=7, description="ISO : lundi = 1 … dimanche = 7")
    age: int = Field(ge=13, le=110)
    city_pop: int = Field(ge=1)
    distance_km: float = Field(ge=0, le=20_000)
    txn_count_24h: int = Field(ge=1, description="Transactions de la carte sur 24 h, celle-ci incluse")
    amt_sum_24h: float = Field(ge=0)
    secs_since_prev_txn: int = Field(ge=0)
    amt_ratio_to_card_mean: float = Field(ge=0)
    days_to_holiday: int = Field(ge=0, le=4, description="Distance au jour férié le plus proche (4 = plus de 3 jours)")

    @field_validator("category")
    @classmethod
    def known_category(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in CATEGORIES:
            raise ValueError(f"catégorie inconnue : {v}")
        return v


class BatchIn(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=1000)


class ScoreOut(BaseModel):
    transaction_id: str | None
    fraud_score: float = Field(description="Score de risque entre 0 et 1 (rang de risque, non calibré)")
    decision: Literal["ALERTE", "ACCEPTÉE"]
    priority: str
    threshold: float
    reasons: list[str]
    model_version: str


class BatchOut(BaseModel):
    model_version: str
    n_transactions: int
    n_alerts: int
    latency_ms: float
    results: list[ScoreOut]
