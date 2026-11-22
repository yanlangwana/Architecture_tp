"""Motifs d'alerte lisibles par un analyste anti-fraude (*reason codes*).

Les règles sont définies une seule fois et évaluées :
- en Python pour l'API temps réel (`reasons_for`) ;
- en Spark pour le scoring batch (`spark_reason_codes`).

Seuils calibrés sur la couche Silver (lift = taux de fraude dans la population ciblée / taux de base 0,52 %) :
montant ≥ 500 $ → ×41 · cumul 24 h ≥ 1 500 $ → ×35 · ratio à l'habitude ≥ 4 → ×23 · nuit → ×3,6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fraudlake.config import HIGH_RISK_CATEGORIES


@dataclass(frozen=True)
class Rule:
    code: str
    feature: str
    op: str             # ">=", "<=", "==", "in"
    threshold: Any
    template: str       # format printf (« %.0f », « %s ») compatible Python et Spark `format_string`


RULES: list[Rule] = [
    Rule("HIGH_AMOUNT", "amt", ">=", 500.0, "Montant élevé (%.0f $)"),
    Rule("VELOCITY_24H", "amt_sum_24h", ">=", 1500.0, "Dépenses cumulées sur 24 h anormales (%.0f $)"),
    Rule("AMT_VS_HABIT", "amt_ratio_to_card_mean", ">=", 4.0, "Montant %.1f fois supérieur à l'habitude de la carte"),
    Rule("NIGHT", "is_night", "==", 1, "Transaction nocturne (22 h – 4 h)"),
    Rule("RISKY_CATEGORY", "category", "in", tuple(HIGH_RISK_CATEGORIES), "Catégorie à risque élevé (%s)"),
    Rule("FAST_SEQUENCE", "secs_since_prev_txn", "<=", 300, "Enchaînement rapide (%.0f s après la transaction précédente)"),
]


def _match(rule: Rule, value: Any) -> bool:
    if value is None:
        return False
    if rule.op == ">=":
        return value >= rule.threshold
    if rule.op == "<=":
        return value <= rule.threshold
    if rule.op == "==":
        return value == rule.threshold
    if rule.op == "in":
        return value in rule.threshold
    raise ValueError(f"Opérateur inconnu : {rule.op}")


def _render(rule: Rule, value: Any) -> str:
    return rule.template % value if "%" in rule.template else rule.template


def reasons_for(record: dict[str, Any], rules: list[Rule] = RULES) -> list[str]:
    """Liste des motifs déclenchés pour une transaction (dictionnaire de features + `amt`)."""
    return [_render(r, record.get(r.feature)) for r in rules if _match(r, record.get(r.feature))]


def spark_reason_codes(rules: list[Rule] = RULES):
    """Colonne Spark `array<string>` des motifs déclenchés (mêmes règles que `reasons_for`)."""
    from pyspark.sql import functions as F

    items = []
    for r in rules:
        col = F.col(r.feature)
        if r.op == "in":
            cond = col.isin(list(r.threshold))
        elif r.op == ">=":
            cond = col >= F.lit(r.threshold)
        elif r.op == "<=":
            cond = col <= F.lit(r.threshold)
        elif r.op == "==":
            cond = col == F.lit(r.threshold)
        else:
            raise ValueError(f"Opérateur inconnu : {r.op}")
        value = col if r.op == "in" else col.cast("double")      # le formateur Java refuse « %.0f » sur un entier
        label = F.format_string(r.template, value) if "%" in r.template else F.lit(r.template)
        items.append(F.when(cond, label))
    return F.array_compact(F.array(*items))
