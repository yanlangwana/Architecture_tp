"""Surveillance du modèle en production : dérive des données (PSI) et règles de ré-entraînement."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from fraudlake.config import PSI_ALERT, PSI_STABLE


def psi(reference: Mapping, current: Mapping, eps: float = 1e-4) -> float:
    """Population Stability Index entre deux distributions exprimées en effectifs par classe.

    PSI = Σ (p_cur − p_ref) · ln(p_cur / p_ref). Les classes absentes d'un côté sont lissées par `eps`.
    """
    keys = set(reference) | set(current)
    n_ref = sum(reference.values()) or 1
    n_cur = sum(current.values()) or 1
    total = 0.0
    for k in keys:
        p_ref = max(reference.get(k, 0) / n_ref, eps)
        p_cur = max(current.get(k, 0) / n_cur, eps)
        total += (p_cur - p_ref) * math.log(p_cur / p_ref)
    return float(total)


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_ALERT:
        return "à surveiller"
    return "dérive"


def retraining_decision(max_psi: float, recall: float | None, recall_gate: float,
                        pr_auc: float | None = None, pr_auc_ref: float | None = None) -> tuple[bool, str]:
    """Règle de déclenchement du ré-entraînement (dérive de données OU dégradation de performance)."""
    reasons = []
    if max_psi >= PSI_ALERT:
        reasons.append(f"dérive de données (PSI max = {max_psi:.2f} ≥ {PSI_ALERT})")
    if recall is not None and recall < recall_gate:
        reasons.append(f"rappel {recall:.0%} < porte {recall_gate:.0%}")
    if pr_auc is not None and pr_auc_ref and pr_auc < 0.9 * pr_auc_ref:
        reasons.append(f"PR-AUC en baisse de plus de 10 % ({pr_auc:.3f} vs {pr_auc_ref:.3f})")
    return (bool(reasons), " ; ".join(reasons) if reasons else "aucun déclencheur")


def psi_by_period(spark_ref, spark_cur, features: Iterable[str], period_col: str,
                  categorical: Iterable[str] = ("category",), n_bins: int = 10):
    """PSI de chaque feature pour chaque période de `spark_cur` vs la référence `spark_ref` (Spark → pandas).

    Les bornes des classes sont les déciles de la référence (calculés une fois), appliqués à toutes les périodes.
    """
    import pandas as pd
    from pyspark.sql import functions as F

    categorical = set(categorical)
    rows = []
    for feat in features:
        if feat in categorical:
            bucket = F.col(feat).cast("string")
        else:
            qs = spark_ref.approxQuantile(feat, [i / n_bins for i in range(1, n_bins)], 0.001)
            edges = sorted(set(q for q in qs if q is not None))
            bucket = F.lit(0)
            for i, e in enumerate(edges, start=1):
                bucket = F.when(F.col(feat) > F.lit(e), F.lit(i)).otherwise(bucket)
        ref_counts = {r["b"]: r["n"] for r in spark_ref.groupBy(bucket.alias("b")).agg(F.count("*").alias("n")).collect()}
        cur = spark_cur.groupBy(F.col(period_col).alias("p"), bucket.alias("b")).agg(F.count("*").alias("n")).collect()
        by_period: dict = {}
        for r in cur:
            by_period.setdefault(r["p"], {})[r["b"]] = r["n"]
        for period, counts in sorted(by_period.items()):
            value = psi(ref_counts, counts)
            rows.append({"feature": feat, "period": str(period), "psi": round(value, 4), "status": psi_status(value),
                         "n_current": int(sum(counts.values()))})
    return pd.DataFrame(rows)
