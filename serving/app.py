"""FraudLake Scoring API — sert le modèle « champion » promu par le notebook J5.

Démarrage : `uvicorn serving.app:app --host 0.0.0.0 --port 8000` (service `fraud-api` du docker-compose).
Documentation interactive : http://localhost:8010/docs
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from fraudlake import __version__
from fraudlake.config import CHAMPION_META_PATH, CHAMPION_PATH, MODEL_FEATURES
from fraudlake.explain import reasons_for
from fraudlake.features import build_model_features, priority_for
from serving.schemas import BatchIn, BatchOut, ScoreOut, TransactionIn


class SparkScorer:
    """Charge le PipelineModel Spark champion depuis HDFS et score des lots de transactions."""

    def __init__(self, model_path: str = CHAMPION_PATH, meta_path: str = CHAMPION_META_PATH):
        self.model_path, self.meta_path = model_path, meta_path
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None
        self._lock = threading.Lock()
        self._spark = None

    def _session(self):
        if self._spark is None:
            from pyspark.sql import SparkSession

            self._spark = (
                SparkSession.builder.appName("FraudLakeScoringAPI")
                .master(os.environ.get("FRAUDLAKE_SPARK_MASTER", "local[2]"))
                .config("spark.driver.memory", "1g")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "2")
                .getOrCreate()
            )
            self._spark.sparkContext.setLogLevel("ERROR")
        return self._spark

    def load(self) -> None:
        from pyspark.ml import PipelineModel

        with self._lock:
            try:
                spark = self._session()
                model = PipelineModel.load(self.model_path)
                meta = json.loads(spark.read.text(self.meta_path).first()[0])
                self.model, self.metadata, self.error = model, meta, None
            except Exception as exc:  # modèle pas encore promu : l'API reste disponible en mode dégradé
                self.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "inconnue"))

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def jvm_alive(self) -> bool:
        """Vérifie que la JVM Spark répond encore (elle peut être tuée par le système en cas de manque de mémoire)."""
        if self._spark is None:
            return self.model is None
        try:
            return not self._spark.sparkContext._jsc.sc().isStopped()
        except Exception:
            return False

    def _reset(self) -> None:
        """Oublie la session Spark morte pour en recréer une propre au prochain appel."""
        from pyspark import SparkContext
        from pyspark.sql import SparkSession

        with contextlib.suppress(Exception):
            self._spark.stop()
        self._spark, self.model = None, None
        SparkContext._active_spark_context, SparkContext._gateway, SparkContext._jvm = None, None, None
        SparkSession._instantiatedSession, SparkSession._activeSession = None, None

    def _score_once(self, feature_rows: list[dict[str, Any]]) -> list[float]:
        from pyspark.ml.functions import vector_to_array

        spark = self._session()
        df = spark.createDataFrame([[r[c] for c in MODEL_FEATURES] for r in feature_rows], schema=self._schema())
        out = self.model.transform(df).select(vector_to_array("probability")[1].alias("p")).collect()
        return [float(r["p"]) for r in out]

    def score(self, feature_rows: list[dict[str, Any]]) -> list[float]:
        try:
            return self._score_once(feature_rows)
        except Exception:
            if self.jvm_alive:
                raise
            self._reset()      # auto-réparation : nouvelle JVM, rechargement du champion, puis une seule nouvelle tentative
            self.load()
            if not self.ready:
                raise
            return self._score_once(feature_rows)

    @staticmethod
    def _schema():
        from pyspark.sql import types as T

        ints = {"hour", "is_night", "is_weekend", "is_online", "age", "txn_count_24h", "secs_since_prev_txn", "days_to_holiday"}
        fields = []
        for c in MODEL_FEATURES:
            if c in ("category", "gender"):
                t = T.StringType()
            elif c in ints:
                t = T.IntegerType()
            else:
                t = T.DoubleType()
            fields.append(T.StructField(c, t))
        return T.StructType(fields)


STATS = {"requests": 0, "transactions": 0, "alerts": 0, "latency_ms_total": 0.0, "started_at": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "scorer"):
        app.state.scorer = SparkScorer()
    if os.environ.get("FRAUDLAKE_SKIP_MODEL_LOAD") != "1":
        threading.Thread(target=app.state.scorer.load, daemon=True).start()   # démarrage non bloquant
    yield


app = FastAPI(
    title="FraudLake Scoring API",
    version=__version__,
    description="Scoring temps réel des transactions par carte bancaire — modèle champion du registre MLflow.",
    lifespan=lifespan,
)


def _scorer():
    scorer = app.state.scorer
    if not scorer.ready:
        raise HTTPException(status_code=503, detail=f"Modèle non chargé ({scorer.error or 'chargement en cours'})")
    return scorer


def _score(transactions: list[TransactionIn]) -> tuple[list[ScoreOut], float]:
    scorer = _scorer()
    t0 = time.perf_counter()
    raw = [t.model_dump() for t in transactions]
    features = [build_model_features(r) for r in raw]
    scores = scorer.score(features)
    results = []
    for tx, feat, s in zip(raw, features, scores, strict=True):
        alert = s >= scorer.threshold
        results.append(ScoreOut(
            transaction_id=tx.get("transaction_id"), fraud_score=round(s, 5),
            decision="ALERTE" if alert else "ACCEPTÉE", priority=priority_for(s, scorer.threshold),
            threshold=scorer.threshold, reasons=reasons_for({**feat, "amt": tx["amt"]}) if alert else [],
            model_version=scorer.version,
        ))
    latency = (time.perf_counter() - t0) * 1000
    STATS["requests"] += 1
    STATS["transactions"] += len(results)
    STATS["alerts"] += sum(r.decision == "ALERTE" for r in results)
    STATS["latency_ms_total"] += latency
    return results, latency


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["exploitation"])
def health():
    s = app.state.scorer
    ok = s.ready and getattr(s, "jvm_alive", True)
    detail = s.error if s.error or ok else "JVM Spark arrêtée : elle sera relancée automatiquement au prochain appel"
    return {"status": "ok" if ok else "degraded", "model_loaded": s.ready, "model_version": s.version if s.ready else None,
            "detail": detail}


@app.get("/model", tags=["exploitation"])
def model_info():
    return {**_scorer().metadata, "features": MODEL_FEATURES}


@app.get("/stats", tags=["exploitation"])
def stats():
    n = max(STATS["requests"], 1)
    return {**{k: v for k, v in STATS.items() if k != "latency_ms_total"},
            "alert_rate": STATS["alerts"] / max(STATS["transactions"], 1),
            "mean_latency_ms": round(STATS["latency_ms_total"] / n, 1),
            "uptime_s": round(time.time() - STATS["started_at"])}


@app.post("/reload", tags=["exploitation"])
def reload_model():
    """Recharge le champion après une promotion dans le registre (déploiement sans redémarrage)."""
    app.state.scorer.load()
    return health()


@app.post("/predict", response_model=ScoreOut, tags=["scoring"])
def predict(tx: TransactionIn):
    results, _ = _score([tx])
    return results[0]


@app.post("/predict/batch", response_model=BatchOut, tags=["scoring"])
def predict_batch(batch: BatchIn):
    results, latency = _score(batch.transactions)
    return BatchOut(model_version=_scorer().version, n_transactions=len(results),
                    n_alerts=sum(r.decision == "ALERTE" for r in results), latency_ms=round(latency, 1), results=results)
