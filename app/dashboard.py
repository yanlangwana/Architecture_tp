"""FraudLake — Centre de pilotage de la détection de fraude bancaire.

Lit exclusivement la couche Gold (/lakehouse/gold/*) produite par les notebooks J3, J4 et J5.
La requête SQL et la justification de chaque visualisation proviennent de la table `data_catalog`.
"""

from __future__ import annotations

import decimal
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=FutureWarning)   # pandas 3 / PySpark : sans impact

LAKEHOUSE = os.environ.get("LAKEHOUSE_ROOT", "/lakehouse").rstrip("/")
GOLD = f"{LAKEHOUSE}/gold"
API_URL = os.environ.get("FRAUDLAKE_API_URL", "http://fraud-api:8000").rstrip("/")

# ── Palette (validée daltonisme, cf. rapport) ───────────────────────────────────
NAVY, INK, INK_2, MUTED, GRID = "#0f1b2d", "#1e293b", "#475569", "#94a3b8", "#edf1f6"
LEGIT, FRAUD, AMBER, GREEN, NEUTRAL = "#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#a3acb9"
SEGMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORANGES = [[0, "#fff5ee"], [0.25, "#fcd3b6"], [0.5, "#f5a272"], [0.75, "#eb6834"], [1, "#9a3412"]]
BLUES = [[0, "#eef4fc"], [0.5, "#8db6ea"], [1, "#1d5fae"]]
WEEKDAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

st.set_page_config(page_title="FraudLake · Centre de pilotage", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ── Thème Plotly commun à tous les graphiques ───────────────────────────────────
pio.templates["fraudlake"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color=INK_2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=SEGMENT_COLORS, barcornerradius=4, separators=", ",
    xaxis=dict(showgrid=False, linecolor="#cfd8e3", ticks="outside", tickcolor="#cfd8e3", ticklen=4,
               title=dict(font=dict(size=12, color=MUTED)), zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor="rgba(0,0,0,0)", zeroline=False,
               title=dict(font=dict(size=12, color=MUTED))),
    hoverlabel=dict(bgcolor="white", bordercolor="#dfe5ee", font=dict(color=NAVY, size=12, family="Inter, sans-serif")),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None, font=dict(size=12, color=INK_2)),
    margin=dict(l=8, r=8, t=30, b=8),
    coloraxis=dict(colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=11, color=MUTED))),
))
pio.templates.default = "plotly_white+fraudlake"

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric, button, input, textarea { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1440px; }
header[data-testid="stHeader"] { background: transparent; }
.msr { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
       letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr;
       -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }

/* ---------- Barre latérale ---------- */
section[data-testid="stSidebar"] { border-right: 1px solid #1f3152; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.brand { display: flex; align-items: center; gap: 11px; padding: 4px 4px 16px 4px; margin-bottom: 8px;
         border-bottom: 1px solid #22385a; }
.brand .logo { width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
               background: linear-gradient(135deg, #eb6834, #f29a57); color: white; font-size: 22px;
               box-shadow: 0 6px 18px rgba(235,104,52,.35); }
.brand .name { font-weight: 800; font-size: 1.12rem; color: #fff; letter-spacing: -.01em; line-height: 1.1; }
.brand .tag { font-size: .72rem; color: #8ea3bf; letter-spacing: .06em; text-transform: uppercase; }
.side-label { font-size: .68rem; font-weight: 700; color: #6f86a6; letter-spacing: .1em; text-transform: uppercase;
              margin: 14px 0 6px 6px; }
.st-key-page, .st-key-page > div { width: 100% !important; }
.st-key-page [role="radiogroup"] { gap: 2px; width: 100%; }
.st-key-page [role="radiogroup"] > div { width: 100%; }
.st-key-page label[data-testid="stRadioOption"] { width: 100%; padding: 8px 12px; border-radius: 9px; margin: 0;
  transition: background .15s; cursor: pointer; }
.st-key-page label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-page label[data-testid="stRadioOption"]:hover { background: rgba(255,255,255,.06); }
.st-key-page label[data-testid="stRadioOption"][data-selected="true"] {
  background: linear-gradient(90deg, rgba(42,120,214,.32), rgba(42,120,214,.08)); box-shadow: inset 3px 0 0 #5aa0f0; }
.st-key-page label[data-testid="stRadioOption"] p { font-size: .9rem; font-weight: 500; color: #c3cfde; }
.st-key-page label[data-testid="stRadioOption"] p span[role="img"] { margin-right: 6px; font-size: 1.15rem; color: #7f96b6; }
.st-key-page label[data-testid="stRadioOption"][data-selected="true"] p { color: #fff; font-weight: 600; }
.st-key-page label[data-testid="stRadioOption"][data-selected="true"] p span[role="img"] { color: #8fc0ff; }
.side-foot { font-size: .72rem; color: #7b90ad; line-height: 1.5; margin-top: 18px; padding-top: 12px; border-top: 1px solid #22385a; }
.side-foot b { color: #b8c6d9; font-weight: 600; }

/* ---------- En-tête de page ---------- */
.page-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; flex-wrap: wrap;
             margin: 0 0 18px 0; padding-bottom: 16px; border-bottom: 1px solid #e2e8f0; }
.page-head .crumb { font-size: .74rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #2a78d6; }
.page-head h1 { font-size: 1.72rem; font-weight: 800; color: #0f1b2d; margin: 2px 0 2px 0; letter-spacing: -.02em; padding: 0; }
.page-head p { margin: 0; color: #64748b; font-size: .93rem; max-width: 760px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: center; gap: 6px; background: #fff; border: 1px solid #e2e8f0; border-radius: 999px;
        padding: 5px 11px; font-size: .78rem; color: #334155; font-weight: 500; box-shadow: 0 1px 2px rgba(15,27,45,.04); }
.chip .dot { width: 7px; height: 7px; border-radius: 50%; background: #1baf7a; box-shadow: 0 0 0 3px rgba(27,175,122,.18); }
.chip .msr { font-size: 15px; color: #64748b; }

/* ---------- Cartes KPI ---------- */
.kpi { position: relative; background: #fff; border: 1px solid #e3e8ef; border-radius: 14px; padding: 15px 16px 13px 16px;
       height: 100%; box-shadow: 0 1px 2px rgba(15,27,45,.04), 0 6px 20px -12px rgba(15,27,45,.12); overflow: hidden; }
.kpi .top { display: flex; align-items: center; gap: 9px; }
.kpi .ico { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; font-size: 18px; flex-shrink: 0; }
.kpi .label { font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; color: #64748b; font-weight: 600; }
.kpi .value { white-space: nowrap; font-size: 1.55rem; font-weight: 800; margin-top: 9px; line-height: 1.15; color: #0f1b2d; letter-spacing: -.02em; }
.kpi .sub { font-size: .8rem; color: #64748b; margin-top: 3px; }
.kpi .vrow { display: flex; align-items: flex-end; justify-content: space-between; gap: 8px; }
.kpi .spark { flex-shrink: 0; margin-bottom: 4px; }
.kpi.blue .ico { background: rgba(42,120,214,.10); color: #2a78d6; }
.kpi.orange .ico { background: rgba(235,104,52,.11); color: #d4561f; }
.kpi.green .ico { background: rgba(27,175,122,.12); color: #12865d; }
.kpi.amber .ico { background: rgba(237,161,0,.13); color: #b07800; }
.kpi.slate .ico { background: rgba(100,116,139,.12); color: #475569; }

/* ---------- Cartes graphiques ---------- */
.card-head { margin: 2px 0 2px 0; }
.card-head .t { font-size: 1rem; font-weight: 700; color: #0f1b2d; letter-spacing: -.01em; }
.card-head .s { font-size: .82rem; color: #64748b; margin-top: 1px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #e3e8ef !important; border-radius: 14px !important;
       box-shadow: 0 1px 2px rgba(15,27,45,.04), 0 6px 20px -14px rgba(15,27,45,.14); }

/* ---------- Encadrés « points clés » ---------- */
.insights { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin: 4px 0 16px 0; }
.insight { background: linear-gradient(180deg, #ffffff, #fbfcfe); border: 1px solid #e3e8ef; border-radius: 12px; padding: 12px 14px;
           display: flex; gap: 11px; align-items: flex-start; }
.insight .msr { font-size: 20px; color: #eb6834; margin-top: 1px; }
.insight.blue .msr { color: #2a78d6; }
.insight.green .msr { color: #1baf7a; }
.insight b { color: #0f1b2d; }
.insight div { font-size: .85rem; color: #475569; line-height: 1.45; }

/* ---------- Divers ---------- */
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700; letter-spacing: .01em; }
.badge.reject { background: rgba(235,104,52,.13); color: #c2410c; }
.badge.keep { background: #eef2f7; color: #475569; }
.badge.ok { background: rgba(27,175,122,.14); color: #0f7a53; }
.lifecycle { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 0 0 16px 0; }
.lifecycle .step { position: relative; background: #fff; border: 1px solid #e3e8ef; border-radius: 12px; padding: 12px 13px; }
.lifecycle .step .n { width: 24px; height: 24px; border-radius: 7px; background: #0f1b2d; color: #fff; font-size: .75rem;
                      font-weight: 700; display: grid; place-items: center; margin-bottom: 8px; }
.lifecycle .step b { display: block; font-size: .88rem; color: #0f1b2d; }
.lifecycle .step span { font-size: .78rem; color: #64748b; }
.layer { border-radius: 12px; padding: 14px 16px; border: 1px solid #e3e8ef; background: #fff; height: 100%; border-top: 4px solid; }
.layer h4 { margin: 0 0 4px 0; font-size: 1rem; font-weight: 800; color: #0f1b2d; padding: 0; }
.layer code { font-size: .74rem; }
.layer p { font-size: .84rem; color: #475569; margin: 6px 0 0 0; line-height: 1.45; }
.test-q { font-weight: 700; color: #0f1b2d; font-size: .95rem; }
.test-d { color: #64748b; font-size: .82rem; }
div[data-testid="stExpander"] details { border-radius: 10px; border-color: #e3e8ef; background: #fafbfd; }
div[data-testid="stExpander"] summary p { font-size: .84rem; color: #475569; font-weight: 500; }
div[data-testid="stMetric"] label p { font-size: .76rem; color: #64748b; text-transform: uppercase; letter-spacing: .04em; }
.verdict { border-radius: 14px; padding: 16px 18px; color: #fff; display: flex; align-items: center; gap: 14px; }
.verdict.alert { background: linear-gradient(120deg, #c2410c, #eb6834); box-shadow: 0 10px 30px -12px rgba(235,104,52,.6); }
.verdict.ok { background: linear-gradient(120deg, #0f7a53, #1baf7a); box-shadow: 0 10px 30px -12px rgba(27,175,122,.6); }
.verdict .msr { font-size: 38px; }
.verdict .big { font-size: 1.45rem; font-weight: 800; letter-spacing: -.01em; line-height: 1.1; }
.verdict .small { font-size: .84rem; opacity: .92; margin-top: 2px; }
.reasons { display: flex; flex-direction: column; gap: 7px; margin-top: 4px; }
.reason-row { display: flex; gap: 9px; align-items: flex-start; background: #fff7f2; border: 1px solid #fbd9c5; border-radius: 10px;
              padding: 8px 11px; font-size: .86rem; color: #7c2d12; }
.reason-row .msr { font-size: 18px; color: #eb6834; }
.none-row { background: #f0fbf6; border-color: #c4ecd9; color: #0f5f43; }
.none-row .msr { color: #1baf7a; }
.meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
.meta div { background: #f6f8fb; border: 1px solid #e3e8ef; border-radius: 10px; padding: 8px 10px; font-size: .76rem; color: #64748b; }
.meta b { display: block; font-size: 1rem; color: #0f1b2d; }
.searchbar { display: flex; align-items: center; gap: 10px; }
.st-key-q input { font-size: .98rem; padding-left: 6px; }
.res-head { display: flex; justify-content: space-between; align-items: center; margin: 2px 0 8px 0; }
.res-head .n { font-size: .9rem; color: #475569; }
.res-head .n b { color: #0f1b2d; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: .76rem; font-weight: 600; }
.pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill.fraud { background: #fdecec; color: #c62828; }
.pill.fp { background: #fff4dc; color: #a36a00; }
.pill.ok { background: #e7f7ef; color: #0f7a53; }
.pill.neutral { background: #eef2f7; color: #475569; }
.drawer { background: #fff; border: 1px solid #e3e8ef; border-radius: 14px; overflow: hidden;
          box-shadow: 0 1px 2px rgba(15,27,45,.04), 0 12px 32px -18px rgba(15,27,45,.25); }
.drawer .top { background: linear-gradient(135deg, #0f1b2d, #1d3557); color: #fff; padding: 16px 18px; }
.drawer .ref { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; color: #9fb3cf; }
.drawer .amt { font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; margin-top: 4px; }
.drawer .when { font-size: .84rem; color: #c8d4e4; }
.drawer .body { padding: 14px 18px 16px 18px; }
.drawer .sec { font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #94a3b8; margin: 12px 0 6px 0; }
.kv { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
.kv div { font-size: .78rem; color: #64748b; }
.kv b { display: block; font-size: .92rem; color: #0f1b2d; font-weight: 600; }
.riskbar { height: 8px; border-radius: 999px; background: #eef2f7; overflow: hidden; margin-top: 4px; }
.riskbar span { display: block; height: 100%; border-radius: 999px; }
.sig { display: flex; gap: 8px; align-items: center; font-size: .84rem; color: #334155; padding: 5px 0; border-bottom: 1px dashed #e8edf3; }
.sig:last-child { border-bottom: 0; }
.sig .msr { font-size: 17px; }
.empty { text-align: center; padding: 40px 10px; color: #94a3b8; font-size: .9rem; }
.empty .msr { font-size: 40px; display: block; margin-bottom: 6px; color: #cbd5e1; }
.foot { margin-top: 28px; padding-top: 14px; border-top: 1px solid #e2e8f0; font-size: .78rem; color: #94a3b8;
        display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Accès aux données Gold
# ──────────────────────────────────────────────────────────────────────────────
def _spark_alive(spark: SparkSession) -> bool:
    """La JVM Spark peut être tuée par le système (manque de mémoire) : on vérifie qu'elle répond encore."""
    try:
        return not spark.sparkContext._jsc.sc().isStopped()
    except Exception:
        return False


@st.cache_resource(show_spinner=False, validate=_spark_alive)
def get_spark() -> SparkSession:
    from pyspark import SparkContext

    # Session précédente morte : on oublie ses références pour pouvoir relancer une JVM propre.
    SparkContext._active_spark_context, SparkContext._gateway, SparkContext._jvm = None, None, None
    SparkSession._instantiatedSession, SparkSession._activeSession = None, None
    return (
        SparkSession.builder
        .appName("FraudLakeDashboard")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


class GoldTableMissing(Exception):
    pass


@st.cache_data(ttl=60, show_spinner=False)
def _load_gold_cached(name: str) -> pd.DataFrame:
    """Lève une exception plutôt que de renvoyer None : un échec n'est ainsi jamais mis en cache."""
    try:
        df = get_spark().read.format("delta").load(f"{GOLD}/{name}").toPandas()
    except Exception as exc:
        msg = str(exc)
        if "PATH_NOT_FOUND" in msg or "is not a Delta table" in msg or "DELTA_MISSING_DELTA_TABLE" in msg:
            raise GoldTableMissing(name) from exc
        raise
    for c in df.columns:      # colonnes DECIMAL Spark → float pour Plotly / formatage
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, decimal.Decimal)).any():
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


SPARK_ERRORS: list[str] = []


def load_gold(name: str) -> pd.DataFrame | None:
    """Table Gold en pandas ; None si elle n'existe pas encore. Une panne Spark est retentée une fois puis signalée."""
    for attempt in range(2):
        try:
            return _load_gold_cached(name)
        except GoldTableMissing:
            return None
        except Exception as exc:
            if attempt == 0:
                get_spark.clear()           # JVM probablement morte : nouvelle session au prochain essai
                continue
            SPARK_ERRORS.append(f"{name} : {type(exc).__name__}")
    return None


# ── Formatage français ──────────────────────────────────────────────────────────
def fmt_int(n: float) -> str:
    return f"{n:,.0f}".replace(",", "\u202f")


def fmt_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f} M$".replace(".", ",")
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f} k$".replace(".", ",")
    return f"{x:.0f} $"


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f} %".replace(".", ",")


def fmt_num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


# ── Composants visuels ──────────────────────────────────────────────────────────
def sparkline(values, color: str, w: int = 66, h: int = 26) -> str:
    v = np.asarray(list(values), dtype=float)
    if len(v) < 2 or not np.isfinite(v).all():
        return ""
    lo, hi = v.min(), v.max()
    span = (hi - lo) or 1.0
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 3 - (v - lo) / span * (h - 6)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    area = f"2,{h} {pts} {w - 2},{h}"
    gid = f"g{abs(hash((color, pts))) % 10**8}"
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".22"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="{area}" fill="url(#{gid})"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.6" fill="{color}"/></svg>')


def kpi(col, label: str, value: str, sub: str = "", tone: str = "blue", icon: str = "insights", spark: str = "") -> None:
    col.markdown(
        f'<div class="kpi {tone}"><div class="top"><span class="ico msr">{icon}</span><span class="label">{label}</span></div>'
        f'<div class="vrow"><div class="value">{value}</div>{spark}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(crumb: str, title: str, subtitle: str) -> None:
    chips = ['<span class="chip"><span class="dot"></span>Lakehouse en ligne</span>',
             f'<span class="chip"><span class="msr">database</span>{len(CATALOG)} tables Gold</span>',
             f'<span class="chip"><span class="msr">calendar_month</span>{D_MIN:%d/%m/%Y} → {D_MAX:%d/%m/%Y}</span>']
    if CHAMPION is not None:
        chips.append(f'<span class="chip"><span class="msr">verified</span>Champion v{CHAMPION["version"]} · {CHAMPION["candidate"]}</span>')
    st.markdown(f'<div class="page-head"><div><div class="crumb">{crumb}</div><h1>{title}</h1><p>{subtitle}</p></div>'
                f'<div class="chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)


def insights(items: list[tuple[str, str, str]]) -> None:
    """Bandeau de points clés : (icône Material, texte HTML, ton)."""
    html = "".join(f'<div class="insight {tone}"><span class="msr">{icon}</span><div>{text}</div></div>' for icon, text, tone in items)
    st.markdown(f'<div class="insights">{html}</div>', unsafe_allow_html=True)


def card(title: str, subtitle: str = ""):
    box = st.container(border=True)
    box.markdown(f'<div class="card-head"><div class="t">{title}</div>' + (f'<div class="s">{subtitle}</div>' if subtitle else "")
                 + "</div>", unsafe_allow_html=True)
    return box


def style_fig(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, showlegend=legend, title=None)
    return fig


def show(fig: go.Figure, height: int = 360, legend: bool = True) -> None:
    fig.update_layout(legend_title_text=None)
    if fig.layout.yaxis.type == "log":
        fig.update_yaxes(dtick=1)
    if fig.layout.xaxis.type == "log":
        fig.update_xaxes(dtick=1)
    st.plotly_chart(style_fig(fig, height, legend), width="stretch", theme=None,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})


def explain(table: str, justification: str) -> None:
    """Encadré « Requête SQL & justification » alimenté par le catalogue Gold."""
    with st.expander("Requête SQL & justification statistique", icon=":material/code:"):
        meta = CATALOG.get(table)
        st.markdown(f"**Table source :** `{GOLD}/{table}`" + (f" — {meta['description']}" if meta else ""))
        if meta:
            st.code(meta["sql"], language="sql")
        st.markdown(f"**Justification :** {justification}")


# ──────────────────────────────────────────────────────────────────────────────
# Chargement
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Lecture des tables Gold sur HDFS…"):
    catalog = load_gold("data_catalog")
    daily = load_gold("fraud_daily")
    monthly = load_gold("fraud_monthly")
    dist = load_gold("amount_distribution")
    by_cat = load_gold("fraud_by_category")
    heat = load_gold("fraud_hour_weekday")
    by_state = load_gold("fraud_by_state")
    sample = load_gold("transactions_sample")
    tests = load_gold("stat_tests")
    segments = load_gold("card_segments")
    seg_prof = load_gold("segment_profiles")
    metrics = load_gold("model_metrics")
    curve = load_gold("model_threshold_curve")
    importance = load_gold("model_feature_importance")
    alerts = load_gold("fraud_alerts")
    mlx = {name: load_gold(f"mlops_{name}") for name in
           ("experiments", "model_registry", "fairness", "quality_gates", "drift", "performance_monthly", "model_card")}

CATALOG = {} if catalog is None else catalog.set_index("table_name").to_dict("index")

if SPARK_ERRORS and (daily is None or by_cat is None):
    st.markdown('<div class="page-head"><div><div class="crumb">FraudLake</div><h1>Moteur Spark momentanément indisponible</h1>'
                '<p>Les tables Gold sont bien sur HDFS, mais la session Spark du tableau de bord ne répond pas (souvent un manque de '
                'mémoire Docker). Elle est relancée automatiquement : rechargez la page dans quelques secondes.</p></div></div>',
                unsafe_allow_html=True)
    if st.button("Réessayer maintenant", type="primary", icon=":material/refresh:"):
        get_spark.clear()
        _load_gold_cached.clear()
        st.rerun()
    st.stop()

if daily is None or by_cat is None:
    st.markdown('<div class="page-head"><div><div class="crumb">FraudLake</div><h1>Tables Gold indisponibles</h1>'
                f'<p>Aucune table trouvée dans <code>{GOLD}</code>.</p></div></div>', unsafe_allow_html=True)
    st.markdown(
        """
**Pour alimenter ce tableau de bord :**
1. `J1_HDFS_Ingestion_Bronze.ipynb` — ingestion des 2 datasets Kaggle dans `/lakehouse/bronze/`
2. `J2_Spark_Delta_Silver.ipynb` — table `/lakehouse/silver/transactions_enriched`
3. `J3_Analytics_SQL_Gold.ipynb` — tables Gold analytiques (KPI, Pareto, séries temporelles, géographie)
4. `J4_Stats_Segmentation_Dashboard.ipynb` — tests statistiques, segments et modèle de scoring
5. `J5_MLOps_Industrialisation.ipynb` — expériences MLflow, registre, déploiement, file d'alertes et surveillance

Ou en une commande : `task test:notebooks`. La page se rafraîchit automatiquement (cache de 60 s).
"""
    )
    st.stop()

daily["trans_date"] = pd.to_datetime(daily["trans_date"])
D_MIN, D_MAX = daily["trans_date"].min().date(), daily["trans_date"].max().date()

CHAMPION = None
if mlx["model_registry"] is not None and len(mlx["model_registry"]):
    reg = mlx["model_registry"]
    al = reg["aliases"].fillna("")
    c_ = reg[al.str.contains("champion") & ~al.str.contains("previous")]
    CHAMPION = (c_ if len(c_) else reg.sort_values("version")).iloc[-1]

# ──────────────────────────────────────────────────────────────────────────────
# Barre latérale : marque, navigation, filtres
# ──────────────────────────────────────────────────────────────────────────────
PAGES = {
    ":material/space_dashboard: Vue d'ensemble": "overview",
    ":material/manage_search: Recherche de transactions": "search",
    ":material/category: Catégories": "categories",
    ":material/schedule: Temporalité": "time",
    ":material/public: Géographie": "geo",
    ":material/groups: Segments clients": "segments",
    ":material/model_training: Modèle de scoring": "model",
    ":material/bolt: Scoring temps réel": "live",
    ":material/science: Tests statistiques": "tests",
    ":material/notifications_active: File d'alertes": "alerts",
    ":material/deployed_code: MLOps": "mlops",
    ":material/account_tree: Données & gouvernance": "data",
}

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo msr">shield_lock</div><div><div class="name">FraudLake</div>'
                '<div class="tag">Fraud Intelligence</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
    slugs = list(PAGES.values())
    wanted_page = st.query_params.get("page", "overview")          # lien direct : ?page=mlops
    page = PAGES[st.radio("Navigation", list(PAGES), label_visibility="collapsed", key="page",
                          index=slugs.index(wanted_page) if wanted_page in slugs else 0)]
    st.query_params["page"] = page

    st.markdown('<div class="side-label">Filtres</div>', unsafe_allow_html=True)
    period = st.date_input("Période d'analyse", value=(D_MIN, D_MAX), min_value=D_MIN, max_value=D_MAX, format="DD/MM/YYYY")
    start, end = (period if isinstance(period, tuple) and len(period) == 2 else (D_MIN, D_MAX))
    all_cats = by_cat.sort_values("fraud_amount", ascending=False)["category"].tolist()
    if st.toggle("Toutes les catégories", value=True):
        cats = all_cats
    else:
        cats = st.multiselect("Catégories de commerçants", all_cats, default=all_cats[:4],
                              help="Filtre les vues catégories et l'échantillon de transactions.") or all_cats
    review_cost = st.number_input("Coût d'investigation d'une alerte ($)", min_value=0.0, max_value=200.0, value=8.0, step=1.0)

    st.markdown(
        '<div class="side-foot"><b>Sources</b><br>kartik2112/fraud-detection<br>donnetew/us-holiday-dates-2004-2021<br><br>'
        f'<b>Lakehouse</b><br>{GOLD} · cache 60 s<br><br>'
        '<b>Projet</b><br>Yan-Mol LANGWANA · Rabab KOULI<br>Prof. Antoine GREA</div>', unsafe_allow_html=True)

daily_f = daily[(daily["trans_date"].dt.date >= start) & (daily["trans_date"].dt.date <= end)]
cat_f = by_cat[by_cat["category"].isin(cats)]
BASE_RATE = 100 * daily["n_fraud"].sum() / daily["n_txn"].sum()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Vue d'ensemble
# ──────────────────────────────────────────────────────────────────────────────
def page_overview() -> None:
    page_header("Tableau de bord exécutif", "Vue d'ensemble de la fraude",
                "1,85 M transactions par carte bancaire (2019 – 2020) consolidées dans un Lakehouse Médaillon HDFS + Delta Lake.")
    n_txn, n_fraud = daily_f["n_txn"].sum(), daily_f["n_fraud"].sum()
    loss, volume = daily_f["fraud_amount"].sum(), daily_f["total_amount"].sum()
    mf = daily_f.set_index("trans_date").resample("MS")[["n_txn", "n_fraud", "fraud_amount"]].sum()
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Transactions", fmt_int(n_txn), f"{start:%d/%m/%Y} → {end:%d/%m/%Y}", "blue", "receipt_long",
        sparkline(mf["n_txn"], LEGIT))
    kpi(c2, "Fraudes", fmt_int(n_fraud), f"taux {fmt_pct(100 * n_fraud / max(n_txn, 1), 3)}", "orange", "gpp_maybe",
        sparkline(mf["n_fraud"], FRAUD))
    kpi(c3, "Montant fraudé", fmt_money(loss), f"{fmt_pct(100 * loss / max(volume, 1))} du volume", "orange", "payments",
        sparkline(mf["fraud_amount"], FRAUD))
    if dist is not None:
        d = dist.set_index("classe")
        kpi(c4, "Panier médian", f"{d.loc['Fraude', 'median_amt']:.0f} $ <span style='font-size:.9rem;color:#94a3b8;font-weight:600'>"
            f"vs {d.loc['Légitime', 'median_amt']:.0f} $</span>",
            f"fraude ×{fmt_num(d.loc['Fraude', 'median_amt'] / d.loc['Légitime', 'median_amt'], 1)} la légitime", "amber", "shopping_cart")
    if len(daily_f):
        peak = daily_f.loc[daily_f["n_fraud"].idxmax()]
        kpi(c5, "Jour le plus touché", f"{peak['trans_date']:%d/%m/%Y}",
            f"{int(peak['n_fraud'])} fraudes · {fmt_money(peak['fraud_amount'])}", "slate", "event_busy")
    st.write("")

    items = []
    if by_cat is not None:
        a = by_cat[by_cat["pareto_class"].str.startswith("A")]
        items.append(("donut_large", f"<b>{len(a)} catégories sur {len(by_cat)}</b> concentrent "
                      f"<b>{fmt_pct(a['fraud_amount'].sum() / by_cat['fraud_amount'].sum() * 100, 0)}</b> des montants fraudés.", ""))
    if heat is not None:
        night = heat["hour"].isin([22, 23, 0, 1, 2, 3])
        rn = heat.loc[night, "n_fraud"].sum() / heat.loc[night, "n_txn"].sum()
        rd = heat.loc[~night, "n_fraud"].sum() / heat.loc[~night, "n_txn"].sum()
        items.append(("dark_mode", f"La nuit (22 h – 4 h), le taux de fraude est <b>×{fmt_num(rn / rd, 1)}</b> celui de la journée.", "blue"))
    if CHAMPION is not None:
        items.append(("verified", f"Le modèle <b>{CHAMPION['candidate']}</b> détecte <b>{fmt_pct(100 * CHAMPION['test_recall'], 0)}</b> "
                      f"des fraudes avec <b>{fmt_pct(100 * CHAMPION['test_precision'], 0)}</b> de précision.", "green"))
    insights(items)

    with card("Fraudes quotidiennes et tendance hebdomadaire",
              "Barres : fraudes par jour · courbe : moyenne glissante 7 jours · losanges : jours fériés US (source n°2)"):
        fig = go.Figure()
        fig.add_bar(x=daily_f["trans_date"], y=daily_f["n_fraud"], name="Fraudes / jour", marker_color="rgba(235,104,52,.28)",
                    marker_line_width=0, hovertemplate="%{x|%d/%m/%Y}<br><b>%{y}</b> fraudes<extra></extra>")
        fig.add_scatter(x=daily_f["trans_date"], y=daily_f["n_fraud_ma7"], name="Moyenne glissante 7 j", mode="lines",
                        line=dict(color=FRAUD, width=2.4, shape="spline", smoothing=0.6),
                        hovertemplate="%{x|%d/%m/%Y}<br>MM7 : <b>%{y:.1f}</b><extra></extra>")
        hol = daily_f[daily_f["is_holiday"] == 1]
        fig.add_scatter(x=hol["trans_date"], y=hol["n_fraud"], mode="markers", name="Jour férié",
                        marker=dict(color=NAVY, size=9, symbol="diamond", line=dict(color="white", width=1.5)),
                        customdata=hol[["holiday_name"]], hovertemplate="%{x|%d/%m/%Y} — %{customdata[0]}<br>%{y} fraudes<extra></extra>")
        fig.update_layout(bargap=0, hovermode="x unified", yaxis_title="fraudes")
        show(fig, 380)
        explain("fraud_daily", "La série brute est très bruitée (effet jour de semaine) : la moyenne glissante sur 7 jours "
                "`AVG() OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` révèle les vagues de fraude. Les jours fériés proviennent "
                "de la jointure avec le second dataset (calendrier US).")

    col_a, col_b = st.columns([1.25, 1])
    with col_a, card("Taux de fraude mensuel", "Un taux, pas un volume : neutralise la saisonnalité des achats"):
        m = monthly.copy()
        colors = [FRAUD if v >= m["fraud_rate_pct"].quantile(0.75) else "#f5b48c" for v in m["fraud_rate_pct"]]
        fig = go.Figure(go.Bar(x=m["trans_month"], y=m["fraud_rate_pct"], marker_color=colors,
                               customdata=m[["n_fraud", "mom_fraud_growth_pct", "fraud_amount"]],
                               hovertemplate="<b>%{x}</b><br>taux : %{y:.3f} %<br>%{customdata[0]} fraudes"
                               "<br>vs mois précédent : %{customdata[1]:+.1f} %<br>pertes : %{customdata[2]:,.0f} $<extra></extra>"))
        fig.add_hline(y=BASE_RATE, line_dash="dot", line_color=MUTED, annotation_text=f"moyenne {fmt_pct(BASE_RATE)}",
                      annotation_font=dict(size=11, color=INK_2))
        fig.update_layout(yaxis_title="taux de fraude (%)", bargap=0.25)
        show(fig, 340, legend=False)
        explain("fraud_monthly", "On suit un **taux** et non un nombre : le volume total de transactions est saisonnier "
                "(pic de décembre), le nombre brut de fraudes surestimerait le risque en fin d'année. `LAG(…, 1)` donne la "
                "variation mois sur mois (survol). Les barres foncées signalent le quartile de mois les plus exposés.")
    with col_b, card("Distribution des montants", "Quartiles, P95 et moyenne (pointillé) — échelle logarithmique"):
        if dist is not None:
            fig = go.Figure()
            for cls, color in [("Légitime", LEGIT), ("Fraude", FRAUD)]:
                r = dist.set_index("classe").loc[cls]
                fig.add_trace(go.Box(name=cls, x=[cls], q1=[r["p25"]], median=[r["median_amt"]], q3=[r["p75"]],
                                     lowerfence=[max(r["p25"] - 1.5 * r["iqr"], 1)], upperfence=[r["p95"]],
                                     mean=[r["mean_amt"]], marker_color=color, boxmean=True, line_width=2))
            fig.update_layout(yaxis_type="log", yaxis_title="montant ($, log)")
            show(fig, 340, legend=False)
            explain("amount_distribution", "Skewness ≈ 45 côté légitime : la moyenne est tirée par quelques achats "
                    "exceptionnels. La **médiane** et l'**IQR** (boîte) sont les indicateurs retenus ; la moustache haute "
                    "s'arrête au P95 calculé par `PERCENTILE_APPROX`.")


# ──────────────────────────────────────────────────────────────────────────────
# 1 bis. Recherche de transactions (console de type backoffice)
# ──────────────────────────────────────────────────────────────────────────────
STATUS_FRAUD, STATUS_FP, STATUS_OK = "Fraude confirmée", "Alerte non confirmée", "Légitime"
STATUS_CSS = {STATUS_FRAUD: "fraud", STATUS_FP: "fp", STATUS_OK: "ok"}
STATUS_SHORT = {STATUS_FRAUD: "🔴 Fraude", STATUS_FP: "🟠 Alerte", STATUS_OK: "🟢 Légitime"}
HIGH_RISK = {"shopping_net", "misc_net", "grocery_pos", "shopping_pos"}


@st.cache_data(ttl=60, show_spinner=False)
def search_base(_sample: pd.DataFrame, _alerts: pd.DataFrame | None) -> pd.DataFrame:
    df = _sample.copy()
    df["trans_ts"] = pd.to_datetime(df["trans_ts"])
    if _alerts is not None:
        a = _alerts[["trans_num", "fraud_score", "priority", "reasons", "merchant"]].copy()
        a["reasons"] = a["reasons"].apply(lambda r: list(r) if r is not None else [])
        df = df.merge(a, on="trans_num", how="left")
    else:
        df["fraud_score"], df["priority"], df["reasons"], df["merchant"] = np.nan, None, None, None
    df["alerted"] = df["fraud_score"].notna()
    df["statut"] = np.select([df["is_fraud"] == 1, df["alerted"]], [STATUS_FRAUD, STATUS_FP], STATUS_OK)
    df["canal"] = np.where(df["is_online"] == 1, "En ligne", "Point de vente")
    df["ref"] = "TX-" + df["trans_num"].str[:10].str.upper()
    return df.sort_values("trans_ts", ascending=False).reset_index(drop=True)


def risk_signals(r: pd.Series) -> list[tuple[str, str, bool]]:
    """Signaux lisibles (mêmes seuils que fraudlake/explain.py)."""
    return [
        ("payments", f"Montant {fmt_num(r['amt'], 2)} $" + (" — au-delà de 500 $" if r["amt"] >= 500 else ""), r["amt"] >= 500),
        ("trending_up", f"{fmt_num(r['amt_ratio_to_card_mean'], 1)} × le panier habituel de la carte", r["amt_ratio_to_card_mean"] >= 4),
        ("dark_mode", f"Heure : {int(r['hour'])} h" + (" — plage nocturne" if r["is_night"] == 1 else ""), r["is_night"] == 1),
        ("category", f"Catégorie {r['category']}" + (" — à risque élevé" if r["category"] in HIGH_RISK else ""), r["category"] in HIGH_RISK),
        ("bolt", f"{int(r['txn_count_24h'])} transactions de la carte sur 24 h", r["txn_count_24h"] >= 6),
        ("near_me", f"{fmt_num(r['distance_km'], 0)} km du domicile", r["distance_km"] >= 100),
    ]


def drawer(r: pd.Series) -> None:
    """Fiche détaillée d'une transaction (un seul bloc HTML pour que la carte reste d'un tenant)."""
    status = r["statut"]
    amount = f"{r['amt']:,.2f} $".replace(",", " ").replace(".", ",")
    merchant = r["merchant"] if isinstance(r["merchant"], str) else "—"
    d = int(r["days_to_holiday"])
    holiday = "non" if d >= 4 else ("le jour même" if d == 0 else f"à {d} jour{'s' if d > 1 else ''}")
    html = (
        f'<div class="drawer"><div class="top"><div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span class="ref">{r["ref"]}</span><span class="pill {STATUS_CSS[status]}">{status}</span></div>'
        f'<div class="amt">{amount}</div><div class="when">{r["trans_ts"]:%d/%m/%Y à %H:%M:%S} · {r["canal"]}</div></div>'
        f'<div class="body"><div class="sec">Transaction</div><div class="kv">'
        f'<div>Catégorie<b>{r["category"]}</b></div><div>Commerçant<b>{merchant}</b></div>'
        f'<div>État<b>{r["state"]}</b></div><div>Jour férié proche<b>{holiday}</b></div></div>'
        f'<div class="sec">Porteur (pseudonymisé)</div><div class="kv">'
        f'<div>Âge<b>{int(r["age"])} ans</b></div><div>Genre<b>{"Femme" if r["gender"] == "F" else "Homme"}</b></div>'
        f'<div>Activité 24 h<b>{int(r["txn_count_24h"])} transactions</b></div>'
        f'<div>Distance domicile<b>{fmt_num(r["distance_km"], 0)} km</b></div></div>'
    )
    if r["alerted"]:
        sc = float(r["fraud_score"])
        col = FRAUD if r["priority"] == "critique" else AMBER
        reasons = "".join(f'<div class="sig"><span class="msr" style="color:{FRAUD}">flag</span>{x}</div>' for x in r["reasons"])
        html += (f'<div class="sec">Décision du modèle champion</div>'
                 f'<div style="display:flex;justify-content:space-between;font-size:.84rem;color:#475569">'
                 f'<span>Score de risque · priorité <b style="color:{col}">{r["priority"]}</b></span>'
                 f'<b style="color:#0f1b2d">{fmt_num(sc)}</b></div>'
                 f'<div class="riskbar"><span style="width:{100 * sc:.0f}%;background:{col}"></span></div>'
                 f'<div style="margin-top:8px">{reasons}</div>')
    else:
        sig = "".join(f'<div class="sig"><span class="msr" style="color:{FRAUD if hit else "#9aa8b9"}">'
                      f'{"error" if hit else "check_circle"}</span>{txt}</div>' for _icon, txt, hit in risk_signals(r))
        note = ("hors période de scoring du champion (avant le 21/06/2020)" if r["trans_ts"] < pd.Timestamp("2020-06-21")
                else "score inférieur au seuil : transaction acceptée")
        html += (f'<div class="sec">Signaux de risque</div>{sig}'
                 f'<div style="font-size:.76rem;color:#94a3b8;margin-top:8px">Pas d&#39;alerte du modèle — {note}.</div>')
    st.markdown(html + "</div></div>", unsafe_allow_html=True)


def page_search() -> None:
    page_header("Console des opérations", "Recherche de transactions",
                "Retrouvez une transaction, filtrez par statut, canal ou montant, et ouvrez sa fiche détaillée avec la décision du modèle.")
    if sample is None:
        st.info("Table `transactions_sample` indisponible : exécutez le notebook J3.")
        return
    base = search_base(sample, alerts)

    with st.container(border=True):
        c1, c3, c2 = st.columns([3.6, 1.7, 1], vertical_alignment="bottom")
        q = c1.text_input("Recherche", key="q", placeholder="Référence (TX-…), commerçant, catégorie ou État — ex. shopping_net, TX, Kutch",
                          label_visibility="collapsed", icon=":material/search:")
        if c2.button("Réinitialiser", icon=":material/restart_alt:", width="stretch"):
            for k in ("q", "f_status", "f_canal", "f_amt", "f_dates", "f_cats"):
                st.session_state.pop(k, None)
            st.rerun()
        f1, f2, f3 = st.columns([2.3, 1.4, 1.6])
        statuses = f1.pills("Statut", [STATUS_FRAUD, STATUS_FP, STATUS_OK], selection_mode="multi", key="f_status") or []
        canal = f2.pills("Canal", ["En ligne", "Point de vente"], selection_mode="multi", key="f_canal") or []
        amax = float(np.ceil(base["amt"].max() / 100) * 100)
        amt_lo, amt_hi = f3.slider("Montant ($)", 0.0, amax, (0.0, amax), step=10.0, format="%.0f $", key="f_amt")
        dmin, dmax = base["trans_ts"].min().date(), base["trans_ts"].max().date()
        dates = c3.date_input("Période", value=(dmin, dmax), min_value=dmin, max_value=dmax, format="DD/MM/YYYY", key="f_dates",
                              label_visibility="collapsed")
        d0, d1 = dates if isinstance(dates, tuple) and len(dates) == 2 else (dmin, dmax)

    v = base[(base["amt"] >= amt_lo) & (base["amt"] <= amt_hi) & (base["trans_ts"].dt.date >= d0) & (base["trans_ts"].dt.date <= d1)
             & base["category"].isin(cats)]
    if statuses:
        v = v[v["statut"].isin(statuses)]
    if canal:
        v = v[v["canal"].isin(canal)]
    if q.strip():
        t = q.strip().lower()
        hay = (v["ref"].str.lower() + " " + v["trans_num"] + " " + v["category"] + " " + v["state"].str.lower() + " "
               + v["merchant"].fillna("").str.lower())
        v = v[hay.str.contains(t, regex=False)]

    w = v["sample_weight"].astype(float)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Résultats", fmt_int(len(v)), f"sur {fmt_int(len(base))} transactions de l'échantillon", "blue", "list_alt")
    kpi(k2, "Volume estimé", fmt_money(float((v["amt"] * w).sum())), "pondéré par le plan de sondage", "slate", "account_balance_wallet")
    kpi(k3, "Fraudes", fmt_int(int(v["is_fraud"].sum())), f"{fmt_money(float(v.loc[v['is_fraud'] == 1, 'amt'].sum()))} fraudés", "orange", "gpp_maybe")
    rate = 100 * float((v["is_fraud"] * w).sum() / w.sum()) if len(v) else 0.0
    kpi(k4, "Taux de fraude estimé", fmt_pct(rate, 2), f"moyenne globale {fmt_pct(BASE_RATE)}", "amber", "percent")
    st.write("")

    left, right = st.columns([1.9, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown(f'<div class="res-head"><span class="n"><b>{fmt_int(len(v))}</b> transactions · plus récentes d&#39;abord</span>'
                    f'<span class="n">Cliquez sur une ligne pour ouvrir sa fiche</span></div>', unsafe_allow_html=True)
        if v.empty:
            st.markdown('<div class="empty"><span class="msr">search_off</span>Aucune transaction ne correspond à ces critères.</div>',
                        unsafe_allow_html=True)
            sel_row = None
        else:
            shown = v.head(2000).assign(statut_aff=lambda d: d["statut"].map(STATUS_SHORT))
            event = st.dataframe(
                shown[["ref", "statut_aff", "trans_ts", "category", "amt", "fraud_score", "state"]],
                hide_index=True, width="stretch", height=560, on_select="rerun", selection_mode="single-row", key="search_table",
                column_config={"ref": st.column_config.TextColumn("référence", width=110),
                               "trans_ts": st.column_config.DatetimeColumn("date", format="DD/MM/YY HH:mm", width=120),
                               "category": "catégorie", "state": st.column_config.TextColumn("État", width=50),
                               "amt": st.column_config.NumberColumn("montant ($)", format="%.2f", width=95),
                               "fraud_score": st.column_config.ProgressColumn("score", format="%.2f", min_value=0.0, max_value=1.0, width=95),
                               "statut_aff": st.column_config.TextColumn("statut", width=100)})
            rows = event.selection.rows if event and event.selection else []
            alerted_idx = np.flatnonzero(shown["alerted"].to_numpy())      # par défaut : la plus récente alerte du modèle
            sel_row = shown.iloc[rows[0]] if rows else shown.iloc[alerted_idx[0] if len(alerted_idx) else 0]
            b1, b2 = st.columns([1, 1])
            b1.caption("Affichage limité aux 2 000 premières lignes" if len(v) > 2000 else "")
            b2.download_button("Exporter les résultats (CSV)",
                               v.drop(columns=["reasons"], errors="ignore").to_csv(index=False).encode("utf-8"),
                               file_name="fraudlake_recherche.csv", mime="text/csv", icon=":material/download:", width="stretch")
    with right:
        if sel_row is not None:
            drawer(sel_row)
    explain("transactions_sample", "Échantillon stratifié de la couche Gold (100 % des fraudes + 2 % des transactions légitimes) joint à la "
            "file d'alertes du champion (`fraud_alerts`, clé `trans_num`). Les volumes et taux sont **pondérés** par `sample_weight` "
            "pour estimer la population ; les comptes de lignes sont ceux de l'échantillon.")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Catégories
# ──────────────────────────────────────────────────────────────────────────────
def page_categories() -> None:
    page_header("Analyse métier", "Où se concentrent les pertes ?",
                "Analyse de Pareto par catégorie de commerçant : la classe A regroupe les catégories générant 80 % des montants fraudés.")
    top = by_cat.sort_values("fraud_amount", ascending=False).iloc[0]
    riskiest = by_cat.sort_values("fraud_rate_pct", ascending=False).iloc[0]
    gap = by_cat.assign(r=by_cat["median_fraud_amt"] / by_cat["median_legit_amt"]).sort_values("r", ascending=False).iloc[0]
    insights([
        ("trophy", f"<b>{top['category']}</b> est la catégorie la plus coûteuse : <b>{fmt_money(top['fraud_amount'])}</b> de pertes.", ""),
        ("warning", f"Taux de fraude le plus élevé : <b>{riskiest['category']}</b> ({fmt_pct(riskiest['fraud_rate_pct'])}).", "blue"),
        ("compare_arrows", f"Écart de panier maximal : <b>{gap['category']}</b> — la fraude y est <b>×{fmt_num(gap['r'], 1)}</b> plus chère.", "green"),
    ])
    col_a, col_b = st.columns([1.2, 1])
    with col_a, card("Pareto des montants fraudés", "Orange : classe A (80 % des pertes) · étiquette : pertes et part cumulée"):
        c = cat_f.sort_values("fraud_amount")
        fig = go.Figure(go.Bar(
            x=c["fraud_amount"], y=c["category"], orientation="h",
            marker_color=[FRAUD if p.startswith("A") else NEUTRAL for p in c["pareto_class"]],
            text=[f"{fmt_money(v)} · {s:.0f} %" for v, s in zip(c["fraud_amount"], c["cum_loss_share_pct"], strict=True)],
            textposition="outside", cliponaxis=False, textfont=dict(size=11, color=INK_2),
            customdata=c[["n_fraud", "fraud_rate_pct", "pareto_class"]],
            hovertemplate="<b>%{y}</b><br>pertes : %{x:,.0f} $<br>%{customdata[0]} fraudes (taux %{customdata[1]:.2f} %)"
                          "<br>%{customdata[2]}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="montant fraudé ($)", xaxis_range=[0, c["fraud_amount"].max() * 1.42], bargap=0.3)
        show(fig, 470, legend=False)
    with col_b, card("Panier médian : fraude vs légitime", "Chaque ligne relie les deux médianes d'une catégorie"):
        c = cat_f.sort_values("median_fraud_amt")
        fig = go.Figure()
        for _, r in c.iterrows():
            fig.add_shape(type="line", x0=r["median_legit_amt"], x1=r["median_fraud_amt"], y0=r["category"], y1=r["category"],
                          line=dict(color="#cbd5e1", width=3))
        fig.add_scatter(x=c["median_legit_amt"], y=c["category"], mode="markers", name="Médiane légitime",
                        marker=dict(color=LEGIT, size=11, line=dict(color="white", width=2)))
        fig.add_scatter(x=c["median_fraud_amt"], y=c["category"], mode="markers", name="Médiane fraude",
                        marker=dict(color=FRAUD, size=11, line=dict(color="white", width=2)))
        fig.update_layout(xaxis_type="log", xaxis_title="montant ($, log)")
        show(fig, 470)
    explain("fraud_by_category", "Les pertes cumulées sont calculées par `SUM(fraud_amount) OVER (ORDER BY fraud_amount DESC "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / SUM(fraud_amount) OVER ()`. Les médianes par classe neutralisent "
            "les montants extrêmes ; l'écart fraude/légitime varie fortement selon la catégorie (Kruskal-Wallis, onglet Tests), "
            "d'où des seuils d'alerte **par catégorie**.")

    if sample is not None:
        with card("Distribution des montants par catégorie et par classe", "Échantillon stratifié : 100 % des fraudes + 2 % des légitimes"):
            s = sample[sample["category"].isin(cats)].assign(classe=lambda d: np.where(d["is_fraud"] == 1, "Fraude", "Légitime"))
            fig = px.box(s, x="category", y="amt", color="classe", points=False, log_y=True,
                         color_discrete_map={"Légitime": LEGIT, "Fraude": FRAUD},
                         category_orders={"category": cat_f.sort_values("fraud_amount", ascending=False)["category"].tolist(),
                                          "classe": ["Légitime", "Fraude"]})
            fig.update_layout(xaxis_title=None, yaxis_title="montant ($, log)", boxgap=0.25, boxgroupgap=0.12)
            show(fig, 420)
            explain("transactions_sample", "Échantillon stratifié pondéré par `sample_weight`. Les boîtes comparent les "
                    "distributions **à l'intérieur** de chaque classe ; pour des totaux population, utiliser la pondération.")

    with st.expander("Tableau détaillé par catégorie", icon=":material/table:"):
        st.dataframe(
            cat_f[["loss_rank", "category", "n_txn", "n_fraud", "fraud_rate_pct", "fraud_amount", "loss_rate_pct",
                   "median_legit_amt", "median_fraud_amt", "iqr_fraud_amt", "cum_loss_share_pct", "pareto_class"]],
            hide_index=True, width="stretch",
            column_config={"loss_rank": st.column_config.NumberColumn("rang"),
                           "category": "catégorie", "n_txn": st.column_config.NumberColumn("transactions", format="%d"),
                           "n_fraud": st.column_config.NumberColumn("fraudes", format="%d"),
                           "fraud_rate_pct": st.column_config.ProgressColumn("taux fraude %", format="%.2f", min_value=0,
                                                                             max_value=float(by_cat["fraud_rate_pct"].max())),
                           "fraud_amount": st.column_config.NumberColumn("pertes ($)", format="%.0f"),
                           "cum_loss_share_pct": st.column_config.ProgressColumn("cumul pertes %", format="%.0f", min_value=0, max_value=100),
                           "pareto_class": "classe"},
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Temporalité
# ──────────────────────────────────────────────────────────────────────────────
def page_time() -> None:
    page_header("Analyse temporelle", "Quand la fraude frappe-t-elle ?",
                "Croisement heure × jour de la semaine : c'est le taux, et non le volume, qui doit piloter l'authentification renforcée.")
    if heat is None:
        st.info("Table `fraud_hour_weekday` indisponible.")
        return
    hourly = heat.groupby("hour").agg(n_txn=("n_txn", "sum"), n_fraud=("n_fraud", "sum")).reset_index()
    hourly["rate"] = 100 * hourly["n_fraud"] / hourly["n_txn"]
    pk = hourly.loc[hourly["rate"].idxmax()]
    night = hourly["hour"].isin([22, 23, 0, 1, 2, 3])
    insights([
        ("dark_mode", f"Heure la plus risquée : <b>{int(pk['hour'])} h</b> avec un taux de <b>{fmt_pct(pk['rate'])}</b>.", ""),
        ("pie_chart", f"La nuit pèse <b>{fmt_pct(100 * hourly.loc[night, 'n_txn'].sum() / hourly['n_txn'].sum(), 0)}</b> des transactions "
                      f"mais <b>{fmt_pct(100 * hourly.loc[night, 'n_fraud'].sum() / hourly['n_fraud'].sum(), 0)}</b> des fraudes.", "blue"),
        ("lock_clock", "Recommandation : <b>authentification forte</b> (3-D Secure) systématique entre 22 h et 4 h.", "green"),
    ])
    with card("Carte de chaleur heure × jour", "Choisissez l'indicateur ; survolez une cellule pour la valeur exacte"):
        metric_label = st.segmented_control("Indicateur", ["Taux de fraude (%)", "Nombre de fraudes", "Montant fraudé ($)"],
                                            default="Taux de fraude (%)", label_visibility="collapsed") or "Taux de fraude (%)"
        metric_col = {"Taux de fraude (%)": "fraud_rate_pct", "Nombre de fraudes": "n_fraud", "Montant fraudé ($)": "fraud_amount"}[metric_label]
        pivot = heat.pivot(index="weekday_label", columns="hour", values=metric_col).reindex(WEEKDAYS)
        fig = px.imshow(pivot, color_continuous_scale=ORANGES, aspect="auto", labels=dict(x="heure", y="", color=""))
        fig.update_xaxes(dtick=1, showgrid=False, ticks="")
        fig.update_yaxes(ticks="")
        fig.update_traces(hovertemplate="%{y} · %{x} h<br>" + metric_label + " : <b>%{z:,.3f}</b><extra></extra>", xgap=3, ygap=3)
        show(fig, 340, legend=False)

    col_a, col_b = st.columns(2)
    with col_a, card("Taux de fraude par heure", "Zones ombrées : 22 h – 4 h"):
        fig = go.Figure(go.Scatter(x=hourly["hour"], y=hourly["rate"], mode="lines+markers",
                                   line=dict(color=FRAUD, width=2.4), marker=dict(size=8, line=dict(color="white", width=1.5)),
                                   fill="tozeroy", fillcolor="rgba(235,104,52,.08)",
                                   hovertemplate="%{x} h : <b>%{y:.2f} %</b><extra></extra>"))
        fig.add_vrect(x0=-0.5, x1=3.5, fillcolor=NAVY, opacity=0.05, line_width=0)
        fig.add_vrect(x0=21.5, x1=23.5, fillcolor=NAVY, opacity=0.05, line_width=0)
        fig.update_layout(xaxis=dict(dtick=2, title="heure"), yaxis_title="taux de fraude (%)")
        show(fig, 320, legend=False)
    with col_b, card("Volume total de transactions par heure", "L'activité est diurne : peu de transactions la nuit"):
        fig = go.Figure(go.Bar(x=hourly["hour"], y=hourly["n_txn"], marker_color=[NEUTRAL if n else LEGIT for n in night],
                               hovertemplate="%{x} h : <b>%{y:,}</b> transactions<extra></extra>"))
        fig.update_layout(xaxis=dict(dtick=2, title="heure"), yaxis_title="transactions", bargap=0.2)
        show(fig, 320, legend=False)
    explain("fraud_hour_weekday", "Agrégation `GROUP BY weekday_num, hour` sur la Silver ; le taux `AVG(is_fraud)` est "
            "comparable entre cellules de volumes différents. Le test du χ² (onglet Tests) confirme la dépendance nuit × fraude.")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Géographie
# ──────────────────────────────────────────────────────────────────────────────
def page_geo() -> None:
    page_header("Analyse géographique", "Exposition au risque par État",
                "Le classement utilise la borne basse de l'intervalle de Wilson pour ne pas surévaluer les États à faible volume.")
    if by_state is None:
        st.info("Table `fraud_by_state` indisponible.")
        return
    c1, c2 = st.columns([1.4, 1])
    geo_metric = c1.segmented_control("Indicateur cartographié", ["Borne basse Wilson", "Taux brut", "Montant fraudé"],
                                      default="Borne basse Wilson") or "Borne basse Wilson"
    min_txn = c2.slider("Volume minimum de transactions par État", 0, 20000, 2000, step=500)
    geo = by_state[by_state["n_txn"] >= min_txn]
    col = {"Borne basse Wilson": "wilson_low_pct", "Taux brut": "fraud_rate_pct", "Montant fraudé": "fraud_amount"}[geo_metric]
    col_a, col_b = st.columns([1.5, 1])
    with col_a, card(f"{geo_metric} par État", f"{len(geo)} États avec au moins {fmt_int(min_txn)} transactions"):
        fig = px.choropleth(geo, locations="state", locationmode="USA-states", scope="usa", color=col,
                            color_continuous_scale=ORANGES,
                            hover_data={"state": True, "n_txn": ":,", "n_fraud": ":,", "fraud_rate_pct": ":.3f",
                                        "wilson_low_pct": ":.3f", "wilson_high_pct": ":.3f", "top_fraud_category": True},
                            labels={col: geo_metric, "n_txn": "transactions", "n_fraud": "fraudes", "fraud_rate_pct": "taux %",
                                    "wilson_low_pct": "IC bas %", "wilson_high_pct": "IC haut %", "top_fraud_category": "catégorie n°1"})
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showlakes=False, subunitcolor="white", landcolor="#eef2f7")
        fig.update_traces(marker_line_color="white", marker_line_width=0.8)
        fig.update_layout(coloraxis_colorbar=dict(title=None, len=0.7))
        show(fig, 430, legend=False)
    with col_b, card("Top 12 États", "Taux observé (point) et intervalle de confiance à 95 % (barre)"):
        top = geo.sort_values("wilson_low_pct", ascending=False).head(12).iloc[::-1]
        fig = go.Figure()
        fig.add_bar(x=top["wilson_high_pct"] - top["wilson_low_pct"], base=top["wilson_low_pct"], y=top["state"], orientation="h",
                    marker_color="rgba(235,104,52,.22)", name="IC Wilson 95 %", hoverinfo="skip")
        fig.add_scatter(x=top["fraud_rate_pct"], y=top["state"], mode="markers", name="taux observé",
                        marker=dict(color=FRAUD, size=10, line=dict(color="white", width=1.5)),
                        customdata=top[["n_txn", "top_fraud_category"]],
                        hovertemplate="<b>%{y}</b> : %{x:.3f} %<br>%{customdata[0]:,} transactions<br>catégorie n°1 : %{customdata[1]}<extra></extra>")
        fig.update_layout(xaxis_title="taux de fraude (%)", barmode="overlay", bargap=0.35)
        show(fig, 430)
    explain("fraud_by_state", "Un État à 3 fraudes sur 9 transactions afficherait 33 % de taux brut : le classement sur la "
            "**borne basse de l'intervalle de Wilson** pénalise les petits effectifs. La catégorie la plus fraudée par État "
            "est obtenue par `ROW_NUMBER() OVER (PARTITION BY state ORDER BY n_fraud DESC) = 1`.")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Segments
# ──────────────────────────────────────────────────────────────────────────────
def page_segments() -> None:
    page_header("Connaissance client", "Segments de porteurs de carte",
                "K-Means sur le comportement légitime de chaque carte : qui sont les clients les plus exposés ?")
    if segments is None or seg_prof is None:
        st.info("Exécutez le notebook J4 pour produire la segmentation des porteurs.")
        return
    seg_names = sorted(seg_prof["segment"])
    cmap = {s: SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i, s in enumerate(seg_names)}
    icons = ["person", "person_pin", "nightlight", "shopping_bag"]
    cols = st.columns(len(seg_prof))
    for i, (col_, (_, r)) in enumerate(zip(cols, seg_prof.sort_values("fraud_rate_pct", ascending=False).iterrows(), strict=False)):
        tone = "orange" if r["fraud_rate_pct"] > BASE_RATE else "blue"
        kpi(col_, r["segment"], fmt_pct(r["fraud_rate_pct"], 2),
            f"{int(r['n_cards'])} cartes · {fmt_pct(r['share_losses_pct'], 0)} des pertes", tone, icons[i % len(icons)])
    st.write("")
    col_a, col_b = st.columns([1.2, 1])
    with col_a, card("Carte des porteurs", "Projection sur les deux premiers axes de l'ACP ; une couleur par segment"):
        fig = px.scatter(segments, x="pc1", y="pc2", color="segment", color_discrete_map=cmap,
                         hover_data={"pc1": False, "pc2": False, "tx_per_month": ":.1f", "median_amt": ":.0f",
                                     "online_share": ":.2f", "night_share": ":.2f", "age": True, "n_fraud": True})
        fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(color="white", width=0.8)))
        fig.update_layout(xaxis_title="composante 1", yaxis_title="composante 2", yaxis=dict(showgrid=True), xaxis=dict(showgrid=True, gridcolor=GRID))
        show(fig, 430)
    with col_b, card("Exposition à la fraude par segment", "Taux de fraude ; pointillé : moyenne globale"):
        sp = seg_prof.sort_values("fraud_rate_pct")
        fig = go.Figure(go.Bar(x=sp["fraud_rate_pct"], y=sp["segment"], orientation="h",
                               marker_color=[cmap[s] for s in sp["segment"]],
                               text=[fmt_pct(v) for v in sp["fraud_rate_pct"]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>taux : %{x:.3f} %<extra></extra>"))
        fig.add_vline(x=BASE_RATE, line_dash="dot", line_color=MUTED, annotation_text=f"moyenne {fmt_pct(BASE_RATE)}",
                      annotation_position="top", annotation_font=dict(size=11, color=INK_2))
        fig.update_layout(xaxis_title="taux de fraude (%)", xaxis_range=[0, sp["fraud_rate_pct"].max() * 1.3], bargap=0.35)
        show(fig, 430, legend=False)
    with st.expander("Profil détaillé des segments", icon=":material/table:"):
        st.dataframe(seg_prof, hide_index=True, width="stretch")
    explain("card_segments", "K-Means sur le comportement **légitime** de chaque carte (fréquence, panier médian, part "
            "e-commerce, part nocturne, âge) après `log1p` des variables asymétriques et `StandardScaler`. Le nombre de "
            "segments est choisi par le score de silhouette ; les noms sont dérivés des centroïdes.")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Modèle de scoring
# ──────────────────────────────────────────────────────────────────────────────
def page_model() -> None:
    page_header("Aide à la décision", "Modèle de scoring & choix du seuil",
                "Simulez le seuil d'alerte : arbitrage entre fraudes détectées, charge d'investigation et gain net.")
    if metrics is None or curve is None:
        st.info("Exécutez le notebook J4 pour entraîner le modèle de scoring.")
        return
    mm = metrics.iloc[0]
    cv = curve.sort_values("threshold").reset_index(drop=True)
    thresholds = cv["threshold"].round(2).tolist()
    default_t = min(thresholds, key=lambda t: abs(t - mm["best_threshold"]))
    with st.container(border=True):
        t = st.select_slider("Seuil de déclenchement d'une alerte (score de risque)", options=thresholds, value=default_t,
                             help="Au-dessus du seuil, la transaction est bloquée / envoyée en investigation.")
    r = cv.loc[cv["threshold"].round(2) == t].iloc[0]
    net = r["amt_caught"] - r["alerts"] * review_cost
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "PR-AUC (test)", fmt_num(mm["pr_auc"]), f"aléatoire = {fmt_num(mm['prevalence'], 4)}", "blue", "query_stats")
    kpi(c2, "Rappel", fmt_pct(100 * r["recall"], 1), f"{fmt_int(r['tp'])} / {fmt_int(r['tp'] + r['fn'])} fraudes détectées", "orange", "radar")
    kpi(c3, "Précision", fmt_pct(100 * r["precision"], 1), f"{fmt_int(r['alerts'])} alertes à traiter", "amber", "target")
    kpi(c4, "Pertes évitées", fmt_money(r["amt_caught"]), f"{fmt_pct(100 * r['amt_caught_share'], 1)} du montant fraudé", "green", "savings")
    kpi(c5, "Gain net estimé", fmt_money(net), f"après {fmt_money(r['alerts'] * review_cost)} d'investigation", "blue", "account_balance")
    st.write("")

    col_a, col_b = st.columns(2)
    with col_a, card("Compromis précision / rappel", "Le point marque le seuil sélectionné"):
        fig = go.Figure()
        fig.add_scatter(x=cv["recall"], y=cv["precision"], mode="lines", line=dict(color=FRAUD, width=2.6),
                        fill="tozeroy", fillcolor="rgba(235,104,52,.07)", name="courbe PR", customdata=cv[["threshold"]],
                        hovertemplate="seuil %{customdata[0]:.2f}<br>rappel %{x:.1%}<br>précision %{y:.1%}<extra></extra>")
        fig.add_scatter(x=[r["recall"]], y=[r["precision"]], mode="markers", name=f"seuil {t:.2f}",
                        marker=dict(color=NAVY, size=14, line=dict(color="white", width=3)))
        fig.add_hline(y=mm["prevalence"], line_dash="dot", line_color=MUTED, annotation_text="modèle aléatoire",
                      annotation_font=dict(size=11, color=INK_2))
        fig.update_layout(xaxis=dict(title="rappel", tickformat=".0%", range=[0, 1.02]),
                          yaxis=dict(title="précision", tickformat=".0%", range=[0, 1.05]))
        show(fig, 380)
    with col_b, card("Gain net selon le seuil", f"Pertes évitées − coût d'investigation ({review_cost:.0f} $ par alerte)"):
        gains = cv["amt_caught"] - cv["alerts"] * review_cost
        fig = go.Figure(go.Scatter(x=cv["threshold"], y=gains, mode="lines", line=dict(color=LEGIT, width=2.6),
                                   fill="tozeroy", fillcolor="rgba(42,120,214,.07)",
                                   hovertemplate="seuil %{x:.2f}<br>gain net <b>%{y:,.0f} $</b><extra></extra>"))
        fig.add_vline(x=t, line_color=NAVY, line_dash="dash", annotation_text="seuil choisi", annotation_position="top left",
                      annotation_font=dict(size=11, color=NAVY))
        fig.add_vline(x=float(cv.loc[gains.idxmax(), "threshold"]), line_color=FRAUD, line_dash="dot",
                      annotation_text="optimum économique", annotation_position="top right", annotation_font=dict(size=11, color=FRAUD))
        fig.update_layout(xaxis_title="seuil", yaxis_title="gain net ($)")
        show(fig, 380, legend=False)

    col_c, col_d = st.columns([1.3, 1])
    with col_c, card("Variables les plus utiles au modèle", "Importance relative (Gradient-Boosted Trees)"):
        if importance is not None:
            imp = importance.sort_values("importance").tail(12)
            fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h",
                                   marker_color=[LEGIT if i >= len(imp) - 3 else "#9cc0ec" for i in range(len(imp))],
                                   hovertemplate="%{y} : %{x:.3f}<extra></extra>"))
            fig.update_layout(xaxis_title="importance relative", bargap=0.3)
            show(fig, 380, legend=False)
    with col_d, card("Matrice de confusion", f"Période de test, seuil {t:.2f}"):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        fig = px.imshow(np.log10(cm + 1), x=["prédit légitime", "prédit fraude"], y=["légitime", "fraude"], color_continuous_scale=BLUES)
        fig.update_traces(text=[[fmt_int(v) for v in row] for row in cm], texttemplate="<b>%{text}</b>", textfont=dict(size=16),
                          customdata=cm, hovertemplate="%{y} / %{x} : %{customdata:,}<extra></extra>", xgap=4, ygap=4)
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(ticks="", side="top")
        fig.update_yaxes(ticks="")
        show(fig, 380, legend=False)
    explain("model_threshold_curve", f"*Gradient-Boosted Trees* (Spark MLlib) entraîné sur {mm['train_period']} et évalué "
            f"sur {mm['test_period']} ({fmt_int(mm['n_test'])} transactions jamais vues). La **PR-AUC** est la métrique "
            "adaptée à une classe rare ; le seuil est un arbitrage **métier** entre pertes évitées et coût d'investigation. "
            "Les scores sont des rangs de risque (entraînement sous-échantillonné), non des probabilités calibrées.")


# ──────────────────────────────────────────────────────────────────────────────
# 6 bis. Scoring temps réel (appel de l'API FastAPI du modèle champion)
# ──────────────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "Achat courant": dict(category="grocery_pos", amt=42.5, gender="F", hour=14, weekday_num=3, age=41, city_pop=25000,
                          distance_km=35.0, txn_count_24h=3, amt_sum_24h=120.0, secs_since_prev_txn=14400,
                          amt_ratio_to_card_mean=0.9, days_to_holiday=4),
    "Achat nocturne en ligne": dict(category="shopping_net", amt=962.4, gender="F", hour=23, weekday_num=6, age=58, city_pop=2380,
                                    distance_km=74.2, txn_count_24h=4, amt_sum_24h=2210.0, secs_since_prev_txn=1260,
                                    amt_ratio_to_card_mean=11.8, days_to_holiday=4),
    "Rafale de petites transactions": dict(category="misc_net", amt=310.0, gender="M", hour=2, weekday_num=5, age=67, city_pop=1200,
                                           distance_km=90.0, txn_count_24h=7, amt_sum_24h=1650.0, secs_since_prev_txn=140,
                                           amt_ratio_to_card_mean=6.2, days_to_holiday=4),
}
JOURS = {1: "lundi", 2: "mardi", 3: "mercredi", 4: "jeudi", 5: "vendredi", 6: "samedi", 7: "dimanche"}


@st.cache_data(ttl=15, show_spinner=False)
def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=4)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def apply_scenario() -> None:
    for k, v in SCENARIOS[st.session_state["scenario"]].items():
        st.session_state[f"tx_{k}"] = v


def page_live() -> None:
    page_header("Démonstration MLOps", "Scoring en temps réel",
                "Une transaction est envoyée à l'API FastAPI qui sert le modèle champion : score, décision et motifs en moins d'une seconde.")
    health, info = api_get("/health"), api_get("/model")
    if not health or not health.get("model_loaded"):
        st.error(f"API de scoring injoignable ({API_URL}). Démarrez le service `fraud-api` : `docker compose up -d fraud-api`.")
        return
    if "tx_amt" not in st.session_state:
        st.session_state["scenario"] = "Achat nocturne en ligne"
        apply_scenario()

    left, right = st.columns([1.15, 1], gap="large")
    with left, card("Transaction à analyser", "Choisissez un scénario puis ajustez les caractéristiques"):
        st.segmented_control("Scénario", list(SCENARIOS), key="scenario", on_change=apply_scenario, label_visibility="collapsed")
        with st.form("tx_form", border=False):
            a, b, c = st.columns(3)
            a.selectbox("Catégorie", sorted(by_cat["category"]), key="tx_category")
            b.number_input("Montant ($)", 0.01, 100000.0, step=10.0, key="tx_amt")
            c.selectbox("Genre", ["F", "M"], key="tx_gender")
            a.slider("Heure", 0, 23, key="tx_hour")
            b.selectbox("Jour", list(JOURS), format_func=JOURS.get, key="tx_weekday_num")
            c.number_input("Âge", 13, 110, key="tx_age")
            st.markdown("<div class='card-head'><div class='s'>Comportement récent de la carte (vélocité)</div></div>", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.number_input("Transactions sur 24 h", 1, 500, key="tx_txn_count_24h")
            b.number_input("Montant cumulé 24 h ($)", 0.0, 1e6, step=50.0, key="tx_amt_sum_24h")
            c.number_input("Ratio à l'habitude", 0.0, 500.0, step=0.5, key="tx_amt_ratio_to_card_mean")
            a.number_input("Secondes depuis la précédente", 0, 10_000_000, step=60, key="tx_secs_since_prev_txn")
            b.number_input("Distance domicile (km)", 0.0, 20000.0, step=5.0, key="tx_distance_km")
            c.number_input("Population de la ville", 1, 10_000_000, step=1000, key="tx_city_pop")
            st.select_slider("Distance au jour férié (jours)", options=[0, 1, 2, 3, 4], key="tx_days_to_holiday",
                             format_func=lambda d: "plus de 3" if d == 4 else str(d))
            submitted = st.form_submit_button("Analyser la transaction", type="primary", icon=":material/bolt:", width="stretch")

    payload = {k: st.session_state[f"tx_{k}"] for k in SCENARIOS["Achat courant"]}
    if submitted or "last_score" not in st.session_state:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{API_URL}/predict", json={"transaction_id": f"demo-{int(time.time())}", **payload}, timeout=30)
            r.raise_for_status()
            res = r.json() | {"latency_ms": (time.perf_counter() - t0) * 1000}
            st.session_state["last_score"] = res
            hist = st.session_state.setdefault("score_history", [])
            hist.insert(0, {"heure": time.strftime("%H:%M:%S"), "catégorie": payload["category"], "montant ($)": payload["amt"],
                            "score": res["fraud_score"], "décision": res["decision"], "priorité": res["priority"]})
            del hist[8:]
        except requests.RequestException as e:
            st.error(f"Échec de l'appel à l'API : {e}")
            return
    res = st.session_state["last_score"]

    with right, card("Verdict du modèle", f"Modèle {info.get('candidate', '?')} v{res['model_version']} · seuil {fmt_num(res['threshold'], 2)}"):
        alert = res["decision"] == "ALERTE"
        st.markdown(
            f'<div class="verdict {"alert" if alert else "ok"}"><span class="msr">{"gpp_bad" if alert else "verified_user"}</span><div>'
            f'<div class="big">{"Transaction bloquée" if alert else "Transaction acceptée"}</div>'
            f'<div class="small">{"Alerte de priorité " + res["priority"] + " envoyée à l&#39;équipe anti-fraude" if alert else "Aucune action requise"}'
            f'</div></div></div>', unsafe_allow_html=True)
        score = 100 * res["fraud_score"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number=dict(suffix=" / 100", font=dict(size=34, color=NAVY), valueformat=".1f"),
            gauge=dict(axis=dict(range=[0, 100], tickwidth=0, tickcolor="white", tickfont=dict(size=11, color=MUTED)),
                       bar=dict(color=FRAUD if alert else GREEN, thickness=0.32), bgcolor="#eef2f7", borderwidth=0,
                       steps=[dict(range=[0, 100 * res["threshold"]], color="#eef6f2"), dict(range=[100 * res["threshold"], 100], color="#fdeee5")],
                       threshold=dict(line=dict(color=NAVY, width=3), thickness=0.9, value=100 * res["threshold"]))))
        fig.update_layout(margin=dict(l=34, r=34, t=24, b=0))
        show(fig, 230, legend=False)
        st.markdown("<div class='card-head'><div class='t' style='font-size:.9rem'>Motifs d'alerte</div></div>", unsafe_allow_html=True)
        rows = "".join(f'<div class="reason-row"><span class="msr">flag</span>{r}</div>' for r in res["reasons"]) or \
            '<div class="reason-row none-row"><span class="msr">check_circle</span>Aucun signal de risque déclenché</div>'
        st.markdown(f'<div class="reasons">{rows}</div>', unsafe_allow_html=True)
        stats = api_get("/stats") or {}
        st.markdown(f'<div class="meta"><div>Latence<b>{fmt_num(res["latency_ms"], 0)} ms</b></div>'
                    f'<div>Requêtes servies<b>{fmt_int(stats.get("requests", 0))}</b></div>'
                    f'<div>PR-AUC test<b>{fmt_num(info.get("test_pr_auc", 0))}</b></div></div>', unsafe_allow_html=True)

    hist = st.session_state.get("score_history", [])
    if hist:
        with card("Historique de la session", "Dernières transactions analysées"):
            st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch",
                         column_config={"score": st.column_config.ProgressColumn("score", format="%.3f", min_value=0.0, max_value=1.0),
                                        "montant ($)": st.column_config.NumberColumn(format="%.2f")})
    with st.expander("Requête envoyée à l'API", icon=":material/code:"):
        st.code(f"POST {API_URL}/predict", language="http")
        st.json(payload)
        st.caption("Documentation interactive de l'API : http://localhost:8010/docs")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Tests statistiques
# ──────────────────────────────────────────────────────────────────────────────
def page_tests() -> None:
    page_header("Rigueur statistique", "Tests d'hypothèses",
                "Avec 1,85 M transactions, presque tout est « significatif » : la décision s'appuie autant sur la taille d'effet "
                "que sur la p-value (α = 5 %).")
    if tests is None:
        st.info("Exécutez le notebook J4 pour produire les tests d'hypothèses.")
        return
    for _, r in tests.iterrows():
        badge = "reject" if r["decision"] == "H0 rejetée" else "keep"
        with st.container(border=True):
            a, b, c, d = st.columns([2.5, 1, 1, 1.3], vertical_alignment="center")
            a.markdown(f"<div class='test-q'>{r['question']}</div><div class='test-d'>{r['test']} — {r['detail']}</div>",
                       unsafe_allow_html=True)
            b.metric("Statistique", f"{r['statistique']:,.1f}".replace(",", " ").replace(".", ","))
            c.metric("p-value", r["p_value_txt"])
            d.markdown(f"<span class='badge {badge}'>{r['decision']}</span><div class='test-d' style='margin-top:6px'>"
                       f"{r['taille_effet']} = <b style='color:#0f1b2d'>{fmt_num(r['valeur_effet'])}</b> · effet {r['intensite_effet']}</div>",
                       unsafe_allow_html=True)
    explain("stat_tests", "Mann-Whitney et Kruskal-Wallis sont des tests de rangs adaptés aux montants très asymétriques ; "
            "les χ² utilisent des tableaux de contingence **exacts** calculés par Spark sur toute la table Silver. "
            "Tailles d'effet : r rang-bisérial, V de Cramér, ε².")


# ──────────────────────────────────────────────────────────────────────────────
# 8. File d'alertes (scoring batch du champion)
# ──────────────────────────────────────────────────────────────────────────────
def reason_key(reason: str) -> str:
    """Libellé générique d'un motif : retire la valeur chiffrée (« Montant 4.6 fois… » → « Montant N fois… »)."""
    return re.sub(r"\d+(?:[.,]\d+)? fois", "N fois", re.sub(r"\s*\(.*\)\s*$", "", reason)).strip()


def page_alerts() -> None:
    page_header("Opérations anti-fraude", "File d'alertes du modèle champion",
                "Transactions scorées par le modèle en production, priorisées et accompagnées de motifs lisibles par un analyste.")
    if alerts is None:
        st.info("Exécutez le notebook J5 pour produire la file d'alertes du modèle champion.")
        return
    al = alerts.copy()
    al["trans_ts"] = pd.to_datetime(al["trans_ts"])
    al["reasons"] = al["reasons"].apply(lambda r: list(r) if r is not None else [])
    crit = al[al["priority"] == "critique"]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Alertes générées", fmt_int(len(al)), f"modèle v{al['model_version'].iloc[0]} · période de test", "orange", "notifications_active")
    kpi(c2, "Priorité critique", fmt_int(len(crit)), f"{fmt_pct(100 * crit['label_confirmed'].mean(), 1)} confirmées fraudes", "amber", "priority_high")
    kpi(c3, "Précision globale", fmt_pct(100 * al["label_confirmed"].mean(), 1), "alertes confirmées par chargeback", "green", "task_alt")
    kpi(c4, "Montant en jeu", fmt_money(al["amt"].sum()),
        f"{fmt_money(al.loc[al['label_confirmed'] == 1, 'amt'].sum())} de fraude réelle", "blue", "paid")
    st.write("")

    perf_ = mlx["performance_monthly"]
    if perf_ is not None:
        p_ = perf_[perf_["period"] != "stress +35 %"]
        tp = float((p_["recall"] * p_["frauds"]).sum())
        n_al, n_fr, n_tx = float(p_["alerts"].sum()), float(p_["frauds"].sum()), float(p_["transactions"].sum())
        fp, fn = n_al - tp, n_fr - tp
        tn = n_tx - n_al - fn
        with card("Flux de détection sur la période de test",
                  f"{fmt_int(n_tx)} transactions scorées par le champion · épaisseur proportionnelle (échelle racine pour la lisibilité)"):
            lab = [f"Transactions<br>{fmt_int(n_tx)}", f"Alertes<br>{fmt_int(n_al)}", f"Acceptées<br>{fmt_int(n_tx - n_al)}",
                   f"Fraudes bloquées<br>{fmt_int(tp)}", f"Faux positifs<br>{fmt_int(fp)}", f"Fraudes manquées<br>{fmt_int(fn)}",
                   f"Légitimes acceptées<br>{fmt_int(tn)}"]
            node_x, node_y = [0.001, 0.45, 0.45, 0.999, 0.999, 0.999, 0.999], [0.5, 0.06, 0.58, 0.02, 0.13, 0.24, 0.66]
            vals = [n_al, n_tx - n_al, tp, fp, fn, tn]
            fig = go.Figure(go.Sankey(
                arrangement="snap",
                node=dict(label=lab, pad=22, thickness=16, line=dict(width=0), x=node_x, y=node_y,
                          color=[NAVY, FRAUD, LEGIT, FRAUD, AMBER, "#e34948", GREEN]),
                link=dict(source=[0, 0, 1, 1, 2, 2], target=[1, 2, 3, 4, 5, 6], value=[np.sqrt(v) for v in vals],
                          customdata=[fmt_int(v) for v in vals], hovertemplate="%{customdata} transactions<extra></extra>",
                          color=["rgba(235,104,52,.35)", "rgba(42,120,214,.22)", "rgba(235,104,52,.45)", "rgba(237,161,0,.4)",
                                 "rgba(227,73,72,.45)", "rgba(27,175,122,.25)"])))
            fig.update_layout(font=dict(size=12, color=NAVY))
            show(fig, 300, legend=False)

    with st.container(border=True):
        f1, f2, f3 = st.columns([1, 1.6, 1])
        prio = f1.pills("Priorité", ["critique", "élevée"], default=["critique", "élevée"], selection_mode="multi") or ["critique", "élevée"]
        reason_codes = sorted({reason_key(r) for rs in al["reasons"] for r in rs})
        wanted = f2.multiselect("Motif d'alerte", reason_codes, placeholder="Tous les motifs",
                                help="Affiche les alertes contenant tous les motifs choisis.")
        min_amt = f3.slider("Montant minimum ($)", 0, int(al["amt"].max()), 0, step=50)
    view = al[al["priority"].isin(prio) & (al["amt"] >= min_amt)]
    if wanted:
        view = view[view["reasons"].apply(lambda rs: set(wanted) <= {reason_key(r) for r in rs})]

    col_a, col_b = st.columns(2)
    with col_a, card("Motifs les plus fréquents", "Règles partagées entre le scoring batch et l'API temps réel"):
        counts = pd.Series([reason_key(r) for rs in view["reasons"] for r in rs], dtype=str).value_counts().sort_values()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=FRAUD,
                               hovertemplate="%{y} : <b>%{x:,}</b> alertes<extra></extra>"))
        fig.update_layout(xaxis_title="alertes", bargap=0.35)
        show(fig, 320, legend=False)
    with col_b, card("Distribution des scores", "Alertes de la sélection, par priorité"):
        fig = px.histogram(view, x="fraud_score", color="priority", nbins=30,
                           color_discrete_map={"critique": FRAUD, "élevée": AMBER}, category_orders={"priority": ["critique", "élevée"]})
        fig.update_layout(xaxis_title="score de risque", yaxis_title="alertes", bargap=0.08)
        show(fig, 320)

    table = view.sort_values("fraud_score", ascending=False).head(300).assign(
        motifs=lambda d: d["reasons"].apply(" · ".join),
        statut=lambda d: np.where(d["label_confirmed"] == 1, "✅ fraude confirmée", "⚪ faux positif"))
    with card(f"{fmt_int(len(view))} alertes dans la sélection", "300 premières par score décroissant"):
        st.dataframe(
            table[["trans_ts", "priority", "fraud_score", "amt", "category", "merchant", "state", "motifs", "statut"]],
            hide_index=True, width="stretch", height=420,
            column_config={"trans_ts": st.column_config.DatetimeColumn("horodatage", format="DD/MM/YYYY HH:mm"),
                           "priority": "priorité", "category": "catégorie", "merchant": "commerçant", "state": "État",
                           "fraud_score": st.column_config.ProgressColumn("score", format="%.3f", min_value=0.0, max_value=1.0),
                           "amt": st.column_config.NumberColumn("montant ($)", format="%.2f"),
                           "motifs": st.column_config.TextColumn("motifs d'alerte", width="large")},
        )
    st.download_button("Exporter la sélection (CSV)", table.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8"),
                       file_name="fraudlake_alertes.csv", mime="text/csv", icon=":material/download:")
    explain("fraud_alerts", "Le modèle champion déployé score les transactions ; au-delà du seuil (choisi sur la validation), "
            "l'alerte reçoit une priorité et des motifs lisibles issus des règles partagées `fraudlake/explain.py` "
            "(mêmes règles dans l'API temps réel). Le statut « confirmée » correspond à l'étiquette connue a posteriori.")


# ──────────────────────────────────────────────────────────────────────────────
# 9. MLOps
# ──────────────────────────────────────────────────────────────────────────────
def page_mlops() -> None:
    page_header("Industrialisation", "Cycle de vie MLOps",
                "Expériences MLflow, portes de qualité et d'équité, registre, déploiement, surveillance de la dérive et décision de ré-entraînement.")
    exp_, reg_, perf_, drift_ = mlx["experiments"], mlx["model_registry"], mlx["performance_monthly"], mlx["drift"]
    if exp_ is None or reg_ is None:
        st.info("Exécutez le notebook J5 (MLOps) pour alimenter cette page.")
        return
    champ = CHAMPION
    steps = [("Expériences", "MLflow Tracking"), ("Portes qualité", "performance · équité"), ("Registre", "@champion / @challenger"),
             ("Déploiement", "HDFS + API :8010"), ("Scoring", "file d'alertes"), ("Surveillance", "PSI · performance")]
    st.markdown('<div class="lifecycle">' + "".join(f'<div class="step"><div class="n">{i}</div><b>{a}</b><span>{b}</span></div>'
                                                    for i, (a, b) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)
    last = perf_[perf_["period"] != "stress +35 %"].iloc[-1] if perf_ is not None else None
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Champion en production", f"v{champ['version']}", f"{champ['candidate']} · données Silver v{champ['data_version']}", "blue", "rocket_launch")
    kpi(c2, "PR-AUC test", fmt_num(champ["test_pr_auc"]), f"seuil {fmt_num(champ['threshold'], 2)}", "green", "query_stats")
    kpi(c3, "Rappel · précision", f"{fmt_pct(100 * champ['test_recall'], 0)} · {fmt_pct(100 * champ['test_precision'], 0)}",
        "au seuil de décision", "orange", "radar")
    if last is not None:
        kpi(c4, f"Surveillance {last['period']}", "Ré-entraîner" if last["retrain"] else "Stable",
            f"PSI max {fmt_num(last['max_feature_psi'], 2)} · rappel {fmt_pct(100 * last['recall'], 0)}",
            "amber" if last["retrain"] else "green", "monitor_heart")
    kpi(c5, "Candidats évalués", str(len(exp_)), f"{int(exp_['gates_passed'].sum())} conformes aux portes", "slate", "science")
    st.caption("Interfaces : MLflow http://localhost:5050 · API de scoring http://localhost:8010/docs")

    col_a, col_b = st.columns([1.2, 1])
    with col_a, card("Expériences MLflow", "PR-AUC par candidat, validation et test"):
        e = exp_.sort_values("valid_pr_auc")
        fig = go.Figure()
        fig.add_bar(y=e["candidate"], x=e["valid_pr_auc"], orientation="h", name="validation", marker_color=LEGIT)
        fig.add_bar(y=e["candidate"], x=e["test_pr_auc"], orientation="h", name="test", marker_color=FRAUD)
        fig.update_layout(barmode="group", xaxis=dict(range=[0, 1], title="PR-AUC"), bargap=0.3, bargroupgap=0.08)
        show(fig, 360)
    with col_b, card("Portes de qualité", "PR-AUC ≥ 0,75 · rappel ≥ 70 % · précision ≥ 50 % · équité 4/5 ≥ 0,80"):
        gates_tbl = exp_[["candidate", "test_pr_auc", "test_recall", "test_precision", "fairness_recall_ratio", "gates_passed"]].copy()
        gates_tbl["portes"] = np.where(gates_tbl["gates_passed"], "✅ conforme", "❌ refusé")
        st.dataframe(gates_tbl.drop(columns="gates_passed"), hide_index=True, width="stretch", height=250,
                     column_config={"candidate": "candidat",
                                    "test_pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
                                    "test_recall": st.column_config.NumberColumn("rappel", format="%.3f"),
                                    "test_precision": st.column_config.NumberColumn("précision", format="%.3f"),
                                    "fairness_recall_ratio": st.column_config.NumberColumn("équité 4/5", format="%.2f")})
        st.dataframe(reg_[["version", "aliases", "candidate", "validation", "test_pr_auc", "created_at"]], hide_index=True, width="stretch",
                     column_config={"aliases": "alias", "candidate": "candidat", "created_at": "créé le",
                                    "test_pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f")})

    if drift_ is not None and perf_ is not None:
        with card("Dérive des données (PSI)", "Par variable et par mois · < 0,10 stable · 0,10 – 0,25 à surveiller · ≥ 0,25 dérive"):
            heat_ = drift_.pivot(index="feature", columns="period", values="psi")
            heat_ = heat_.reindex(list(dict.fromkeys(drift_["feature"])))
            fig = px.imshow(heat_, color_continuous_scale=[[0, "#f3f7fc"], [0.2, "#fde3cf"], [0.5, "#eb6834"], [1, "#7a2a0b"]],
                            zmin=0, zmax=0.5, aspect="auto", text_auto=".2f", labels=dict(color="PSI"))
            fig.update_traces(xgap=3, ygap=3, hovertemplate="%{y} · %{x}<br>PSI = <b>%{z:.3f}</b><extra></extra>", textfont=dict(size=11))
            fig.update_xaxes(showgrid=False, title=None, ticks="")
            fig.update_yaxes(title=None, ticks="")
            show(fig, 480, legend=False)

        col_c, col_d = st.columns([1.2, 1])
        with col_c, card("Performance mensuelle du champion", "Pointillé : porte de rappel à 70 %"):
            p_ = perf_[perf_["period"] != "stress +35 %"]
            fig = go.Figure()
            fig.add_scatter(x=p_["period"], y=p_["recall"], name="rappel", mode="lines+markers", line=dict(color=FRAUD, width=2.5),
                            marker=dict(size=8, line=dict(color="white", width=1.5)))
            fig.add_scatter(x=p_["period"], y=p_["precision"], name="précision", mode="lines+markers", line=dict(color=LEGIT, width=2.5),
                            marker=dict(size=8, line=dict(color="white", width=1.5)))
            fig.add_hline(y=0.70, line_dash="dot", line_color=MUTED, annotation_text="porte rappel 70 %",
                          annotation_font=dict(size=11, color=INK_2))
            fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1.05]), hovermode="x unified")
            show(fig, 340)
        with col_d, card("Décisions de ré-entraînement", "Règle : dérive PSI hors calendrier ou rappel sous la porte"):
            st.dataframe(perf_[["period", "max_feature_psi", "recall", "precision", "retrain", "retrain_reason"]],
                         hide_index=True, width="stretch", height=300,
                         column_config={"period": "période",
                                        "max_feature_psi": st.column_config.NumberColumn("PSI max", format="%.2f"),
                                        "recall": st.column_config.NumberColumn("rappel", format="%.2f"),
                                        "precision": st.column_config.NumberColumn("précision", format="%.2f"),
                                        "retrain": st.column_config.CheckboxColumn("ré-entraîner"),
                                        "retrain_reason": st.column_config.TextColumn("motif", width="large")})
        explain("mlops_drift", "PSI = Σ (p_prod − p_réf) · ln(p_prod / p_réf) sur les déciles de la période d'entraînement. "
                "Les variables calendaires (`days_to_holiday`, `is_weekend`) varient mécaniquement et ne déclenchent pas de "
                "ré-entraînement ; décembre montre une vraie dérive saisonnière des features de vélocité.")

    if mlx["fairness"] is not None:
        with card("Équité du champion", "Rappel par groupe — règle des 4/5 : le minimum doit atteindre 80 % du maximum"):
            fr = mlx["fairness"].assign(label=lambda d: d["dimension"] + " · " + d["groupe"]).sort_values(["dimension", "rappel"])
            fig = go.Figure(go.Bar(x=fr["rappel"], y=fr["label"], orientation="h",
                                   marker_color=[LEGIT if d == "genre" else FRAUD for d in fr["dimension"]],
                                   text=[f"{v:.0%}" for v in fr["rappel"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y} : rappel %{x:.1%}<extra></extra>"))
            fig.update_layout(xaxis=dict(range=[0, 1.1], tickformat=".0%"), bargap=0.35)
            show(fig, 300, legend=False)
    if mlx["model_card"] is not None:
        with st.expander("Fiche modèle (model card) du dernier candidat", icon=":material/description:"):
            st.markdown(mlx["model_card"]["markdown"].iloc[0])


# ──────────────────────────────────────────────────────────────────────────────
# 10. Données & gouvernance
# ──────────────────────────────────────────────────────────────────────────────
def page_data() -> None:
    page_header("Gouvernance", "Architecture Médaillon & catalogue",
                "Du fichier CSV brut à l'indicateur métier : chaque couche a un rôle, un format et des garanties.")
    layers = [
        ("#b26b3a", "Bronze", "/lakehouse/bronze/", "Données brutes ingérées telles quelles + lignage : <b>transactions</b> (1,85 M lignes) "
         "et <b>us_holidays</b> (342 lignes). Delta Lake."),
        ("#8a99ad", "Silver", "/lakehouse/silver/transactions_enriched", "Données nettoyées, <b>pseudonymisées</b> (SHA-256 salé), "
         "20 features, jointure calendrier. Delta partitionné par mois, Z-Order <code>card_id</code>."),
        ("#d9a400", "Gold", "/lakehouse/gold/", f"<b>{len(CATALOG)} tables</b> agrégées et compactées, seules lues par ce tableau de bord."),
        ("#2a78d6", "Modèles & MLOps", "/lakehouse/models/ · MLflow :5050 · API :8010", "Champion déployé + métadonnées, "
         "registre MLflow avec alias, API FastAPI <code>/predict</code> et <code>/reload</code>."),
    ]
    cols = st.columns(4)
    for col, (color, name, path, text) in zip(cols, layers, strict=True):
        col.markdown(f'<div class="layer" style="border-top-color:{color}"><h4>{name}</h4><code>{path}</code><p>{text}</p></div>',
                     unsafe_allow_html=True)
    st.write("")
    if catalog is not None:
        with card("Catalogue des tables Gold", "Description, notebook producteur et date de mise à jour"):
            st.dataframe(catalog[["table_name", "description", "producer", "updated_at"]].sort_values("table_name"),
                         hide_index=True, width="stretch", height=460,
                         column_config={"table_name": "table", "producer": "producteur", "updated_at": "mise à jour"})
    insights([("lock", "Aucune donnée personnelle directe (nom, adresse, numéro de carte) n'est présente au-delà de la couche Bronze.", "blue"),
              ("history", "Toutes les tables sont versionnées par Delta Lake : <b>Time Travel</b> et <b>RESTORE</b> possibles.", "green")])


{
    "overview": page_overview, "search": page_search, "categories": page_categories, "time": page_time, "geo": page_geo, "segments": page_segments,
    "model": page_model, "live": page_live, "tests": page_tests, "alerts": page_alerts, "mlops": page_mlops, "data": page_data,
}[page]()

st.markdown('<div class="foot"><span>FraudLake · Lakehouse Médaillon HDFS · Spark · Delta Lake · MLflow</span>'
            '<span>Yan-Mol LANGWANA &amp; Rabab KOULI — Architecture des systèmes Big Data · Prof. Antoine GREA</span></div>',
            unsafe_allow_html=True)
