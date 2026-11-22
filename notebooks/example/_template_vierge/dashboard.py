import streamlit as st
import plotly.express as px  # noqa: F401
from pyspark.sql import SparkSession

st.set_page_config(page_title="Lakehouse Analytics", layout="wide")
st.title("Explorateur Data Lakehouse (HDFS + Delta Lake)")


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("StreamlitGoldReader")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@st.cache_data(ttl=60)
def load_gold_table(hdfs_path: str):
    spark = get_spark()
    return spark.read.format("delta").load(hdfs_path).toPandas()


st.info(
    "Cluster HDFS connecté (`hdfs://namenode:9000`). "
    "Alimentez vos tables Delta dans `/lakehouse/gold/` depuis JupyterLab pour afficher vos analyses ici."
)

with st.expander("Exemple d'intégration d'une requête expliquée (attendu pour le projet)"):
    st.code(
        """
SELECT category,
       COUNT(*) AS total_orders,
       PERCENTILE_APPROX(delivery_days, 0.5) AS median_delivery_days
FROM delta.`/lakehouse/silver/orders`
GROUP BY category
ORDER BY total_orders DESC;
        """,
        language="sql",
    )
    st.markdown(
        "**Justification statistique :** Utilisation de la médiane plutôt que de la moyenne "
        "pour neutraliser les retards logistiques extrêmes."
    )

# ==============================================================================
# TODO (Projet Jours 3 & 4) — À ADAPTER À VOTRE SUJET KAGGLE
# ==============================================================================
# 1. Chargez vos propres tables Gold avec `load_gold_table("/lakehouse/gold/<votre_table>")`.
# 2. Ajoutez des filtres interactifs (`st.selectbox`, `st.multiselect`, `st.slider`) et des
#    graphiques Plotly Express (`px.bar`, `px.scatter`, `px.line`, `px.box`, `px.pie`).
# 3. Documentez sous chaque analyse la requête Spark SQL utilisée ainsi que la justification
#    statistique de vos indicateurs (médiane vs moyenne, IQR, Pareto, test d'hypothèse, clustering).
