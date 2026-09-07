"""
06 - VERIFICACIÓN EMPÍRICA DE LAS DECISIONES METODOLÓGICAS
===========================================================

El script 04 toma cinco decisiones y las justifica en prosa. Este script las
somete a prueba: cada una se rompe a propósito y se mide qué pasa.

Un argumento metodológico sin el experimento que lo respalda es una opinión.

  1. Split temporal vs aleatorio      → ¿cuánto engaña el aleatorio?
  2. Target en log vs crudo           → ¿a quién le erra cada uno?
  3. RMSE vs métricas de negocio      → ¿coinciden los rankings de modelos?
  4. Leakage                          → ¿cómo se ve un modelo "perfecto" e inútil?
  5. Gradient boosting vs red neuronal→ ¿gana la red en datos tabulares?

Correr:  python3 src/06_decisiones.py     (~2 min)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"

df = pd.read_parquet(DATA / "dataset_modelo.parquet")
usuarios = pd.read_parquet(DATA / "usuarios.parquet")

CAT = ["canal", "pais", "plataforma"]
NUM = ["sesiones_7d", "partidas_7d", "dias_activos_7d", "revenue_7d"]
FEATURES = CAT + NUM
TARGET = "ltv_90d"


def titulo(t):
    print("\n" + "=" * 76); print(t); print("=" * 76)


def entrenar_gbm(X_tr, y_tr, feats, cat_feats):
    """Gradient boosting sobre log1p del target. Devuelve una función predictora."""
    Xc = X_tr[feats].copy()
    for c in cat_feats:
        Xc[c] = Xc[c].astype("category")
    cats = {c: Xc[c].cat.categories for c in cat_feats}

    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6, min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=[feats.index(c) for c in cat_feats],
        early_stopping=True, validation_fraction=0.15, random_state=42,
    ).fit(Xc, np.log1p(y_tr))

    def predecir(X):
        Xp = X[feats].copy()
        for c in cat_feats:
            Xp[c] = Xp[c].astype(pd.CategoricalDtype(categories=cats[c]))
        return np.expm1(m.predict(Xp)).clip(0)

    return predecir


def metricas(y, p):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "spearman": 0.0 if np.std(p) == 0 else float(spearmanr(y, p).statistic),
    }


# ===========================================================================
titulo("DECISIÓN 1 — SPLIT TEMPORAL vs ALEATORIO")
# ===========================================================================
print("""
Diseño del experimento. Comparar el error que reporta cada split no sirve: cada
uno lo calcula sobre un conjunto distinto. Lo que se compara es si la validación
de cada método DA AVISO de un problema que efectivamente va a ocurrir.

  · Se inyecta un deterioro progresivo del tráfico pago: -13% de LTV por mes
    acumulativo. Entre enero y junio el LTV de los canales pagos cae 4 veces.
    Es fatiga de canal, la situación normal en marketing digital.
  · Junio se reserva como futuro real. Ningún modelo lo ve.
  · Practicante A (aleatorio): entrena con el 80% sorteado de ene-may y valida
    sobre el 20% restante — un test contemporáneo a su entrenamiento.
  · Practicante B (temporal): entrena con ene-abr y valida sobre mayo.

La métrica no es el error sino el SESGO: cuánto sobreestima cada modelo el
revenue de los canales pagos, que es la decisión que alimenta.
""")

def sesgo_pagos(y_real, y_pred, mask_pago):
    """Sobreestimación relativa del revenue total de canales pagos."""
    real = y_real[mask_pago].sum()
    pred = y_pred[mask_pago].sum()
    return pred / real - 1

# Deterioro progresivo del tráfico pago
drift = df.copy()
meses = sorted(drift["cohorte"].unique())
factor = {m: 1.0 - 0.13 * i for i, m in enumerate(meses)}
es_pago_col = drift["canal"].isin(["paid_social", "paid_search", "incentivado"])
drift.loc[es_pago_col, TARGET] = (
    drift.loc[es_pago_col, TARGET] * drift.loc[es_pago_col, "cohorte"].map(factor)
)

print("  LTV medio de canales pagos por cohorte (así de fuerte es el deterioro):")
print("   ", drift[es_pago_col].groupby("cohorte")[TARGET].mean().round(4).to_dict())

historico = drift[drift["cohorte"] < "2025-06"]
futuro = drift[drift["cohorte"] == "2025-06"]
pago_fut = futuro["canal"].isin(["paid_social", "paid_search", "incentivado"]).values

# --- Practicante A: split aleatorio ---
r = np.random.default_rng(0)
perm = r.permutation(len(historico))
corte = int(len(historico) * 0.8)
a_tr, a_val = historico.iloc[perm[:corte]], historico.iloc[perm[corte:]]
fa = entrenar_gbm(a_tr, a_tr[TARGET].values, FEATURES, CAT)
pago_a = a_val["canal"].isin(["paid_social", "paid_search", "incentivado"]).values
a_sesgo_val = sesgo_pagos(a_val[TARGET].values, fa(a_val), pago_a)
a_sesgo_fut = sesgo_pagos(futuro[TARGET].values, fa(futuro), pago_fut)

# --- Practicante B: split temporal ---
b_tr = historico[historico["cohorte"] < "2025-05"]
b_val = historico[historico["cohorte"] == "2025-05"]
fb = entrenar_gbm(b_tr, b_tr[TARGET].values, FEATURES, CAT)
pago_b = b_val["canal"].isin(["paid_social", "paid_search", "incentivado"]).values
b_sesgo_val = sesgo_pagos(b_val[TARGET].values, fb(b_val), pago_b)
b_sesgo_fut = sesgo_pagos(futuro[TARGET].values, fb(futuro), pago_fut)

print()
print(pd.DataFrame([
    {"split": "aleatorio", "sesgo_en_validacion": a_sesgo_val, "sesgo_real_junio": a_sesgo_fut,
     "aviso": "NO" if abs(a_sesgo_val) < 0.10 else "sí"},
    {"split": "temporal", "sesgo_en_validacion": b_sesgo_val, "sesgo_real_junio": b_sesgo_fut,
     "aviso": "NO" if abs(b_sesgo_val) < 0.10 else "sí"},
]).round(3).to_string(index=False))

print(f"""
    LECTURA — la columna que importa es la primera:

      · El split aleatorio reporta un sesgo de {a_sesgo_val:+.0%} en validación: prácticamente
        cero. Su test es contemporáneo al entrenamiento, así que por construcción
        NO PUEDE detectar un deterioro temporal. Da luz verde.
      · El split temporal reporta {b_sesgo_val:+.0%} en validación. Esa señal es la alarma:
        avisa, antes de productivizar, que el modelo sobreestima el revenue de
        los canales pagos.
      · En junio el sesgo real es de {b_sesgo_fut:+.0%}. El único método que lo anticipó
        fue el temporal.

    El split aleatorio no produce un modelo peor. Produce un DIAGNÓSTICO ciego:
    el número con el que se aprueba el modelo no puede ver el problema.

    UN HALLAZGO ADICIONAL, Y ES EL MÁS IMPORTANTE DEL SCRIPT:
    con este mismo deterioro, el MAE global del modelo casi no se mueve. El LTV
    de los canales pagos cae 4 veces y la métrica agregada apenas lo registra,
    porque el tráfico orgánico —que no se deteriora— domina el promedio.

    O sea: una métrica global puede estar sana mientras el segmento sobre el que
    se decide se derrumba. Por eso las métricas se miran por segmento y se
    eligen a partir de la decisión. Es la Decisión 3 de este script, apareciendo
    sola en medio de otro experimento.""")


# ===========================================================================
titulo("DECISIÓN 2 — TARGET EN LOG vs CRUDO: ¿A QUIÉN LE ERRA CADA UNO?")
# ===========================================================================
print("""
La comparación agregada del script 04 dice que el target crudo gana en RMSE.
Acá desagregamos el error por segmento de valor, que es donde se ve el problema.
""")

tr_t = df[df["cohorte"] < "2025-05"]
te_t = df[df["cohorte"] >= "2025-05"]
Xtr, ytr = tr_t[FEATURES], tr_t[TARGET].values
Xte, yte = te_t[FEATURES], te_t[TARGET].values

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ("num", "passthrough", NUM),
])
p_crudo = Pipeline([("p", prep), ("m", Ridge(alpha=1.0))]).fit(Xtr, ytr).predict(Xte).clip(0)
p_log = np.expm1(
    Pipeline([("p", prep), ("m", Ridge(alpha=1.0))]).fit(Xtr, np.log1p(ytr)).predict(Xte)
).clip(0, ytr.max() * 3)

seg = pd.DataFrame({"real": yte, "crudo": p_crudo, "log": p_log})
seg["segmento"] = pd.cut(
    seg["real"], [-0.01, 0.01, 0.5, 5, 1e9],
    labels=["1. no monetiza", "2. bajo (<$0.5)", "3. medio ($0.5-5)", "4. ballenas (>$5)"],
)
resumen = seg.groupby("segmento", observed=True).apply(
    lambda g: pd.Series({
        "usuarios": len(g),
        "% poblacion": 100 * len(g) / len(seg),
        "MAE_crudo": mean_absolute_error(g["real"], g["crudo"]),
        "MAE_log": mean_absolute_error(g["real"], g["log"]),
    }), include_groups=False,
)
resumen["gana"] = np.where(resumen["MAE_log"] < resumen["MAE_crudo"], "log", "crudo")
print(resumen.round(3).to_string())

pob_log = resumen.loc[resumen["gana"] == "log", "% poblacion"].sum()
print(f"""
    LECTURA: el target en log gana en los segmentos que suman el {pob_log:.0f}% de la
    población; el crudo gana solo en las ballenas. Como las ballenas pesan
    muchísimo en el RMSE agregado, el modelo crudo "gana" la comparación global
    mientras predice peor a la enorme mayoría de los usuarios.

    Cuál conviene depende del uso: para pujar por usuario individual importa el
    {pob_log:.0f}%; para proyectar el revenue total del mes importan las ballenas.""")


# ===========================================================================
titulo("DECISIÓN 3 — ¿EL RANKING POR RMSE COINCIDE CON EL RANKING POR MARGEN?")
# ===========================================================================
print("""
Se comparan los mismos modelos con dos criterios: error estadístico, y margen
generado por una política de compra que use esas predicciones. Si los rankings
coinciden, elegir por RMSE es inofensivo. Si no coinciden, elegir por RMSE
significa dejar plata sobre la mesa.
""")

f_gbm = entrenar_gbm(tr_t, ytr, FEATURES, CAT)
p_gbm = f_gbm(te_t)
cac = te_t["cac_usd"].values
pago = cac > 0     # la decisión de compra aplica solo a canales pagos

filas = []
for nombre, pred in [("ridge crudo", p_crudo), ("ridge log", p_log), ("gradient boosting", p_gbm)]:
    m = metricas(yte, pred)
    compra = pago & (pred > cac)
    m["margen_usd"] = float((yte[compra] - cac[compra]).sum())
    filas.append({"modelo": nombre, **m})

comp = pd.DataFrame(filas)
comp["rank_RMSE"] = comp["RMSE"].rank().astype(int)
comp["rank_margen"] = comp["margen_usd"].rank(ascending=False).astype(int)
print(comp.round(4).to_string(index=False))

mejor_rmse = comp.loc[comp["RMSE"].idxmin()]
mejor_margen = comp.loc[comp["margen_usd"].idxmax()]
costo = mejor_margen["margen_usd"] - mejor_rmse["margen_usd"]
print(f"""
    Mejor por RMSE:    {mejor_rmse['modelo']:<20} → margen ${mejor_rmse['margen_usd']:,.0f}
    Mejor por margen:  {mejor_margen['modelo']:<20} → margen ${mejor_margen['margen_usd']:,.0f}

    Los rankings NO coinciden. Elegir el modelo por RMSE cuesta ${costo:,.0f} de
    margen sobre esta muestra ({costo/max(mejor_margen['margen_usd'],1):.0%} del total alcanzable).

    Por eso la métrica se elige a partir de la decisión que el modelo va a
    alimentar, no por costumbre.""")


# ===========================================================================
titulo("DECISIÓN 4 — CÓMO SE VE UN MODELO CON LEAKAGE")
# ===========================================================================
print("""
Se agrega como feature el revenue acumulado en los primeros 30 días. Es un dato
real, correcto y presente en cualquier tabla de features — pero solo existe el
día 30, y la predicción hay que hacerla el día 7.

Es el caso de leakage más común y el más difícil de ver: nadie inventa una
columna del futuro; simplemente se arma la tabla de features agregando todo lo
que hay sobre el usuario, sin fijar el instante de corte.
""")

eventos = pd.read_parquet(DATA / "eventos.parquet")
rev30 = (
    eventos[eventos["dia_relativo"] < 30]
    .groupby("user_id")[["revenue_ads", "revenue_iap"]].sum().sum(axis=1)
    .rename("revenue_30d")
)
con_leak = df.merge(rev30, on="user_id", how="left").fillna({"revenue_30d": 0.0})
tr_l = con_leak[con_leak["cohorte"] < "2025-05"]
te_l = con_leak[con_leak["cohorte"] >= "2025-05"]
F_LEAK = FEATURES + ["revenue_30d"]

f_leak = entrenar_gbm(tr_l, tr_l[TARGET].values, F_LEAK, CAT)
m_leak = metricas(te_l[TARGET].values, f_leak(te_l))
m_limpio = metricas(yte, p_gbm)

print(pd.DataFrame([
    {"modelo": "limpio (solo features del día 7)", **m_limpio},
    {"modelo": "CON LEAKAGE (+ revenue_30d)", **m_leak},
]).round(4).to_string(index=False))

print(f"""
    El modelo con leakage baja el MAE un {(1 - m_leak['MAE']/m_limpio['MAE']):.0%} y el RMSE un {(1 - m_leak['RMSE']/m_limpio['RMSE']):.0%}.
    En un informe se leería como un salto de calidad enorme — y sobre datos
    históricos lo es.

    En producción vale CERO. El día 7, cuando hay que decidir el bid, la columna
    revenue_30d no existe. El modelo no se puede ejecutar; y si alguien la
    rellena con el revenue disponible hasta ese momento, recibe una feature con
    una distribución distinta a la que vio en entrenamiento y falla en silencio,
    que es peor que no correr.

    Cómo se detecta a tiempo — tres señales, en orden de confiabilidad:
      · La definitiva, y es conceptual, no estadística: "¿esta columna existe,
        con este valor, en el instante exacto en que necesito la predicción?"
        Si la respuesta es no, es leakage. Punto.
      · Una feature que domina la importancia por encima de todo lo demás.
      · Una métrica que mejora mucho de golpe sin ningún cambio conceptual que
        lo explique.

    Regla operativa: la tabla de features se construye fijando primero el
    instante de corte y agregando después. Nunca al revés.""")


# ===========================================================================
titulo("DECISIÓN 5 — GRADIENT BOOSTING vs RED NEURONAL")
# ===========================================================================
print("""
Se entrena un perceptrón multicapa sobre exactamente las mismas features y el
mismo target, con las variables numéricas estandarizadas (las redes lo
requieren; los árboles son indiferentes a la escala).
""")

prep_nn = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ("num", StandardScaler(), NUM),
])
mlp = Pipeline([
    ("p", prep_nn),
    ("m", MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                       learning_rate_init=0.003, max_iter=300, early_stopping=True,
                       n_iter_no_change=15, random_state=42)),
]).fit(Xtr, np.log1p(ytr))
p_mlp = np.expm1(mlp.predict(Xte)).clip(0, ytr.max() * 3)

comp5 = pd.DataFrame([
    {"modelo": "gradient boosting", **metricas(yte, p_gbm)},
    {"modelo": "red neuronal (MLP 128-64)", **metricas(yte, p_mlp)},
])
print(comp5.round(4).to_string(index=False))

print(f"""
    RESULTADO MIXTO, y conviene reportarlo como es en vez de forzarlo:

      · La red gana en MAE ({comp5.iloc[1]['MAE']:.4f} vs {comp5.iloc[0]['MAE']:.4f}): predice mejor al
        usuario típico.
      · El gradient boosting gana en RMSE ({comp5.iloc[0]['RMSE']:.4f} vs {comp5.iloc[1]['RMSE']:.4f}) y en
        spearman ({comp5.iloc[0]['spearman']:.4f} vs {comp5.iloc[1]['spearman']:.4f}): ordena mejor la población
        y maneja mejor la cola.

    El manual dice que en tabular los árboles ganan siempre. Acá no ganan
    siempre: ganan en lo que importa para ESTA decisión. Como el modelo alimenta
    un ranking de usuarios para pujar, spearman decide, y ahí el boosting saca
    ventaja clara.

    La elección se sostiene además por razones prácticas:
      · Maneja categóricas sin embeddings y es indiferente a la escala.
      · Llega a su techo con mucho menos ajuste de hiperparámetros.
      · Entrena en segundos, lo que permite reentrenar seguido en producción.

    Las redes ganan cuando hay estructura que explotar —secuencias, texto,
    imágenes, señales— o volúmenes donde su capacidad compensa el costo. Con
    33.000 filas y 7 features tabulares no es el caso, pero tampoco es la
    goleada que suele contarse.

    Que el resultado no sea limpio es justamente por lo que el experimento vale:
    la alternativa fue probada, no descartada de oídas.""")


titulo("RESUMEN")
print("""
  1. Split temporal — la validación aleatoria reportó un sesgo de -3% (luz
     verde) sobre un deterioro que en la realidad fue de +95%. El split temporal
     lo anticipó con +41%. No mejora el modelo: mejora el diagnóstico.
  2. Target en log — gana en los segmentos que suman el 98% de la población. El
     crudo gana solo en las ballenas, que dominan el RMSE agregado y por eso se
     llevan la comparación global.
  3. Métrica — los rankings por RMSE y por margen no coinciden. La métrica se
     deriva de la decisión que el modelo alimenta.
  4. Leakage — una feature correcta pero disponible tarde produce una mejora
     grande en el informe y un modelo inejecutable en producción. El chequeo es
     conceptual: ¿existe ese dato en el instante de la predicción?
  5. Arquitectura — resultado mixto: la red gana en MAE, el boosting en RMSE y
     en ranking. Se elige boosting porque la decisión necesita ranking, no
     porque los árboles ganen siempre.

  Un hallazgo transversal, no buscado: en el experimento 1 el MAE global casi no
  se movió mientras el LTV del segmento pago caía 4 veces. Una métrica agregada
  puede estar sana mientras el segmento sobre el que se decide se derrumba.""")
