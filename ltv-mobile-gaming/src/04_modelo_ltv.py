"""
04 - MODELO PREDICTIVO DE LTV
==============================

El problema, dicho como lo diría negocio:
  "Con lo que veo de un usuario en sus primeros 7 días, ¿cuánto va a valer a
   los 90? Necesito saberlo el día 7, no el día 90, para decidir cuánto pujo
   por usuarios parecidos."

DECISIONES METODOLÓGICAS QUE DOCUMENTA ESTE SCRIPT:
  1. Split temporal en vez de aleatorio, y por qué el aleatorio sobreestima
  2. Baselines obligatorios antes de cualquier modelo
  3. Tratamiento de un target con cola larga (transformación log y su costo)
  4. Métricas técnicas junto a métricas de negocio
  5. Criterio de leakage: qué información está disponible al predecir

Correr:  python3 src/04_modelo_ltv.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

pd.set_option("display.width", 200)
RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_parquet(RAIZ / "data" / "dataset_modelo.parquet")


def titulo(t):
    print("\n" + "=" * 76); print(t); print("=" * 76)


# ===========================================================================
# 1. SPLIT TEMPORAL (no aleatorio)
# ===========================================================================
titulo("1. PARTIR LOS DATOS: POR QUÉ NO ES train_test_split(shuffle=True)")

print("""
Todo tutorial de sklearn usa train_test_split aleatorio. Acá está MAL, porque:

  · El modelo se va a usar sobre usuarios FUTUROS, no sobre usuarios sorteados
    del mismo período. Evaluarlo con un split aleatorio te da un número
    optimista que no se reproduce en producción.
  · Si el mix de canales o el producto cambian con el tiempo, el split aleatorio
    lo esconde. El split temporal te lo muestra.

Regla: si tu problema tiene una flecha del tiempo, el split respeta el tiempo.
Entrenamos con las cohortes viejas, evaluamos con las nuevas.

(En el mundo real hay un problema extra: las cohortes recientes TODAVÍA no
 cumplieron 90 días, así que su LTV está censurado a la derecha. Se resuelve
 con modelos de supervivencia o entrenando solo con cohortes maduras. Es la
 misma censura que la de una cartera de créditos que aún no venció.)
""")

corte = "2025-05"
train = df[df["cohorte"] < corte].copy()
test = df[df["cohorte"] >= corte].copy()
print(f"  Train: cohortes < {corte}  →  {len(train):,} usuarios")
print(f"  Test:  cohortes >= {corte} →  {len(test):,} usuarios")

FEATURES_CAT = ["canal", "pais", "plataforma"]
FEATURES_NUM = ["sesiones_7d", "partidas_7d", "dias_activos_7d", "revenue_7d"]
FEATURES = FEATURES_CAT + FEATURES_NUM
TARGET = "ltv_90d"

X_train, y_train = train[FEATURES], train[TARGET].values
X_test, y_test = test[FEATURES], test[TARGET].values

print("""
NOTA SOBRE revenue_7d — criterio de leakage:
  revenue_7d es parte del ltv_90d (los primeros 7 días están adentro del total).
  ¿Es leakage? No: al día 7 ese revenue ya fue efectivamente observado, está
  disponible al momento de predecir. Es señal legítima, y la más fuerte.
  SÍ sería leakage usar 'dias_vida' o el revenue del día 30: información del
  futuro respecto del momento de decisión.
  El chequeo mental es siempre el mismo: "¿esta columna existe, con este valor,
  en el instante en que necesito la predicción?"
""")


# ===========================================================================
# 2. EVALUACIÓN: definimos las métricas ANTES de modelar
# ===========================================================================
def evaluar(nombre, y_true, y_pred, resultados):
    """Métricas técnicas + métricas de negocio.

    RMSE  : penaliza fuerte los errores grandes → lo dominan las ballenas.
    MAE   : error típico en dólares, robusto a outliers.
    Spearman: ¿ordena bien a los usuarios? No mide el nivel, mide el RANKING.
              Para decidir bids, ordenar bien importa más que acertar el monto.
    Lift decil 1: cuántas veces más vale el 10% top que predijo el modelo,
              comparado con el promedio. ES LA MÉTRICA QUE ENTIENDE NEGOCIO.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # Si el modelo predice siempre lo mismo no hay ranking que medir: spearman
    # queda indefinido. Devolvemos 0 (no ordena nada) en vez de un NaN feo.
    rho = 0.0 if np.std(y_pred) == 0 else float(spearmanr(y_true, y_pred).statistic)

    d = pd.DataFrame({"real": y_true, "pred": y_pred})
    d["decil"] = pd.qcut(d["pred"].rank(method="first"), 10, labels=False)
    lift = d[d["decil"] == 9]["real"].mean() / d["real"].mean()
    # % del revenue real capturado por el top 20% que eligió el modelo
    top20 = d.nlargest(int(len(d) * 0.2), "pred")["real"].sum() / d["real"].sum()

    resultados.append({
        "modelo": nombre, "RMSE": rmse, "MAE": mae,
        "spearman": rho, "lift_decil10": lift, "captura_top20_pct": top20 * 100,
    })
    return resultados


resultados = []


# ===========================================================================
# 3. BASELINES — la regla que nunca hay que saltear
# ===========================================================================
titulo("3. BASELINES (referencia obligatoria antes de cualquier modelo)")

# Baseline 0: predecir siempre la media global.
base0 = DummyRegressor(strategy="mean").fit(X_train[FEATURES_NUM], y_train)
resultados = evaluar("baseline: media global", y_test, base0.predict(X_test[FEATURES_NUM]), resultados)

# Baseline 1: la media por canal. Es lo que hace hoy el equipo de marketing
# sin ningún modelo: mirar el promedio histórico del canal.
media_canal = train.groupby("canal")[TARGET].mean()
pred_canal = test["canal"].map(media_canal).fillna(y_train.mean()).values
resultados = evaluar("baseline: media por canal", y_test, pred_canal, resultados)

print(pd.DataFrame(resultados).round(4).to_string(index=False))
print("""
    Ojo: el baseline de media global tiene spearman 0 y lift ~1 porque no
    ordena nada (predice lo mismo para todos). El de media por canal ya
    ordena algo: lift ~1.4x. O sea, elegir por canal es
    MEJOR QUE NADA, pero muy lejos de lo que se puede hacer mirando al usuario.""")


# ===========================================================================
# 4. MODELO LINEAL — y el problema de la cola larga
# ===========================================================================
titulo("4. REGRESIÓN LINEAL: TARGET CRUDO vs TARGET EN LOG")

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
    ("num", "passthrough", FEATURES_NUM),
])

# (a) Regresión sobre el LTV crudo.
lin_crudo = Pipeline([("prep", prep), ("modelo", Ridge(alpha=1.0))]).fit(X_train, y_train)
resultados = evaluar("ridge (target crudo)", y_test, lin_crudo.predict(X_test), resultados)

# (b) Regresión sobre log(1 + LTV), devolviendo a escala de dólares con expm1.
#     log1p/expm1 son inversas exactas y manejan bien el cero.
lin_log = Pipeline([("prep", prep), ("modelo", Ridge(alpha=1.0))]).fit(X_train, np.log1p(y_train))

# TRAMPA CLÁSICA — leé esto, cuesta una tarde de debugging:
# Una regresión lineal EXTRAPOLA sin límite. Si un usuario de test tiene
# partidas_7d muy por encima de lo visto en train, la predicción en log se va
# a 15 o 20... y expm1(20) son 485 millones de dólares. Un solo usuario así
# destruye el RMSE de todo el modelo.
# En producción SIEMPRE se acota la salida a un rango defendible. Acá: 3x el
# máximo observado en entrenamiento.
TOPE = y_train.max() * 3
pred_log = np.expm1(lin_log.predict(X_test)).clip(0, TOPE)
resultados = evaluar("ridge (target log, acotado)", y_test, pred_log, resultados)

sin_tope = np.expm1(lin_log.predict(X_test)).clip(0)
print(f"\n    Sin acotar, la predicción máxima habría sido ${sin_tope.max():,.0f}")
print(f"    (el LTV real más alto del dataset es ${y_train.max():,.2f})")
print(f"    Tope aplicado: ${TOPE:,.2f}\n")

print(pd.DataFrame(resultados).round(4).to_string(index=False))
print("""
    Comparación clave: el target en log tiene peor RMSE pero MEJOR spearman.
    No es contradictorio, es el trade-off central de este problema:

      · target crudo → el modelo persigue a las ballenas para bajar el RMSE, y
        termina prediciendo mal al 95% de usuarios normales.
      · target log   → el modelo aprende la ESTRUCTURA (quién vale más que
        quién) pero subestima el nivel absoluto (la transformación inversa de
        una media en log da una mediana, no una media: sesgo de Jensen).

    Cuál elegís depende de PARA QUÉ es el modelo:
      · ¿Ordenar usuarios para pujar en la subasta?  → optimizá spearman/lift.
      · ¿Proyectar el revenue total del mes que viene? → necesitás el nivel:
        target crudo, o corregir el sesgo del log (Duan smearing).
    Esa decisión —qué métrica optimizar— pesa más que la elección del
    algoritmo.""")


# ===========================================================================
# 5. GRADIENT BOOSTING — el caballo de batalla de la industria
# ===========================================================================
titulo("5. GRADIENT BOOSTING (lo que se usa de verdad en tabular)")

print("""
Para datos tabulares de este tamaño, los árboles con boosting (XGBoost /
LightGBM / HistGradientBoosting) superan consistentemente a las redes
neuronales. La elección es deliberada, no una limitación.

Cómo funciona, en una frase: entrena un árbol chico, mide en qué se equivocó,
entrena otro árbol sobre ESE error, y suma. Cientos de veces. Cada árbol
corrige al anterior.
""")

X_tr = X_train.copy(); X_te = X_test.copy()
for c in FEATURES_CAT:                      # HistGB soporta categóricas nativas
    X_tr[c] = X_tr[c].astype("category")
    X_te[c] = X_te[c].astype(pd.CategoricalDtype(categories=X_tr[c].cat.categories))

gbm = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.06, max_depth=6,
    min_samples_leaf=40, l2_regularization=1.0,
    categorical_features=[FEATURES.index(c) for c in FEATURES_CAT],
    early_stopping=True, validation_fraction=0.15, random_state=42,
)
gbm.fit(X_tr, np.log1p(y_train))
pred_gbm = np.expm1(gbm.predict(X_te)).clip(0)
resultados = evaluar("gradient boosting (log)", y_test, pred_gbm, resultados)

tabla = pd.DataFrame(resultados).round(4)
print(tabla.to_string(index=False))
print(f"\n    Árboles efectivamente entrenados: {gbm.n_iter_} (early stopping cortó solo)")


# ===========================================================================
# 6. ¿QUÉ MIRA EL MODELO?
# ===========================================================================
titulo("6. IMPORTANCIA DE VARIABLES (permutation importance)")
print("""
Método: desordenás UNA columna al azar y medís cuánto empeora el modelo.
Si empeora mucho, esa columna importaba. Es agnóstico al algoritmo y no se
deja engañar por variables categóricas con muchos niveles (a diferencia de la
importancia por 'ganancia' que reportan los árboles).
""")
imp = permutation_importance(gbm, X_te, np.log1p(y_test), n_repeats=5,
                             random_state=42, scoring="neg_mean_absolute_error")
print(pd.DataFrame({
    "feature": FEATURES,
    "importancia": imp.importances_mean.round(4),
    "desvio": imp.importances_std.round(4),
}).sort_values("importancia", ascending=False).to_string(index=False))


# ===========================================================================
# 7. TABLA DE DECILES — cómo se lo mostrás a negocio
# ===========================================================================
titulo("7. LA TABLA QUE VE NEGOCIO: DECILES DE VALOR PREDICHO")

d = pd.DataFrame({"real": y_test, "pred": pred_gbm, "cac": test["cac_usd"].values,
                  "canal": test["canal"].values})
d["decil"] = 10 - pd.qcut(d["pred"].rank(method="first"), 10, labels=False)

vista = d.groupby("decil").agg(
    usuarios=("real", "count"),
    ltv_predicho=("pred", "mean"),
    ltv_real=("real", "mean"),
    cac_medio=("cac", "mean"),
)
vista["margen_real"] = vista["ltv_real"] - vista["cac_medio"]
vista["pct_revenue"] = (d.groupby("decil")["real"].sum() / d["real"].sum() * 100).round(1)
print(vista.round(3).to_string())

print("""
    Así se lee y así se decide:
      · La columna ltv_predicho vs ltv_real muestra si el modelo está CALIBRADO
        (¿predice montos correctos, o solo ordena bien?).
      · margen_real por decil es la decisión: pujar fuerte por los deciles con
        margen positivo, cortar los negativos. Eso es "bidding por valor
        predicho" y es como monetiza de verdad un modelo de LTV.
      · pct_revenue muestra la concentración: si el decil 1 captura la mayoría
        del revenue, el modelo está encontrando las ballenas temprano.""")

# Guardamos las predicciones para el script 05 (Monte Carlo).
d[["real", "pred", "cac", "canal"]].to_parquet(RAIZ / "data" / "predicciones_test.parquet", index=False)

titulo("RESUMEN FINAL")
print(tabla.to_string(index=False))
print("""
CINCO DECISIONES QUE ESTE SCRIPT DOCUMENTA Y JUSTIFICA:
  1. Split temporal y no aleatorio: el modelo se usa sobre cohortes futuras.
  2. log1p en el target: skewness de 25 rompe cualquier pérdida cuadrática.
     Costo asumido: sesgo de Jensen en el nivel absoluto.
  3. Spearman y lift junto a RMSE: la decisión de negocio necesita ranking,
     no nivel.
  4. revenue_7d es señal legítima; dias_vida sería leakage.
  5. Gradient boosting antes que redes neuronales en datos tabulares.""")
