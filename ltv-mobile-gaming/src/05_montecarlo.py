"""
05 - MONTE CARLO: EL RIESGO DE LA DECISIÓN
===========================================

Un modelo de LTV no termina en una predicción: termina en una decisión de
inversión. Este script cuantifica el riesgo de esa decisión con herramientas
estándar de finanzas cuantitativas.

  Herramienta de finanzas              Nombre en data science
  ------------------------------------  ---------------------------------------
  Simular flujos con supuestos          Simulación de Monte Carlo
  Intervalo de confianza de una          Bootstrap
    proyección con datos históricos
  VaR al 95%                            Percentil 5 de la distribución de P&L
  Stress test / escenario adverso       Análisis de sensibilidad
  ¿Invierto o no en esta operación?     Decisión bajo incertidumbre
  Payback period                        Break-even de LTV vs CAC

LA PREGUNTA QUE RESPONDEMOS:
  paid_social tiene LTV/CAC de 1.08. Está en el filo. Con el promedio, la
  respuesta es "sí, apenas". Pero el promedio de una distribución con cola
  larga es un número muy inestable: depende de si en la muestra cayeron 2
  ballenas o 5. ¿Cuál es la probabilidad REAL de perder plata invirtiendo
  USD 50.000 en ese canal? Eso no se responde con un promedio.

Correr:  python3 src/05_montecarlo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
pd.set_option("display.width", 200)

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"

usuarios = pd.read_parquet(DATA / "usuarios.parquet")
dataset = pd.read_parquet(DATA / "dataset_modelo.parquet")


def titulo(t):
    print("\n" + "=" * 76); print(t); print("=" * 76)


# ===========================================================================
# 1. BOOTSTRAP: ¿cuán confiable es el LTV promedio que reporto?
# ===========================================================================
titulo("1. BOOTSTRAP — el intervalo de confianza del LTV promedio")

print("""
El bootstrap es la idea más útil de la estadística aplicada y se explica en una
línea: se remuestrea la propia muestra con reposición, muchas veces, y se mira
cómo varía el estadístico. Sin fórmulas cerradas ni supuestos de normalidad.

Por qué importa acá: con una distribución así de sesgada, el teorema central
del límite converge LENTO. El intervalo de confianza clásico (media ± 1.96·EE)
subestima la incertidumbre real. El bootstrap no.
""")

N_BOOT = 4000
filas = []

for canal, grupo in dataset.groupby("canal"):
    ltv = grupo["ltv_90d"].values
    n = len(ltv)
    # Matriz de N_BOOT remuestreos de tamaño n. Vectorizado: sin loops.
    idx = rng.integers(0, n, size=(N_BOOT, n))
    medias = ltv[idx].mean(axis=1)

    cac = grupo["cac_usd"].mean()
    filas.append({
        "canal": canal, "n": n, "cac": cac,
        "ltv_medio": ltv.mean(),
        "ic_2.5": np.percentile(medias, 2.5),
        "ic_97.5": np.percentile(medias, 97.5),
        "ancho_ic_rel": (np.percentile(medias, 97.5) - np.percentile(medias, 2.5)) / ltv.mean(),
        "p_ltv_mayor_cac": float((medias > cac).mean()) if cac > 0 else 1.0,
    })

boot = pd.DataFrame(filas).sort_values("ltv_medio", ascending=False)
print(boot.round(4).to_string(index=False))
print("""
    ACÁ ESTÁ EL PUNTO: mirá 'p_ltv_mayor_cac' de paid_search. El LTV/CAC
    puntual da 0.95 (parece que pierde), pero con la incertidumbre de la
    muestra la probabilidad de que sea rentable NO es 0. Y paid_social, que
    puntualmente "gana", tampoco es un sí seguro.

    Reportar "LTV/CAC = 1.08" sin intervalo es el equivalente a presentar un
    VAN positivo sin análisis de sensibilidad: técnicamente cierto, y aun así
    insuficiente para decidir.""")


# ===========================================================================
# 2. MONTE CARLO: distribución del P&L de una inversión de marketing
# ===========================================================================
titulo("2. MONTE CARLO — ¿qué pasa si pongo USD 50.000 en paid_social?")

PRESUPUESTO = 50_000.0
CANAL = "paid_social"

pool = dataset.loc[dataset["canal"] == CANAL, "ltv_90d"].values
cac = float(usuarios.loc[usuarios["canal"] == CANAL, "cac_usd"].mean())
n_usuarios_comprados = int(PRESUPUESTO / cac)

print(f"""
Montaje de la simulación:
  Presupuesto:            ${PRESUPUESTO:,.0f}
  CAC del canal:          ${cac:.2f}
  Usuarios que compro:    {n_usuarios_comprados:,}
  Escenarios simulados:   10.000

Cada escenario simula el P&L = revenue generado - costo, con DOS fuentes de
incertidumbre (ver el comentario del código, es el concepto central del script).
Es el mismo Monte Carlo con el que valuás una cartera; acá el activo es un
usuario y el "riesgo de modelo" es que el canal deje de comportarse como ayer.
""")

N_SIM = 10_000

# --- LOS DOS TIPOS DE INCERTIDUMBRE (concepto clave, en DS y en finanzas) ---
#
# ALEATORIA (aleatoric): el azar irreducible de qué usuarios me tocan.
#   Con 166.000 usuarios, esta se promedia hasta desaparecer. Ley de los
#   grandes números. Si simulás SOLO esto, te da P(perder)=0% y te creés que
#   la inversión es segura. Es la trampa.
#
# DE PARÁMETRO / EPISTÉMICA (epistemic): yo NO conozco la verdadera
#   distribución de LTV del canal. La estimé con una muestra finita de 15.000
#   usuarios del pasado. Esa estimación tiene error, y ese error NO se
#   diluye comprando más usuarios.
#
# En finanzas esto se llama "riesgo de modelo" o "riesgo de supuestos", y es
# el que domina las pérdidas — no la volatilidad del día a día.
# Un Monte Carlo que ignora la incertidumbre de parámetro es exactamente un
# VaR que asume que la matriz de covarianzas es la verdadera. Ya sabemos cómo
# termina eso.
#
# Estructura de la simulación (jerárquica, 2 niveles):
#   nivel 1 → bootstrapeo el pool histórico  = "¿y si la realidad fuera otra?"
#   nivel 2 → sorteo la cohorte de ese pool  = "¿y qué usuarios me tocan?"
#   extra   → factor de deterioro: las cohortes futuras rara vez son iguales a
#             las pasadas (fatiga del canal, competencia, cambios de algoritmo).
#             Lo modelamos como un shock lognormal de ~15% de dispersión.

n_pool = len(pool)
pnl = np.empty(N_SIM)
for s in range(N_SIM):
    pool_escenario = pool[rng.integers(0, n_pool, n_pool)]        # nivel 1
    deterioro = rng.lognormal(mean=-0.012, sigma=0.15)            # shock de canal
    ltv_esperado = pool_escenario.mean() * deterioro
    # A este volumen el promedio de la cohorte ~ el promedio del pool escenario,
    # así que el nivel 2 aporta poco; lo dejamos explícito igual para que se vea.
    ruido_muestral = rng.normal(0, pool_escenario.std() / np.sqrt(n_usuarios_comprados))
    pnl[s] = (ltv_esperado + ruido_muestral) * n_usuarios_comprados - PRESUPUESTO

roi = pnl / PRESUPUESTO

print("DISTRIBUCIÓN DEL RESULTADO")
print(f"  P&L esperado (media):     ${pnl.mean():>12,.0f}   ({roi.mean():+.1%} sobre el presupuesto)")
print(f"  P&L mediano:              ${np.median(pnl):>12,.0f}")
print(f"  Desvío estándar:          ${pnl.std():>12,.0f}")
print()
print(f"  Probabilidad de PERDER:   {(pnl < 0).mean():>12.1%}")
print(f"  VaR 95% (peor 5%):        ${np.percentile(pnl, 5):>12,.0f}")
print(f"  Expected Shortfall 95%:   ${pnl[pnl <= np.percentile(pnl, 5)].mean():>12,.0f}")
print(f"  Escenario optimista p95:  ${np.percentile(pnl, 95):>12,.0f}")

# Contraste didáctico: qué habría pasado ignorando la incertidumbre de parámetro
pnl_ingenuo = np.array([
    pool[rng.integers(0, n_pool, n_usuarios_comprados)].sum() - PRESUPUESTO
    for _ in range(1000)
])
print(f"""
  CONTRASTE — el mismo cálculo SIN incertidumbre de parámetro:
    P&L esperado:           ${pnl_ingenuo.mean():>12,.0f}
    Probabilidad de perder: {(pnl_ingenuo < 0).mean():>12.1%}   <-- da 0% y es MENTIRA
    Desvío:                 ${pnl_ingenuo.std():>12,.0f}   <-- {pnl.std()/max(pnl_ingenuo.std(),1):.0f}x más chico

  Ese "0% de probabilidad de perder" es el tipo de número que hace que alguien
  apruebe una inversión que después sale mal. La incertidumbre no estaba en qué
  usuarios te tocan: estaba en si el canal se sigue comportando como ayer.

  Traducción para el comité de inversión: 'retorno esperado positivo, pero con
  una probabilidad relevante de pérdida concentrada en el supuesto de que el
  canal no se deteriore; en el peor 5% de escenarios perdemos X'.""")


# ===========================================================================
# 3. SENSIBILIDAD: ¿cuánto tiene que bajar el CAC para que cierre?
# ===========================================================================
titulo("3. ANÁLISIS DE SENSIBILIDAD — el CAC de break-even")

print("Probabilidad de que la inversión sea rentable, según el CAC negociado:\n")
print(f"  {'CAC':>8}  {'usuarios':>10}  {'P&L esperado':>14}  {'P(perder)':>10}")
print("  " + "-" * 48)
for cac_test in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    n_u = int(PRESUPUESTO / cac_test)
    medias = np.array([pool[rng.integers(0, n_pool, n_pool)].mean() * rng.lognormal(-0.012, 0.15)
                       for _ in range(2000)])
    sims = medias * n_u - PRESUPUESTO
    print(f"  ${cac_test:>7.2f}  {n_u:>10,}  ${sims.mean():>13,.0f}  {(sims < 0).mean():>9.1%}")

print("""
    Esta tabla es la que llevás a la negociación con el ad network. No decís
    'el canal no cierra': decís 'a USD 0.30 la probabilidad de perder es X;
    necesito 0.25 para que el riesgo sea aceptable'. Eso es análisis de
    sensibilidad de toda la vida, aplicado a marketing.""")


# ===========================================================================
# 4. EL VALOR ECONÓMICO DEL MODELO (esto justifica tu sueldo)
# ===========================================================================
titulo("4. ¿CUÁNTA PLATA VALE EL MODELO DEL SCRIPT 04?")

print("""
Un modelo sin número de negocio adjunto no se aprueba. Comparamos dos
políticas de compra sobre el mismo conjunto de usuarios de test:

  POLÍTICA A (sin modelo): compro todos los usuarios de los canales cuyo LTV
                           promedio histórico supera su CAC.
  POLÍTICA B (con modelo): compro solo los usuarios cuyo LTV PREDICHO por el
                           modelo supera su CAC.

La diferencia entre A y B es el valor incremental del modelo.
(Limitación declarada: es un backtest, no un experimento. Asume targeting a
 nivel usuario y que el mercado no reacciona a los bids. La validación correcta
 es un A/B test.)
""")

pred = pd.read_parquet(DATA / "predicciones_test.parquet")

# La decisión de inversión aplica solo a canales PAGOS: al usuario orgánico no
# lo comprás, viene solo. Incluirlo infla el ROI de las dos políticas por igual
# y esconde la comparación. Filtrarlo es la diferencia entre un análisis que
# convence y uno que un director tira a la basura en 10 segundos.
pred = pred[pred["cac"] > 0].reset_index(drop=True)

# Política A: decisión por canal, con el promedio.
ltv_medio_canal = pred.groupby("canal")["real"].mean()
canales_ok = ltv_medio_canal[ltv_medio_canal > pred.groupby("canal")["cac"].mean()].index
mask_a = pred["canal"].isin(canales_ok)
pnl_a = pred.loc[mask_a, "real"].sum() - pred.loc[mask_a, "cac"].sum()

# Política B: decisión por usuario, con la predicción del modelo.
mask_b = pred["pred"] > pred["cac"]
pnl_b = pred.loc[mask_b, "real"].sum() - pred.loc[mask_b, "cac"].sum()

# Techo teórico: si supiéramos el futuro (oráculo). Sirve para saber cuánto
# margen de mejora queda: si ya estás cerca del oráculo, no gastes más tiempo.
mask_o = pred["real"] > pred["cac"]
pnl_o = pred.loc[mask_o, "real"].sum() - pred.loc[mask_o, "cac"].sum()

comp = pd.DataFrame([
    {"politica": "A - por canal (sin modelo)", "usuarios_comprados": int(mask_a.sum()),
     "inversion": pred.loc[mask_a, "cac"].sum(), "revenue": pred.loc[mask_a, "real"].sum(), "pnl": pnl_a},
    {"politica": "B - por usuario (con modelo)", "usuarios_comprados": int(mask_b.sum()),
     "inversion": pred.loc[mask_b, "cac"].sum(), "revenue": pred.loc[mask_b, "real"].sum(), "pnl": pnl_b},
    {"politica": "Oráculo (techo teórico)", "usuarios_comprados": int(mask_o.sum()),
     "inversion": pred.loc[mask_o, "cac"].sum(), "revenue": pred.loc[mask_o, "real"].sum(), "pnl": pnl_o},
])
comp["roi"] = comp["revenue"] / comp["inversion"]
print(comp.round(2).to_string(index=False))

mejora = pnl_b - pnl_a
captura = (pnl_b - pnl_a) / (pnl_o - pnl_a) if pnl_o != pnl_a else float("nan")
print(f"""
    Valor incremental del modelo en la muestra de test: ${mejora:,.0f}
    sobre {len(pred):,} usuarios → ${mejora/len(pred):.3f} por usuario.
    El modelo captura el {captura:.0%} del margen que capturaría un oráculo.

    ASÍ SE PRESENTA UN MODELO. No "logré spearman de 0.97": eso al negocio no
    le dice nada. Se presenta como "a este volumen de adquisición, la política
    basada en el modelo genera USD X adicionales por mes". El número técnico va
    en el apéndice.""")

titulo("CIERRE")
print("""
El aporte de este script no es el modelo sino el marco de decisión:

  · Un LTV/CAC puntual no alcanza para decidir. Lo que decide es la
    probabilidad de pérdida y el escenario adverso.
  · La incertidumbre que importa no es la de qué usuarios te tocan —esa se
    diluye con volumen— sino la de si el canal se sigue comportando como ayer.
  · Un modelo se presenta en unidades de margen incremental, no de métricas
    técnicas. El Spearman va en el apéndice.

Es el mismo criterio con el que se evalúa una posición en una mesa de dinero,
aplicado a la compra de usuarios.""")
