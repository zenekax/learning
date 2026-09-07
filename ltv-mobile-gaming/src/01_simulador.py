"""
01 - SIMULADOR DE USUARIOS DE UN JUEGO MOBILE
==============================================

Qué hace: genera un dataset sintético realista de usuarios de un juego trivia
mobile (tipo Preguntados), con instalaciones, retención diaria y monetización.

Por qué un dataset simulado y no uno descargado:

  1. El proceso generador es conocido y auditable. Cuando el modelo del script
     04 predice bien, se puede verificar exactamente de qué señal viene — algo
     imposible con un dataset opaco.
  2. Permite construir escenarios controlados (¿qué pasa si la retención cae un
     20%?) para validar que el pipeline responde como debe.
  3. La simulación misma es parte del entregable: modelar la dinámica de
     retención y monetización de un juego mobile exige entender el negocio.

Contrapartida, declarada: los datos sintéticos no contienen las patologías de
los datos reales (bots, atribución rota, fraude, eventos duplicados).

Correr:  python3 src/01_simulador.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# SEMILLA: fija la aleatoriedad para que el dataset sea reproducible.
# Regla de oro en DS: si no es reproducible, no es un experimento.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)

N_USUARIOS = 50_000
FECHA_INICIO = pd.Timestamp("2025-01-01")
VENTANA_INSTALL_DIAS = 180   # 6 meses de instalaciones
HORIZONTE_DIAS = 90          # medimos LTV a 90 días (estándar de la industria)

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
DATA.mkdir(exist_ok=True)


# ===========================================================================
# PARTE 1 — DIMENSIONES DEL USUARIO (canal, país, device)
# ===========================================================================
# Cada canal de adquisición trae usuarios de distinta CALIDAD y a distinto CAC.
# Esto es exactamente el problema de negocio: ¿en qué canal pongo la plata?
#
#   cuota      = qué % del volumen de instalaciones trae ese canal
#   calidad    = multiplicador sobre la propensión a retener y a gastar
#   cac_usd    = costo de adquirir un usuario por ese canal
CANALES = {
    #  nombre           cuota  calidad  cac_usd
    "organico":       (0.34,   1.35,    0.00),
    "cross_promo":    (0.16,   1.15,    0.10),
    "paid_social":    (0.30,   0.85,    0.30),
    "paid_search":    (0.14,   1.05,    0.55),
    "incentivado":    (0.06,   0.35,    0.12),   # tráfico basura: instala y se va
}

# El país mueve el ARPU muchísimo (poder adquisitivo + precios de la tienda).
PAISES = {
    #  nombre   cuota   mult_arpu
    "BR":      (0.30,   0.55),
    "AR":      (0.14,   0.40),
    "MX":      (0.18,   0.60),
    "ES":      (0.13,   1.30),
    "US":      (0.25,   2.30),
}


def _muestrear_categoria(diccionario, n):
    """Sortea n valores de un diccionario {nombre: (cuota, ...)} según su cuota.

    numpy no sabe de diccionarios, así que separamos claves y probabilidades.
    rng.choice() sortea con reemplazo respetando esas probabilidades.
    """
    nombres = list(diccionario.keys())
    probs = np.array([v[0] for v in diccionario.values()], dtype=float)
    probs = probs / probs.sum()          # normalizamos por si no suman exacto 1
    return rng.choice(nombres, size=n, p=probs)


canal = _muestrear_categoria(CANALES, N_USUARIOS)
pais = _muestrear_categoria(PAISES, N_USUARIOS)
es_ios = rng.random(N_USUARIOS) < 0.28   # ~28% iOS, resto Android

# Traducimos la categoría a su multiplicador numérico.
# np.array([...])[indices] es "vectorizar un lookup": mucho más rápido que un for.
calidad_canal = np.array([CANALES[c][1] for c in canal])
cac_usuario = np.array([CANALES[c][2] for c in canal])
mult_arpu_pais = np.array([PAISES[p][1] for p in pais])
mult_arpu_device = np.where(es_ios, 1.9, 1.0)   # iOS monetiza ~2x más

# Fecha de instalación: uniforme sobre la ventana.
dia_install = rng.integers(0, VENTANA_INSTALL_DIAS, size=N_USUARIOS)
fecha_install = FECHA_INICIO + pd.to_timedelta(dia_install, unit="D")


# ===========================================================================
# PARTE 2 — VARIABLES LATENTES: engagement y propensión a pagar
# ===========================================================================
# "Latente" = no observable directamente. El usuario tiene un nivel de interés
# real que nosotros nunca vemos; solo vemos sus CONSECUENCIAS (sesiones, compras).
# Todo modelo predictivo es, en el fondo, un intento de estimar esta variable
# latente a partir de sus consecuencias observables.
#
# Beta(a, b) da valores entre 0 y 1, asimétrica hacia la izquierda con a<b:
# muchos usuarios con poco engagement, pocos con mucho. Así es la realidad.
engagement = rng.beta(a=1.6, b=4.0, size=N_USUARIOS) * calidad_canal
engagement = np.clip(engagement, 0.01, 0.99)

# Propensión a pagar: correlacionada con engagement pero NO idéntica.
# (hay usuarios muy enganchados que jamás pagan, y viceversa)
ruido = rng.normal(0, 0.35, N_USUARIOS)
propension_pago = np.clip(engagement * 0.7 + ruido * 0.3, 0.001, 0.99)


# ===========================================================================
# PARTE 3 — RETENCIÓN: ¿cuántos días sobrevive el usuario?
# ===========================================================================
# La retención en mobile NO es exponencial (tasa de abandono constante).
# Sigue una ley de potencias: el churn es brutal el día 1 y se aplana después,
# porque los que sobreviven son cada vez mejores usuarios ("selección").
#
# Modelamos con una Weibull de forma k<1 = "hazard decreciente":
#   probabilidad de abandonar hoy, dado que sobreviviste hasta hoy, va bajando.
#
# En finanzas esto es EXACTAMENTE una curva de supervivencia de cartera:
# probabilidad de que un cliente/crédito siga vivo en t. Mismo objeto matemático.
# Parámetros CALIBRADOS: se ajustaron hasta que D1/D7/D30 caen dentro de los
# benchmarks públicos de la industria (ver bloque de validación al final).
# Sin esa calibración un simulador es solo un generador de números al azar.
escala_vida = 0.10 + (engagement ** 1.8) * 22.0
forma = 0.45                                # k < 1 → hazard decreciente
dias_vida = escala_vida * rng.weibull(forma, N_USUARIOS)
dias_vida = np.clip(np.round(dias_vida), 1, 400).astype(int)

# Benchmarks de la industria para validar: D1 ~35-40%, D7 ~15-18%, D30 ~6-8%
d1 = (dias_vida >= 2).mean()
d7 = (dias_vida >= 8).mean()
d30 = (dias_vida >= 31).mean()


# ===========================================================================
# PARTE 4 — GENERACIÓN DE EVENTOS DIARIOS
# ===========================================================================
# Para cada usuario generamos una fila por día activo, con:
#   sesiones, partidas jugadas, ads vistos, revenue por ads, revenue por compras.
#
# Ojo con el volumen: 50.000 usuarios generan cientos de miles de filas.
# Por eso construimos con listas de arrays y concatenamos UNA vez al final,
# en vez de hacer df.append() en un loop (eso es cuadrático y te mata el runtime).

filas_uid, filas_dia, filas_ses, filas_part = [], [], [], []
filas_rev_ads, filas_rev_iap = [], []

for i in range(N_USUARIOS):
    vida = min(dias_vida[i], HORIZONTE_DIAS)
    dias = np.arange(vida)                      # día 0, 1, 2, ... relativo al install

    # --- Intensidad de uso: decae con el tiempo (novedad que se gasta) ---
    decaimiento = np.exp(-dias / (escala_vida[i] * 1.5 + 4))
    lam_sesiones = (0.4 + engagement[i] * 4.0) * decaimiento
    sesiones = rng.poisson(lam_sesiones)        # Poisson = conteo de eventos por día

    # Días con 0 sesiones son días inactivos (el usuario "duerme" y vuelve).
    partidas = rng.poisson(sesiones * (2.0 + engagement[i] * 6))

    # --- Revenue por publicidad: proporcional a partidas, eCPM por país ---
    ads_vistos = rng.binomial(partidas, 0.55)
    ecpm = 0.008 * mult_arpu_pais[i] * mult_arpu_device[i]   # USD por impresión
    rev_ads = ads_vistos * ecpm

    # --- Revenue por compras in-app: ZERO-INFLATED + LOGNORMAL ---
    # Este es EL punto clave de la monetización mobile y hay que entenderlo:
    #   (a) la gran mayoría de los días/usuarios gastan exactamente 0
    #   (b) los que gastan siguen una lognormal → cola derecha muy larga
    #   (c) resultado: ~2% de usuarios ("ballenas"/whales) hace ~60% del revenue
    #
    # Esto rompe todo modelo que asuma normalidad. Es la razón por la que
    # después vamos a predecir log(LTV) y no LTV directo.
    p_compra_dia = propension_pago[i] * 0.075 * decaimiento
    compras = rng.binomial(1, np.clip(p_compra_dia, 0, 1))
    monto = rng.lognormal(mean=0.9, sigma=1.25, size=vida) * mult_arpu_pais[i] * mult_arpu_device[i]
    rev_iap = compras * monto

    filas_uid.append(np.full(vida, i, dtype=np.int32))
    filas_dia.append(dias.astype(np.int16))
    filas_ses.append(sesiones.astype(np.int16))
    filas_part.append(partidas.astype(np.int16))
    filas_rev_ads.append(rev_ads)
    filas_rev_iap.append(rev_iap)

eventos = pd.DataFrame({
    "user_id":      np.concatenate(filas_uid),
    "dia_relativo": np.concatenate(filas_dia),
    "sesiones":     np.concatenate(filas_ses),
    "partidas":     np.concatenate(filas_part),
    "revenue_ads":  np.concatenate(filas_rev_ads).round(6),
    "revenue_iap":  np.concatenate(filas_rev_iap).round(6),
})

# Fecha calendario = fecha de install + día relativo.
eventos = eventos.merge(
    pd.DataFrame({"user_id": np.arange(N_USUARIOS), "fecha_install": fecha_install}),
    on="user_id", how="left",
)
eventos["fecha"] = eventos["fecha_install"] + pd.to_timedelta(eventos["dia_relativo"], unit="D")
eventos = eventos.drop(columns=["fecha_install"])

# Sacamos los días totalmente inactivos (sin sesiones): así se ve un log real.
eventos = eventos[eventos["sesiones"] > 0].reset_index(drop=True)


# ===========================================================================
# PARTE 5 — TABLA DE USUARIOS
# ===========================================================================
usuarios = pd.DataFrame({
    "user_id": np.arange(N_USUARIOS),
    "fecha_install": fecha_install,
    "canal": canal,
    "pais": pais,
    "plataforma": np.where(es_ios, "ios", "android"),
    "cac_usd": cac_usuario.round(4),
    # dias_vida y engagement son variables LATENTES: en la vida real no existen.
    # Las guardamos solo para diagnóstico. NO se usan como features del modelo
    # (usarlas sería "data leakage": entrenar con la respuesta adentro).
    "_latente_dias_vida": dias_vida,
    "_latente_engagement": engagement.round(4),
})

DATA.mkdir(exist_ok=True)
usuarios.to_parquet(DATA / "usuarios.parquet", index=False)
eventos.to_parquet(DATA / "eventos.parquet", index=False)


# ===========================================================================
# VALIDACIÓN — ¿el dataset se parece a la realidad?
# ===========================================================================
_agg = eventos.groupby("user_id")[["revenue_ads", "revenue_iap"]].sum()
_agg = _agg.reindex(np.arange(N_USUARIOS), fill_value=0.0)
rev_total = _agg.sum(axis=1)
rev_iap_total = _agg["revenue_iap"]
pagadores = rev_iap_total > 0

print("=" * 68)
print("DATASET GENERADO")
print("=" * 68)
print(f"Usuarios:            {N_USUARIOS:,}")
print(f"Filas de eventos:    {len(eventos):,}")
print(f"Ventana instalación: {VENTANA_INSTALL_DIAS} días | Horizonte LTV: {HORIZONTE_DIAS} días")
print()
print("RETENCIÓN (benchmark industria entre paréntesis)")
print(f"  D1:  {d1:6.1%}   (35-40%)")
print(f"  D7:  {d7:6.1%}   (15-18%)")
print(f"  D30: {d30:6.1%}   ( 6-8%)")
print()
print("MONETIZACIÓN")
arpu = rev_total.mean()
print(f"  ARPU 90d (todos):          ${arpu:,.3f}")
print(f"  Conversión a pagador IAP:  {pagadores.mean():6.2%}   (benchmark 2-5%)")
print(f"  ARPPU (solo pagadores):    ${rev_total[pagadores].mean():,.2f}")
print(f"  Mix revenue: ads {_agg['revenue_ads'].sum()/rev_total.sum():.0%} / IAP {_agg['revenue_iap'].sum()/rev_total.sum():.0%}")

# Concentración: el famoso "las ballenas pagan la fiesta"
orden = rev_total.sort_values(ascending=False)
for pct in (0.01, 0.05, 0.10):
    top = orden.head(int(N_USUARIOS * pct)).sum() / orden.sum()
    print(f"  Top {pct:>4.0%} de usuarios genera {top:.1%} del revenue")

print()
print("LTV vs CAC POR CANAL  (esto es la decisión de negocio real)")
resumen = usuarios.assign(ltv_90d=rev_total.values).groupby("canal").agg(
    usuarios=("user_id", "count"),
    cac=("cac_usd", "mean"),
    ltv_90d=("ltv_90d", "mean"),
)
resumen["roi"] = np.where(resumen["cac"] > 0, resumen["ltv_90d"] / resumen["cac"], np.inf)
print(resumen.round(3).to_string())
print()
print(f"Guardado en: {DATA}")
