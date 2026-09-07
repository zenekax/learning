"""
02 - EXPLORACIÓN CON PANDAS + ANÁLISIS DE COHORTES
===================================================

Exploración del dataset generado en el script 01: distribución del target,
cohortes de retención, unit economics por canal y detección de señal temprana.

NOTA METODOLÓGICA — cohortes:
  Un análisis de cohortes de producto es formalmente idéntico a un análisis de
  cartera por camada de originación: se agrupa por momento de alta, se sigue el
  comportamiento en el tiempo y se comparan camadas entre sí para separar el
  efecto "cambió el producto" del efecto "cambió el mix de usuarios". Es la
  misma herramienta con la que se lee mora por cosecha en riesgo crediticio.

Correr:  python3 src/02_exploracion.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"

usuarios = pd.read_parquet(DATA / "usuarios.parquet")
eventos = pd.read_parquet(DATA / "eventos.parquet")


def titulo(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ===========================================================================
# 1. RECONOCIMIENTO DEL TERRENO
# ===========================================================================
# Lo primero que hace cualquier DS con una tabla nueva. Siempre. Sin excepción.
titulo("1. ¿QUÉ TENGO ENTRE MANOS?")
print(">>> usuarios.head(3)")
print(usuarios.head(3).to_string())
print(f"\nusuarios: {usuarios.shape[0]:,} filas x {usuarios.shape[1]} columnas")
print(f"eventos:  {eventos.shape[0]:,} filas x {eventos.shape[1]} columnas")

print("\n>>> ¿Hay nulos? (si los hubiera, decidir qué hacer ANTES de modelar)")
print(usuarios.isna().sum().to_string())

print("\n>>> Distribución por canal:  usuarios['canal'].value_counts(normalize=True)")
print(usuarios["canal"].value_counts(normalize=True).round(3).to_string())


# ===========================================================================
# 2. LA VARIABLE OBJETIVO: LTV a 90 días
# ===========================================================================
titulo("2. LA VARIABLE OBJETIVO (target): LTV 90 DÍAS POR USUARIO")

# groupby + sum: colapsamos ~178k eventos a 1 fila por usuario.
ltv = (
    eventos.groupby("user_id")[["revenue_ads", "revenue_iap"]]
    .sum()
    .reindex(usuarios["user_id"], fill_value=0.0)   # los que nunca volvieron: 0
)
ltv["ltv_90d"] = ltv["revenue_ads"] + ltv["revenue_iap"]

print(">>> ltv['ltv_90d'].describe()")
print(ltv["ltv_90d"].describe().round(4).to_string())

# ---- ACÁ ESTÁ EL PUNTO MÁS IMPORTANTE DE TODO EL PROYECTO ----------------
# Mirá la diferencia entre media y mediana. La media es varias veces la mediana:
# la distribución está brutalmente sesgada a la derecha. Esto NO es un detalle
# estadístico, es la naturaleza del negocio: unos pocos usuarios pagan todo.
#
# Consecuencias prácticas sobre el modelado:
#   - La media es un mal resumen. Reportar "ARPU" sin percentiles engaña.
#   - Modelos que minimizan error cuadrático (RMSE) van a obsesionarse con las
#     ballenas y arruinar la predicción del 95% restante.
#   - Solución habitual: predecir log(1 + LTV), o partir el problema en dos
#     (¿va a pagar? y si paga, ¿cuánto?). Lo vemos en el script 04.
q = ltv["ltv_90d"].quantile([0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
print("\n>>> Percentiles (la media miente cuando hay cola larga)")
print(f"    media   ${ltv['ltv_90d'].mean():8.3f}")
for k, v in q.items():
    print(f"    p{k*100:<6.1f} ${v:8.3f}")
print(f"    máximo  ${ltv['ltv_90d'].max():8.3f}")

asimetria = ltv["ltv_90d"].skew()
print(f"\n    Asimetría (skewness): {asimetria:.1f}  → una normal tiene 0.")
print(f"    Asimetría de log(1+LTV): {np.log1p(ltv['ltv_90d']).skew():.1f}  ← por eso se usa el log")


# ===========================================================================
# 3. ANÁLISIS DE COHORTES (= análisis por camada de originación)
# ===========================================================================
titulo("3. COHORTES DE RETENCIÓN")

# Definimos la cohorte por MES de instalación.
usuarios["cohorte"] = usuarios["fecha_install"].dt.to_period("M").astype(str)

# Unimos la cohorte a los eventos (merge = JOIN de SQL).
ev = eventos.merge(usuarios[["user_id", "cohorte", "canal", "pais"]], on="user_id", how="left")

# ¿Cuántos usuarios de cada cohorte siguen activos en la semana N?
ev["semana"] = ev["dia_relativo"] // 7

activos = ev.groupby(["cohorte", "semana"])["user_id"].nunique().reset_index(name="activos")
tamano_cohorte = usuarios.groupby("cohorte")["user_id"].count().rename("tamano")

activos = activos.merge(tamano_cohorte, on="cohorte")
activos["retencion"] = activos["activos"] / activos["tamano"]

# pivot_table convierte formato largo → matriz. Es LA visualización de cohortes.
matriz = activos.pivot_table(index="cohorte", columns="semana", values="retencion")
print(">>> Retención por cohorte de instalación (filas) y semana de vida (columnas)")
print((matriz.iloc[:, :9] * 100).round(1).to_string())
print("\n    Leer así: se lee por FILA (cómo envejece cada cohorte) y por COLUMNA")
print("    (si la semana 1 empeora mes a mes, algo cambió: producto o calidad de")
print("     tráfico). Igual que leer mora por cosecha.")


# ===========================================================================
# 4. LA PREGUNTA DE NEGOCIO: ¿DÓNDE PONGO LA PLATA?
# ===========================================================================
titulo("4. UNIT ECONOMICS POR CANAL")

u = usuarios.join(ltv["ltv_90d"], on="user_id")

resumen = u.groupby("canal").agg(
    usuarios=("user_id", "count"),
    cac=("cac_usd", "mean"),
    ltv_medio=("ltv_90d", "mean"),
    ltv_mediana=("ltv_90d", "median"),
    pct_pagadores=("ltv_90d", lambda s: (s > 1).mean()),
)
resumen["ltv_cac"] = np.where(resumen["cac"] > 0, resumen["ltv_medio"] / resumen["cac"], np.nan)
resumen["margen_usd"] = resumen["ltv_medio"] - resumen["cac"]
resumen["contribucion_total"] = resumen["margen_usd"] * resumen["usuarios"]

print(resumen.sort_values("ltv_cac", ascending=False).round(3).to_string())
print("""
    LECTURA DE TESORERO:
      · cross_promo:   LTV/CAC ~6.7 → escalar todo lo que se pueda.
      · paid_social:   LTV/CAC ~1.1 → está EN EL FILO. Con una regla de pulgar
                       de "LTV/CAC > 3" lo matás; pero el margen absoluto es
                       positivo y trae volumen. Acá el modelo agrega valor real:
                       predecir QUÉ usuario vale permite pujar distinto por
                       cada uno en vez de decidir por el promedio del canal.
      · paid_search:   LTV/CAC ~0.95 → pierde plata a 90 días. ¿Y a 180? Ese
                       es el argumento clásico para extender el horizonte.
      · incentivado:   LTV/CAC ~0.3 → cortar. No hay defensa posible.

    Ojo con el promedio: es el error #1. El LTV medio de un canal está dominado
    por 2 o 3 ballenas. Por eso la columna mediana está casi toda en cero.""")


# ===========================================================================
# 5. LO QUE SE VE EN LOS PRIMEROS 7 DÍAS, ¿PREDICE EL LTV A 90?
# ===========================================================================
titulo("5. ¿HAY SEÑAL TEMPRANA? (esto habilita todo el modelo predictivo)")

# El negocio no puede esperar 90 días para saber si un canal sirve.
# La pregunta es: con lo que veo en la primera semana, ¿puedo anticipar el LTV?
temprano = ev[ev["dia_relativo"] < 7].groupby("user_id").agg(
    sesiones_7d=("sesiones", "sum"),
    partidas_7d=("partidas", "sum"),
    dias_activos_7d=("dia_relativo", "nunique"),
    revenue_7d=("revenue_iap", "sum"),
).reindex(usuarios["user_id"], fill_value=0)

comp = temprano.join(ltv["ltv_90d"])
print(">>> Correlación de cada señal temprana con el LTV a 90 días")
print(comp.corr()["ltv_90d"].drop("ltv_90d").round(3).sort_values(ascending=False).to_string())

print("\n>>> LTV 90d medio según días activos en la primera semana")
tabla = comp.groupby("dias_activos_7d").agg(
    usuarios=("ltv_90d", "count"),
    ltv_90d_medio=("ltv_90d", "mean"),
)
tabla["indice_vs_promedio"] = (tabla["ltv_90d_medio"] / comp["ltv_90d"].mean()).round(2)
print(tabla.round(3).to_string())
print("""
    Un usuario activo 6-7 días en su primera semana vale un orden de magnitud
    más que uno que entró una vez. ESA es la señal que explota el modelo del
    script 04, y es la que te permite decidir el bid de marketing en el día 7
    en vez del día 90.""")

# Guardamos las features y el target para el script 04.
dataset = (
    usuarios[["user_id", "canal", "pais", "plataforma", "cac_usd", "cohorte", "fecha_install"]]
    .merge(temprano.reset_index(), on="user_id", how="left")
    .merge(ltv[["ltv_90d"]].reset_index(), on="user_id", how="left")
)
dataset.to_parquet(DATA / "dataset_modelo.parquet", index=False)
print(f"\nDataset para modelar guardado: {DATA / 'dataset_modelo.parquet'}  ({len(dataset):,} filas)")
