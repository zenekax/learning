"""
03 - SQL ANALÍTICO
===================

Las mismas preguntas del script 02, resueltas en SQL sobre los archivos parquet.

DuckDB es una base analítica embebida que corre SQL directamente sobre parquet,
sin servidor. El dialecto y el modelo de ejecución son equivalentes a los de
Athena, BigQuery o Redshift, así que las queries son portables a un data
warehouse real casi sin cambios.

PATRONES QUE CUBRE:
  1. CTE (WITH ... AS)     → partir una query compleja en pasos legibles
  2. WINDOW FUNCTIONS      → cálculos "por fila mirando su grupo" sin colapsarlo
  3. AGREGACIÓN CONDICIONAL→ COUNT(*) FILTER (WHERE ...) / SUM(CASE WHEN ...)
  4. PERCENTILES           → percentile_cont, porque la media miente

Correr:  python3 src/03_sql.py
"""

import duckdb
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"

con = duckdb.connect()
# Registramos los parquet como si fueran tablas. Ahora es SQL puro.
con.execute(f"CREATE VIEW usuarios AS SELECT * FROM '{DATA}/usuarios.parquet'")
con.execute(f"CREATE VIEW eventos  AS SELECT * FROM '{DATA}/eventos.parquet'")


def correr(titulo, explicacion, sql, limite=15):
    print("\n" + "=" * 76)
    print(titulo)
    print("=" * 76)
    print(explicacion.strip())
    print("\n--- SQL " + "-" * 68)
    print(sql.strip())
    print("--- resultado " + "-" * 62)
    df = con.execute(sql).df()
    print(df.head(limite).to_string(index=False))


# ===========================================================================
correr(
    "1. AGREGACIÓN CONDICIONAL — retención D1/D7/D30 en una sola pasada",
    """
El error de principiante es correr tres queries, una por cada métrica.
FILTER (WHERE ...) te deja calcular varias condiciones en una sola pasada sobre
la tabla. Con 500 millones de filas eso es la diferencia entre 3 minutos y 9.

(En Postgres/MySQL sin FILTER, el equivalente es
 COUNT(DISTINCT CASE WHEN cond THEN user_id END) — sabé escribir las dos.)
""",
    """
SELECT
    u.canal,
    COUNT(DISTINCT u.user_id)                                             AS instalaciones,
    COUNT(DISTINCT e.user_id) FILTER (WHERE e.dia_relativo = 1)           AS activos_d1,
    COUNT(DISTINCT e.user_id) FILTER (WHERE e.dia_relativo BETWEEN 7 AND 13)  AS activos_d7,
    ROUND(100.0 * COUNT(DISTINCT e.user_id) FILTER (WHERE e.dia_relativo = 1)
          / COUNT(DISTINCT u.user_id), 1)                                 AS ret_d1_pct,
    ROUND(100.0 * COUNT(DISTINCT e.user_id) FILTER (WHERE e.dia_relativo BETWEEN 7 AND 13)
          / COUNT(DISTINCT u.user_id), 1)                                 AS ret_d7_pct
FROM usuarios u
LEFT JOIN eventos e ON e.user_id = u.user_id
GROUP BY u.canal
ORDER BY ret_d7_pct DESC
""",
)

# ===========================================================================
correr(
    "2. CTE + PERCENTILES — por qué el ARPU promedio te miente",
    """
Un CTE (WITH nombre AS (...)) es una tabla temporal con nombre. Te deja escribir
la query como un razonamiento en pasos en vez de un monstruo anidado.

Acá calculamos LTV por usuario (paso 1) y recién después lo resumimos (paso 2).
La distancia entre la mediana y el p99 es la estructura del negocio mobile.
""",
    """
WITH ltv_usuario AS (
    SELECT
        u.user_id,
        u.canal,
        u.pais,
        COALESCE(SUM(e.revenue_ads + e.revenue_iap), 0) AS ltv_90d
    FROM usuarios u
    LEFT JOIN eventos e ON e.user_id = u.user_id
    GROUP BY u.user_id, u.canal, u.pais
)
SELECT
    pais,
    COUNT(*)                                                   AS usuarios,
    ROUND(AVG(ltv_90d), 3)                                     AS ltv_medio,
    ROUND(percentile_cont(0.50) WITHIN GROUP (ORDER BY ltv_90d), 3) AS mediana,
    ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY ltv_90d), 3) AS p95,
    ROUND(percentile_cont(0.99) WITHIN GROUP (ORDER BY ltv_90d), 2) AS p99,
    ROUND(MAX(ltv_90d), 2)                                     AS maximo,
    ROUND(100.0 * COUNT(*) FILTER (WHERE ltv_90d > 1) / COUNT(*), 2) AS pct_pagadores
FROM ltv_usuario
GROUP BY pais
ORDER BY ltv_medio DESC
""",
)

# ===========================================================================
correr(
    "3. WINDOW FUNCTIONS — concentración del revenue (la curva de Pareto)",
    """
Una window function calcula algo sobre un conjunto de filas SIN colapsarlas,
a diferencia de GROUP BY. Acá:

  ROW_NUMBER() OVER (ORDER BY ltv DESC)  → ranking de cada usuario
  SUM(ltv) OVER (ORDER BY ltv DESC)      → suma ACUMULADA hasta esa fila
  SUM(ltv) OVER ()                       → total general, repetido en cada fila

Dividiendo acumulado / total obtenés la curva de concentración. En finanzas es
literalmente una curva de Lorenz. Misma matemática que medir concentración de
riesgo por contraparte en una cartera.
""",
    """
WITH ltv_usuario AS (
    SELECT u.user_id, COALESCE(SUM(e.revenue_ads + e.revenue_iap), 0) AS ltv
    FROM usuarios u LEFT JOIN eventos e ON e.user_id = u.user_id
    GROUP BY u.user_id
),
ranking AS (
    SELECT
        user_id,
        ltv,
        ROW_NUMBER() OVER (ORDER BY ltv DESC)                        AS rank_usuario,
        SUM(ltv)     OVER (ORDER BY ltv DESC
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ltv_acum,
        SUM(ltv)     OVER ()                                         AS ltv_total,
        COUNT(*)     OVER ()                                         AS n_usuarios
    FROM ltv_usuario
)
SELECT
    ROUND(100.0 * rank_usuario / n_usuarios, 1) AS pct_usuarios,
    ROUND(100.0 * ltv_acum / ltv_total, 1)      AS pct_revenue_acumulado
FROM ranking
WHERE rank_usuario IN (
    CAST(n_usuarios * 0.001 AS INT), CAST(n_usuarios * 0.01 AS INT),
    CAST(n_usuarios * 0.05 AS INT),  CAST(n_usuarios * 0.10 AS INT),
    CAST(n_usuarios * 0.20 AS INT),  CAST(n_usuarios * 0.50 AS INT)
)
ORDER BY pct_usuarios
""",
)

# ===========================================================================
correr(
    "4. NTILE + agregación — deciles de valor (segmentación operativa)",
    """
NTILE(10) parte la población en 10 grupos iguales según un orden. Es la forma
estándar de armar deciles de valor para después accionar sobre ellos
(campañas, ofertas, atención al cliente).

En riesgo crediticio esto se llama "scorecard por deciles" y es la tabla con la
que se decide el corte de aprobación. Idéntico.
""",
    """
WITH ltv_usuario AS (
    SELECT u.user_id, u.canal, COALESCE(SUM(e.revenue_ads + e.revenue_iap), 0) AS ltv
    FROM usuarios u LEFT JOIN eventos e ON e.user_id = u.user_id
    GROUP BY u.user_id, u.canal
),
deciles AS (
    SELECT *, NTILE(10) OVER (ORDER BY ltv DESC) AS decil FROM ltv_usuario
)
SELECT
    decil,
    COUNT(*)                                    AS usuarios,
    ROUND(AVG(ltv), 4)                          AS ltv_medio,
    ROUND(SUM(ltv), 2)                          AS ltv_total,
    ROUND(100.0 * SUM(ltv) / SUM(SUM(ltv)) OVER (), 1) AS pct_del_revenue
FROM deciles
GROUP BY decil
ORDER BY decil
""",
)

# ===========================================================================
correr(
    "5. COHORTES EN SQL — retención por cohorte de instalación",
    """
Misma matriz de cohortes del script 02, pero en SQL. Esta query es un clásico
absoluto: 'dame retención por cohorte mensual y semana de vida'.

Truco de lectura: date_trunc('month', fecha_install) define la cohorte;
dia_relativo / 7 define la edad. El denominador viene de un CTE aparte porque
necesitás el tamaño de la cohorte, no la cantidad de filas de eventos.
""",
    """
WITH cohortes AS (
    SELECT
        user_id,
        strftime(date_trunc('month', fecha_install), '%Y-%m') AS cohorte
    FROM usuarios
),
tamanos AS (
    SELECT cohorte, COUNT(*) AS n FROM cohortes GROUP BY cohorte
),
actividad AS (
    SELECT
        c.cohorte,
        e.dia_relativo / 7 AS semana,
        COUNT(DISTINCT e.user_id) AS activos
    FROM eventos e
    JOIN cohortes c ON c.user_id = e.user_id
    WHERE e.dia_relativo < 56
    GROUP BY 1, 2
)
SELECT
    a.cohorte,
    t.n AS tamano_cohorte,
    MAX(CASE WHEN semana = 1 THEN ROUND(100.0*activos/t.n, 1) END) AS sem_1,
    MAX(CASE WHEN semana = 2 THEN ROUND(100.0*activos/t.n, 1) END) AS sem_2,
    MAX(CASE WHEN semana = 4 THEN ROUND(100.0*activos/t.n, 1) END) AS sem_4,
    MAX(CASE WHEN semana = 7 THEN ROUND(100.0*activos/t.n, 1) END) AS sem_7
FROM actividad a
JOIN tamanos t ON t.cohorte = a.cohorte
GROUP BY a.cohorte, t.n
ORDER BY a.cohorte
""",
)

# ===========================================================================
correr(
    "6. LAG / LEAD — detectar el día en que el usuario se va (churn)",
    """
LAG() te da el valor de la fila ANTERIOR dentro de la partición. Sirve para
calcular diferencias entre eventos consecutivos: días entre sesiones, tiempo
entre compras, gap entre pagos.

Acá medimos el gap entre días activos consecutivos por usuario. Un gap grande
es la firma del churn: el usuario se fue y volvió (o no volvió más).
""",
    """
WITH gaps AS (
    SELECT
        user_id,
        dia_relativo,
        dia_relativo - LAG(dia_relativo) OVER (
            PARTITION BY user_id ORDER BY dia_relativo
        ) AS dias_desde_ultima_sesion
    FROM eventos
)
SELECT
    CASE
        WHEN dias_desde_ultima_sesion = 1  THEN '1 - consecutivo'
        WHEN dias_desde_ultima_sesion <= 3 THEN '2 - gap corto (2-3d)'
        WHEN dias_desde_ultima_sesion <= 7 THEN '3 - gap medio (4-7d)'
        WHEN dias_desde_ultima_sesion <= 30 THEN '4 - gap largo (8-30d)'
        ELSE '5 - resucitado (>30d)'
    END AS tipo_gap,
    COUNT(*) AS sesiones,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM gaps
WHERE dias_desde_ultima_sesion IS NOT NULL
GROUP BY 1
ORDER BY 1
""",
)

print("\n" + "=" * 76)
print("""Todas las queries corren sobre los parquet sin cargarlos a memoria.
El mismo SQL es portable a Athena o BigQuery cambiando solo la referencia a la
tabla.""")
print("=" * 76)
