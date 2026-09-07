# Predicción de LTV en un juego mobile — proyecto de portfolio

Proyecto end-to-end de data science sobre monetización en gaming mobile.

Predice el **Lifetime Value a 90 días** de un usuario a partir de su
comportamiento en los **primeros 7 días**, y traduce esa predicción en una
decisión de inversión de marketing con su riesgo cuantificado.

---

## Resultado en una tabla

| Modelo | RMSE | MAE | Spearman | Lift decil 1 | % revenue capturado en top 20% |
|---|---|---|---|---|---|
| Baseline: media global | 4.76 | 1.02 | 0.00 | 0.8x | 17% |
| Baseline: media por canal | 4.75 | 1.00 | 0.23 | 1.4x | 39% |
| Ridge (target crudo) | **2.61** | 0.51 | 0.76 | 8.4x | 94% |
| Ridge (target log, acotado) | 6.91 | 0.54 | 0.87 | 8.2x | 93% |
| **Gradient Boosting (target log)** | 3.64 | **0.31** | **0.97** | **8.5x** | **94%** |

El mejor RMSE no es el mejor modelo. Ridge sobre el target crudo gana en RMSE
porque persigue a las ballenas; el gradient boosting sobre log ordena mucho
mejor a la población (Spearman 0.97 vs 0.76), que es lo que la decisión de
negocio realmente necesita. **Elegir la métrica según la decisión, no según la
costumbre, es el punto central del proyecto.**

**Impacto económico:** una política de compra basada en el LTV predicho por
usuario genera **+USD 2.046 de margen sobre la muestra de test** (11.074
usuarios de canales pagos) contra la política actual de decidir por promedio de
canal — el **97% del margen que capturaría un oráculo con información perfecta**.

---

## Cómo correr

```bash
pip install -r requirements.txt

python3 src/01_simulador.py     # genera el dataset (~10 s)
python3 src/02_exploracion.py   # EDA + cohortes
python3 src/03_sql.py           # las mismas preguntas en SQL analítico
python3 src/04_modelo_ltv.py    # baselines → modelos → evaluación
python3 src/05_montecarlo.py    # riesgo de la decisión + valor del modelo
```

Los scripts corren en orden y cada uno deja sus salidas en `data/`.
No hay dependencias externas ni credenciales: todo es reproducible con `seed=42`.

---

## Qué hay en cada script

**`01_simulador.py` — Simulación del proceso generador de datos**
50.000 usuarios con retención Weibull de hazard decreciente, monetización
zero-inflated lognormal, y heterogeneidad por canal, país y plataforma.
Calibrado contra benchmarks públicos de la industria:

| Métrica | Simulado | Benchmark |
|---|---|---|
| Retención D1 | 39.8% | 35-40% |
| Retención D7 | 18.6% | 15-18% |
| Retención D30 | 6.4% | 6-8% |
| Conversión a pagador | 5.0% | 2-5% |
| Top 1% de usuarios | 51% del revenue | 40-55% |

**`02_exploracion.py` — Exploración y cohortes**
Distribución del target (skewness 25.2 en crudo, 4.5 en log), matriz de
retención por cohorte mensual, unit economics por canal, y verificación de que
existe señal temprana: un usuario activo 7 días en su primera semana vale
**7.2x el promedio** a 90 días.

**`03_sql.py` — SQL analítico (DuckDB sobre parquet)**
Seis queries con agregación condicional (`FILTER`), CTEs, window functions
(`ROW_NUMBER`, `SUM OVER`, `NTILE`, `LAG`), percentiles y curva de
concentración. Incluye ejercicios sin resolver.

**`04_modelo_ltv.py` — Modelo predictivo**
Split temporal (no aleatorio), baselines obligatorios, tratamiento de la cola
larga, gradient boosting con early stopping, permutation importance, y tabla de
deciles con margen por decil. Documenta por qué `revenue_7d` es señal legítima
y no leakage.

**`06_decisiones.py` — Verificación empírica de la metodología**
Cada decisión del script 04 se rompe a propósito y se mide el efecto. Resultados
principales:

| Experimento | Hallazgo |
|---|---|
| Split aleatorio vs temporal | Con un deterioro real de +95% de sesgo, la validación aleatoria reportó −3% (luz verde); la temporal, +41% |
| Target log vs crudo | El log gana en los segmentos que suman el 98% de la población; el crudo solo en las ballenas |
| RMSE vs margen | Los rankings de modelos no coinciden: elegir por RMSE cuesta 7% del margen alcanzable |
| Leakage | Agregar `revenue_30d` (disponible recién el día 30) lleva el Spearman a 1.000 y el modelo a ser inejecutable |
| Boosting vs MLP | Resultado mixto: la red gana en MAE, el boosting en RMSE y ranking |

Hallazgo transversal no buscado: el MAE global casi no se movió mientras el LTV
del segmento pago caía 4 veces. Una métrica agregada puede estar sana mientras
el segmento sobre el que se decide se derrumba.

**`05_montecarlo.py` — Riesgo de la decisión**
Bootstrap del LTV medio por canal, Monte Carlo jerárquico separando
incertidumbre aleatoria de incertidumbre de parámetro, análisis de sensibilidad
del CAC de break-even, y cuantificación del valor económico del modelo.

El hallazgo más importante del script: una simulación que ignora la
incertidumbre de parámetro reporta **0% de probabilidad de pérdida**; incluirla
la lleva a **35%**. La diferencia es la que hace que se apruebe una inversión
que después sale mal.

---

## Decisiones metodológicas defendibles

1. **Split temporal, no aleatorio.** El modelo se usa sobre cohortes futuras.
   Un split aleatorio da un número optimista que no se reproduce en producción.
2. **`log1p` en el target.** Skewness de 25 rompe cualquier modelo que
   minimice error cuadrático. Costo asumido: sesgo de Jensen en el nivel
   absoluto, aceptable porque el uso es rankear, no proyectar totales.
3. **Predicciones acotadas.** Una regresión lineal extrapola sin límite; con
   `expm1` eso produce predicciones de cientos de millones de dólares por un
   solo outlier. Se acota a 3x el máximo observado en entrenamiento.
4. **Métricas de negocio junto a las técnicas.** Spearman, lift por decil y
   margen incremental en dólares, no solo RMSE.
5. **Gradient boosting antes que redes neuronales.** Se probó un MLP sobre las
   mismas features: gana en MAE, pierde en RMSE y en ranking. Como el modelo
   alimenta un ranking de usuarios, decide Spearman y ahí el boosting saca
   ventaja clara. El resultado es mixto y se reporta como tal.

---

## Limitaciones (declaradas, no escondidas)

- **Los datos son sintéticos.** Ventaja: el proceso generador es conocido y
  auditable. Desventaja: no hay las patologías de los datos reales
  (bots, atribución rota, fraude, eventos duplicados, GDPR).
- **La comparación de políticas es un backtest, no un experimento.** Asume
  targeting a nivel usuario y que el mercado no reacciona a los bids. La
  validación correcta es un A/B test.
- **No hay censura a la derecha.** En producción las cohortes recientes no
  cumplieron 90 días y su LTV está censurado; se resuelve con modelos de
  supervivencia o entrenando solo con cohortes maduras.
- **No hay pipeline productivo.** El modelo no está servido detrás de una API
  ni tiene monitoreo de drift.

---

## Qué demuestra el proyecto

| Competencia | Dónde está |
|---|---|
| Modelos predictivos de Lifetime Value | `04_modelo_ltv.py` |
| Simulaciones numéricas | `01_simulador.py`, `05_montecarlo.py` |
| Modelado estadístico | Weibull, lognormal zero-inflated, bootstrap, Monte Carlo jerárquico |
| SQL | `03_sql.py` — CTEs, window functions, percentiles |
| Python | Todo el proyecto |
| Proyectos end-to-end | Simulación → EDA → SQL → modelo → decisión de negocio |
| Orientación a producto y negocio | Cada script cierra en una decisión, no en una métrica |
| Criterio sobre el propio trabajo | Limitaciones declaradas, no escondidas |

**Lo que este proyecto no cubre y está en curso:** redes neuronales (PyTorch),
AWS (S3/Athena/SageMaker), Spark y NLP.

---

## Estructura

```
├── src/
│   ├── 01_simulador.py       # proceso generador de datos
│   ├── 02_exploracion.py     # EDA + cohortes
│   ├── 03_sql.py             # SQL analítico
│   ├── 04_modelo_ltv.py      # modelo predictivo
│   └── 05_montecarlo.py      # riesgo y valor económico
├── data/                     # generado, no versionado
└── requirements.txt
```
