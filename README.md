# learning

Proyectos con los que aprendo haciendo. Cada carpeta es un proyecto completo,
reproducible y documentado.

---

### [`ltv-mobile-gaming/`](./ltv-mobile-gaming) — Predicción de Lifetime Value en un juego mobile

Pipeline end-to-end de data science: simulación del proceso generador de datos,
exploración, SQL analítico, modelo predictivo con validación temporal y
cuantificación del riesgo de la decisión de negocio.

Predice el LTV a 90 días de un usuario a partir de sus primeros 7 días, y
traduce esa predicción en una política de inversión de marketing con su riesgo
medido.

**Resultados:** Spearman 0.97 · lift 8.5x en el decil superior · el 94% del
revenue capturado en el 20% de usuarios mejor rankeados · +USD 2.046 de margen
incremental sobre la muestra de test, equivalente al 97% del techo teórico de un
oráculo con información perfecta.

`Python` · `pandas` · `numpy` · `scikit-learn` · `DuckDB` · `SQL analítico` ·
`Monte Carlo` · `bootstrap`

---

Nacho Basso — tesorería y finanzas, construyendo hacia data science.
