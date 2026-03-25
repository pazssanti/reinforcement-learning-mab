# 🎰 Multi-Armed Bandit — Reinforcement Learning en Python

Exploración del problema **Multi-Armed Bandit (MAB)** en aprendizaje por refuerzo. Implementación y comparación estadística de tres algoritmos clásicos para el dilema exploración vs. explotación, con recompensas discretas.

## 📌 Problema

El problema MAB modela la toma de decisiones bajo incertidumbre: un agente elige entre múltiples opciones ("brazos"), cada una con una distribución de recompensa desconocida. El objetivo es **maximizar la recompensa acumulada** en el tiempo, equilibrando la exploración de nuevas opciones con la explotación de las mejores conocidas.

Aplicaciones reales: A/B testing, sistemas de recomendación, ensayos clínicos, trading algorítmico.

## ⚙️ Algoritmos implementados

### 1. Epsilon-Greedy (ε-greedy)
Explora aleatoriamente con probabilidad ε; explota la mejor opción conocida con probabilidad 1−ε.

$$\pi(a|s) = \begin{cases} 1-\varepsilon + \frac{\varepsilon}{|A|} & \text{si } a = \arg\max Q(s,a) \\ \frac{\varepsilon}{|A|} & \text{en otro caso} \end{cases}$$

### 2. Upper Confidence Bound (UCB)
Selecciona el brazo con el mayor límite superior de confianza, favoreciendo opciones poco exploradas.

$$a_t = \arg\max_a \left[ Q(s,a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]$$

### 3. Thompson Sampling
Enfoque bayesiano: muestrea desde distribuciones Beta actualizadas con éxitos y fracasos observados.

$$\text{Update: } \text{Beta}(S_a + 1,\ F_a + 1)$$

## 📈 Métricas de evaluación

- **Recompensa promedio** por paso de tiempo
- **Recompensa acumulada** vs. óptimo teórico
- **Arrepentimiento acumulado** (cumulative regret)
- **Comparación estadística:** ANOVA de una vía + Tukey HSD

## 🏆 Resultados

| Algoritmo | Recompensa promedio | Arrepentimiento | vs. Epsilon-Greedy |
|-----------|--------------------|-----------------|--------------------|
| Epsilon-Greedy | Línea base | Mayor | — |
| UCB | Superior | Menor | Diferencia significativa (p < 0.05) |
| **Thompson Sampling** | **Superior** | **Menor** | **Diferencia significativa (p < 0.05)** |

> **Conclusión:** Thompson Sampling y UCB superan significativamente a Epsilon-Greedy. No se encontraron diferencias significativas entre Thompson Sampling y UCB (Tukey HSD).

La clave de la superioridad de UCB y Thompson Sampling es su capacidad de **explotar la incertidumbre** de forma inteligente, evitando quedarse atascado en opciones subóptimas.

## 🔧 Optimización de hiperparámetros

| Algoritmo | Parámetro | Búsqueda |
|-----------|-----------|---------|
| Epsilon-Greedy | ε | Grid search |
| UCB | c (confianza) | Grid search |
| Thompson Sampling | α, β (prior Beta) | Grid search |

## 🛠 Herramientas

`Python` · `numpy` · `scipy` · `statsmodels` · `matplotlib`

## 📁 Contenido del repositorio

```
├── multi_armed_bandit.py       # Implementación completa
├── informe_final.pdf           # Informe con resultados y análisis estadístico
└── README.md
```

## 📚 Referencias

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2-3), 235–256.
- Chapelle, O. & Li, L. (2011). An empirical evaluation of Thompson sampling. *NeurIPS*.
