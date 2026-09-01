"""
Árboles de Decisión Multivariable — Sistema de Riego Inteligente
------------------------------------------------------------------
App companion interactiva del notebook `arbol_decision_multivariable.ipynb`.
Curso: Machine Learning · Módulo: Algoritmos de Clasificación (CRISP-DM)

Ejecutar localmente:  streamlit run app.py
Dependencias mínimas: streamlit, pandas, numpy, scikit-learn, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Árbol de Decisión Multivariable",
    page_icon="🌳",
    layout="wide",
)

# ----------------------------------------------------------------------------
# 1. Generación del dataset sintético (parametrizable desde la barra lateral)
# ----------------------------------------------------------------------------

@st.cache_data
def generar_dataset(n, ruido_pct, umbral_humedad, umbral_lluvia, hora_ini, hora_fin, seed):
    rng = np.random.default_rng(seed)

    temperatura = rng.uniform(10, 38, n)
    humedad_suelo = rng.uniform(5, 90, n)
    hora_dia = rng.integers(0, 24, n)
    prob_lluvia = rng.uniform(0, 100, n)

    suelo_seco = humedad_suelo < umbral_humedad
    sin_lluvia = prob_lluvia < umbral_lluvia
    hora_no_calurosa = (hora_dia < hora_ini) | (hora_dia > hora_fin)

    regar_base = suelo_seco & sin_lluvia & hora_no_calurosa
    ruido = rng.random(n) < (ruido_pct / 100)
    regar = np.where(ruido, ~regar_base, regar_base)

    df = pd.DataFrame({
        "temperatura": temperatura.round(1),
        "humedad_suelo": humedad_suelo.round(1),
        "hora_dia": hora_dia,
        "prob_lluvia": prob_lluvia.round(1),
        "regar": np.where(regar, "Sí", "No"),
    })
    return df


@st.cache_resource
def entrenar_arbol(df, features, max_depth, criterio, test_size, seed):
    X = df[features]
    y = df["regar"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    arbol = DecisionTreeClassifier(
        criterion=criterio, max_depth=max_depth, random_state=seed
    )
    arbol.fit(X_train, y_train)
    return arbol, X_train, X_test, y_train, y_test


def entropia(p):
    """Entropía de una variable binaria con proporción p para la clase positiva."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# ----------------------------------------------------------------------------
# Barra lateral: controles globales (dataset + hiperparámetros del árbol)
# ----------------------------------------------------------------------------

st.sidebar.header("⚙️ Parámetros del dataset")
n = st.sidebar.slider("Número de muestras (n)", 100, 2000, 600, step=50)
ruido_pct = st.sidebar.slider("Ruido de sensores (%)", 0, 30, 6)
umbral_humedad = st.sidebar.slider("Umbral suelo seco (humedad %)", 10, 60, 35)
umbral_lluvia = st.sidebar.slider("Umbral sin lluvia (prob. %)", 10, 70, 40)
hora_ini, hora_fin = st.sidebar.slider("Rango de horas calurosas (evitar)", 0, 23, (11, 18))
seed = st.sidebar.number_input("Semilla aleatoria", 0, 9999, 42)

st.sidebar.markdown("---")
st.sidebar.header("🌳 Hiperparámetros del árbol")
max_depth = st.sidebar.slider("Profundidad máxima (max_depth)", 1, 15, 3)
criterio = st.sidebar.selectbox("Criterio de división", ["entropy", "gini"], index=0)
test_size = st.sidebar.slider("Proporción de prueba (test_size)", 0.1, 0.5, 0.25, step=0.05)

df = generar_dataset(n, ruido_pct, umbral_humedad, umbral_lluvia, hora_ini, hora_fin, seed)
FEATURES = ["temperatura", "humedad_suelo", "hora_dia", "prob_lluvia"]
arbol, X_train, X_test, y_train, y_test = entrenar_arbol(
    df, FEATURES, max_depth, criterio, test_size, seed
)
y_pred = arbol.predict(X_test)

# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------

st.title("🌳 Árboles de Decisión Multivariable")
st.caption(
    "Sistema de riego inteligente · Cambia los parámetros de la izquierda y observa "
    "cómo cambian los datos, el árbol y sus métricas en tiempo real."
)

tabs = st.tabs([
    "📖 Teoría",
    "🌡️ Datos",
    "📊 Evaluación",
    "⭐ Importancia",
    "🌲 Ver árbol",
    "📉 Sobreajuste",
    "🗺️ Frontera 2D",
    "🔮 Predicción en vivo",
])

# ----------------------------------------------------------------------------
# TAB 1 — Teoría (entropía y ganancia de información, con calculadora en vivo)
# ----------------------------------------------------------------------------

with tabs[0]:
    st.subheader("¿Qué es un árbol de decisión?")
    st.markdown(
        "Un árbol de decisión clasifica haciendo una **secuencia de preguntas** "
        "del tipo `¿variable ≤ valor?`. Cada nodo interno prueba una variable, cada "
        "rama es una respuesta y cada hoja es una predicción final. El algoritmo "
        "elige, en cada nodo, la variable y el corte que **más reduce la entropía** "
        "(incertidumbre) del grupo."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Entropía**")
        st.latex(r"H = -\sum_i p_i \cdot \log_2(p_i)")
        st.markdown("**Ganancia de información**")
        st.latex(r"G = H(\text{padre}) - \sum_i \frac{n_i}{n} H(\text{hijo}_i)")

    with col2:
        st.markdown("**Calculadora interactiva de entropía (2 clases)**")
        p = st.slider("Proporción de la clase positiva (p)", 0.0, 1.0, 0.5, step=0.01)
        h = entropia(p)
        st.metric("Entropía H", f"{h:.3f} bits")
        xs = np.linspace(0.001, 0.999, 200)
        hs = [entropia(x) for x in xs]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, hs, color="tab:blue")
        ax.axvline(p, color="tab:red", linestyle="--", label=f"p = {p:.2f}")
        ax.scatter([p], [h], color="tab:red", zorder=5)
        ax.set_xlabel("p (proporción clase positiva)")
        ax.set_ylabel("Entropía H")
        ax.set_title("H(p) para un problema de 2 clases")
        ax.legend()
        st.pyplot(fig)
        st.caption(
            "Fíjate que H es máxima (1 bit) cuando las clases están 50/50 "
            "(máxima incertidumbre) y cae a 0 cuando un grupo es totalmente puro."
        )

    st.markdown("---")
    st.subheader("El problema: sistema de riego inteligente")
    st.markdown(
        """
        Simulamos sensores IoT que deciden si **regar o no**:

        | Variable | Descripción | Rango típico |
        |---|---|---|
        | `temperatura` | Temperatura ambiente (°C) | 10 – 38 |
        | `humedad_suelo` | Humedad del suelo (%) | 5 – 90 |
        | `hora_dia` | Hora del día (0–23) | 0 – 23 |
        | `prob_lluvia` | Probabilidad de lluvia (%) | 0 – 100 |

        La regla real (que el árbol debe *aproximar* a partir de los datos, con ruido de sensor) es:
        **regar si el suelo está seco, no va a llover pronto, y no es una hora calurosa.**
        """
    )

# ----------------------------------------------------------------------------
# TAB 2 — Exploración de datos
# ----------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(15), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Balance de clases**")
        conteo = df["regar"].value_counts(normalize=True).round(3)
        st.bar_chart(conteo)
        st.caption(
            f"Clase 'Sí': {conteo.get('Sí', 0):.1%} · Clase 'No': {conteo.get('No', 0):.1%}. "
            "Con reglas que exigen varias condiciones a la vez, la clase 'Sí' suele quedar minoritaria."
        )
    with col2:
        st.markdown("**Estadísticas descriptivas**")
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("**Dispersión interactiva (elige 2 variables)**")
    c1, c2 = st.columns(2)
    var_x = c1.selectbox("Eje X", FEATURES, index=1)
    var_y = c2.selectbox("Eje Y", FEATURES, index=3)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for clase, color in zip(["Sí", "No"], ["tab:blue", "tab:red"]):
        subset = df[df["regar"] == clase]
        ax.scatter(subset[var_x], subset[var_y], s=14, alpha=0.6, label=clase, color=color)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title(f"{var_x} vs. {var_y}")
    ax.legend(title="¿Regar?")
    st.pyplot(fig)

# ----------------------------------------------------------------------------
# TAB 3 — Entrenamiento y evaluación
# ----------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Resultado del entrenamiento")
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy en test", f"{acc:.3f}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Reporte de clasificación**")
        reporte = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(reporte).transpose().round(3), use_container_width=True)
        with st.expander("¿Qué significa cada métrica?"):
            st.markdown(
                "- **precision**: de lo que el modelo predijo como esa clase, ¿qué % acertó?\n"
                "- **recall**: de los casos reales de esa clase, ¿qué % detectó el modelo?\n"
                "- **f1-score**: media armónica entre precision y recall.\n"
                "- El **accuracy global** puede ocultar un mal desempeño en la clase minoritaria "
                "('Sí' suele tener menos ejemplos) — por eso conviene mirar precision/recall por clase."
            )

    with col2:
        st.markdown("**Matriz de confusión**")
        cm = confusion_matrix(y_test, y_pred, labels=arbol.classes_)
        fig, ax = plt.subplots(figsize=(4.2, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(arbol.classes_)))
        ax.set_yticks(range(len(arbol.classes_)))
        ax.set_xticklabels(arbol.classes_)
        ax.set_yticklabels(arbol.classes_)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        st.pyplot(fig)

# ----------------------------------------------------------------------------
# TAB 4 — Importancia de variables
# ----------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Importancia de cada variable")
    importancias = pd.Series(arbol.feature_importances_, index=FEATURES).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    importancias.plot(kind="barh", color="tab:blue", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (ganancia de información acumulada)")
    st.pyplot(fig)
    st.dataframe(importancias.rename("importancia").round(4), use_container_width=True)
    st.caption(
        "Esto suma, para cada variable, cuánta ganancia de información aportó en total "
        "a lo largo de todos los nodos del árbol — no solo en la raíz."
    )

# ----------------------------------------------------------------------------
# TAB 5 — Visualización del árbol
# ----------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Estructura del árbol entrenado")
    altura = max(6, 2.2 * (max_depth + 1))
    fig, ax = plt.subplots(figsize=(16, altura))
    plot_tree(
        arbol,
        feature_names=FEATURES,
        class_names=arbol.classes_,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    st.pyplot(fig)
    with st.expander("Ver árbol como texto (export_text)"):
        st.code(export_text(arbol, feature_names=FEATURES))
    st.caption(
        "En cada nodo: la variable y el corte elegido (`variable <= valor`), la impureza "
        "(`entropy`/`gini`), el número de muestras (`samples`) y la distribución de clases (`value`)."
    )

# ----------------------------------------------------------------------------
# TAB 6 — Profundidad y sobreajuste
# ----------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Accuracy vs. profundidad del árbol")

    @st.cache_data
    def curva_profundidad(_df, features, criterio, test_size, seed, max_prof=15):
        X = _df[features]
        y = _df["regar"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        profundidades = range(1, max_prof + 1)
        acc_train, acc_test = [], []
        for d in profundidades:
            modelo = DecisionTreeClassifier(criterion=criterio, max_depth=d, random_state=seed)
            modelo.fit(X_train, y_train)
            acc_train.append(accuracy_score(y_train, modelo.predict(X_train)))
            acc_test.append(accuracy_score(y_test, modelo.predict(X_test)))
        return list(profundidades), acc_train, acc_test

    profundidades, acc_train_l, acc_test_l = curva_profundidad(df, FEATURES, criterio, test_size, seed)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(profundidades, acc_train_l, marker="o", label="Entrenamiento")
    ax.plot(profundidades, acc_test_l, marker="o", label="Prueba")
    ax.axvline(max_depth, color="gray", linestyle="--", alpha=0.6, label=f"max_depth actual ({max_depth})")
    ax.set_xlabel("Profundidad máxima (max_depth)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(
        "Cuando la curva de entrenamiento sigue subiendo mientras la de prueba se estanca o "
        "baja, el árbol está empezando a sobreajustar (memorizar ruido). Ajusta `max_depth` "
        "en la barra lateral y observa cómo cambian las métricas de la pestaña Evaluación."
    )

# ----------------------------------------------------------------------------
# TAB 7 — Frontera de decisión en 2D (variables elegibles)
# ----------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Frontera de decisión con 2 variables")
    st.caption(
        "Los árboles de decisión solo pueden dividir el espacio con líneas rectas "
        "horizontales y verticales (nunca diagonales), porque cada corte usa una sola "
        "variable a la vez. Elige dos variables para verlo."
    )
    c1, c2, c3 = st.columns(3)
    fx = c1.selectbox("Variable X", FEATURES, index=1, key="fx2d")
    fy = c2.selectbox("Variable Y", [f for f in FEATURES if f != fx], index=0, key="fy2d")
    prof_2d = c3.slider("max_depth (solo para este árbol 2D)", 1, 10, 4)

    X2 = df[[fx, fy]]
    y_all = df["regar"]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )
    arbol_2d = DecisionTreeClassifier(criterion=criterio, max_depth=prof_2d, random_state=seed)
    arbol_2d.fit(X2_train, y2_train)

    x_min, x_max = X2[fx].min() - 2, X2[fx].max() + 2
    y_min, y_max = X2[fy].min() - 2, X2[fy].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    Z = arbol_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(Z == "Sí", 1, 0).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(xx, yy, Z, alpha=0.25, levels=1, cmap="coolwarm")
    for clase, color in zip(["Sí", "No"], ["tab:blue", "tab:red"]):
        subset = X2_test[y2_test == clase]
        ax.scatter(subset[fx], subset[fy], s=14, alpha=0.8, label=clase, color=color)
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_title(f"Frontera de decisión: {fx} vs. {fy}")
    ax.legend(title="¿Regar?")
    st.pyplot(fig)
    st.metric("Accuracy del árbol 2D (test)", f"{accuracy_score(y2_test, arbol_2d.predict(X2_test)):.3f}")

# ----------------------------------------------------------------------------
# TAB 8 — Predicción en vivo con el árbol completo (4 variables)
# ----------------------------------------------------------------------------

with tabs[7]:
    st.subheader("Prueba el árbol con tus propios valores")
    st.caption("Ajusta las 4 variables como si fueran lecturas de sensores y observa la predicción y el camino que sigue el árbol.")

    c1, c2 = st.columns(2)
    with c1:
        v_temp = st.slider("Temperatura (°C)", 10.0, 38.0, 24.0)
        v_humedad = st.slider("Humedad del suelo (%)", 5.0, 90.0, 40.0)
    with c2:
        v_hora = st.slider("Hora del día", 0, 23, 9)
        v_lluvia = st.slider("Probabilidad de lluvia (%)", 0.0, 100.0, 20.0)

    entrada = pd.DataFrame([{
        "temperatura": v_temp,
        "humedad_suelo": v_humedad,
        "hora_dia": v_hora,
        "prob_lluvia": v_lluvia,
    }])[FEATURES]

    pred = arbol.predict(entrada)[0]
    proba = arbol.predict_proba(entrada)[0]
    clases = arbol.classes_

    col1, col2 = st.columns([1, 1])
    with col1:
        if pred == "Sí":
            st.success(f"🌿 Predicción: **{pred}, regar**")
        else:
            st.info(f"🚫 Predicción: **{pred}, no regar**")
        for c, p in zip(clases, proba):
            st.write(f"P({c}) = {p:.2f}")

    with col2:
        st.markdown("**Camino recorrido en el árbol**")
        arbol_texto = export_text(arbol, feature_names=FEATURES)
        nodo_indicador = arbol.decision_path(entrada)
        hoja_id = arbol.apply(entrada)[0]
        nodos_recorridos = nodo_indicador.indices
        feature = arbol.tree_.feature
        threshold = arbol.tree_.threshold
        pasos = []
        for nodo_id in nodos_recorridos:
            if nodo_id == hoja_id:
                continue
            f = FEATURES[feature[nodo_id]]
            t = threshold[nodo_id]
            valor = entrada.iloc[0][f]
            direccion = "≤" if valor <= t else ">"
            pasos.append(f"¿{f} ≤ {t:.2f}? → {f} = {valor:.1f} → **{direccion}** → sigue por la rama {'izquierda' if direccion=='≤' else 'derecha'}")
        for i, paso in enumerate(pasos, 1):
            st.markdown(f"{i}. {paso}")
        st.caption(f"Llega a la hoja #{hoja_id} con predicción «{pred}».")

st.markdown("---")
st.caption(
    "App companion del notebook `arbol_decision_multivariable.ipynb` · "
    "Curso Machine Learning (ET0178) · Institución Universitaria Pascual Bravo"
)
