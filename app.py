import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# 1. Configuración de la Página
# ============================================
st.set_page_config(
    page_title="Simulador de Biorreactor: Transición Batch a Continuo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Simulador Dinámico de Biorreactor (Método de Euler)")
st.markdown("""
Esta aplicación simula el estado transitorio al iniciar un cultivo continuo (Quimiostato) 
después de una etapa Batch. Resuelve los balances de masa usando el método numérico de Euler.
""")

# ============================================
# 2. Panel Lateral - Parámetros de Entrada
# ============================================
st.sidebar.header("Parámetros del Sistema")

with st.sidebar.expander("🧫 Cinética (Monod) y Rendimiento", expanded=True):
    mu_max = st.number_input("µ_max (1/h)", min_value=0.0, value=0.5, step=0.05, format="%.2f")
    Ks = st.number_input("Ks (g/L)", min_value=0.001, value=0.1, step=0.05, format="%.2f")
    Yxs = st.number_input("Rendimiento Yx/s (g X/g S)", min_value=0.01, value=0.5, step=0.05, format="%.2f")

with st.sidebar.expander("⚙️ Operación y Diseño", expanded=True):
    V = st.number_input("Volumen del Reactor V (L)", min_value=0.1, value=2.0, step=0.1)
    F = st.number_input("Caudal de Entrada F (L/h)", min_value=0.0, value=0.6, step=0.1)
    S_in = st.number_input("Sustrato de Entrada S_in (g/L)", min_value=0.0, value=10.0, step=0.5)

with st.sidebar.expander("⏱️ Condiciones Iniciales y Tiempo", expanded=True):
    st.markdown("Valores al final del Batch (t=0 del continuo)")
    X0 = st.number_input("Biomasa Inicial X(0) (g/L)", min_value=0.0, value=1.0, step=0.1)
    S0 = st.number_input("Sustrato Inicial S(0) (g/L)", min_value=0.0, value=0.5, step=0.1)
    t0 = st.number_input("Tiempo inicial t(0) (h)", value=0.0)
    t_final = st.number_input("Tiempo a evaluar (h)", min_value=1.0, value=30.0, step=1.0)

# ============================================
# 3. Cálculos Preliminares
# ============================================
# Cálculo de Dilución
if V > 0:
    D_op = F / V
else:
    D_op = 0

# Cálculo de Dm según la imagen proporcionada
# Asumimos que 'Se' en la imagen se refiere a la concentración de entrada S_in
if Ks + S_in > 0 and S_in >= 0:
    try:
        term_sqrt = np.sqrt(Ks / (Ks + S_in))
        Dm_image = mu_max * (1 - term_sqrt)
    except:
        Dm_image = 0 # Manejo de errores si la raíz cuadrada es inválida
else:
    Dm_image = 0

st.header("1. Resultados Preliminares")
col1, col2 = st.columns(2)
col1.metric(label="Dilución Operativa (D = F/V)", value=f"{D_op:.4f} (1/h)")
col2.metric(label="Parámetro Dm (Ec. Imagen)", value=f"{Dm_image:.4f}", help="Calculado usando S_in como Se en la fórmula proporcionada.")

# ============================================
# 4. Motor de Simulación (Método de Euler)
# ============================================
@st.cache_data # Usamos caché para no recalcular si los inputs no cambian
def run_simulation(mu_max, Ks, Yxs, D, S_in, X0, S0, t0, t_final):
    # Definición del paso de tiempo
    n_steps = 10000
    dt = (t_final - t0) / n_steps
    
    # Inicialización de arrays
    t = np.linspace(t0, t_final, n_steps + 1)
    X = np.zeros(n_steps + 1)
    S = np.zeros(n_steps + 1)
    mu_vals = np.zeros(n_steps + 1)
    qs_vals = np.zeros(n_steps + 1)
    dXdt_vals = np.zeros(n_steps + 1) # Para verificar estado estacionario
    
    # Condiciones iniciales
    X[0] = X0
    S[0] = S0
    
    steady_state_idx = None
    found_ss = False
    
    # Bucle de Euler
    for i in range(n_steps):
        # a. Calcular cinéticas actuales
        # Evitar división por cero si S es muy pequeño y negativo por error numérico
        S_curr = max(0, S[i]) 
        mu = mu_max * S_curr / (Ks + S_curr)
        qs = mu / Yxs
        
        # Guardar valores cinéticos
        mu_vals[i] = mu
        qs_vals[i] = qs
        
        # b. Calcular derivadas
        dX_dt = (mu - D) * X[i]
        dS_dt = D * (S_in - S[i]) - qs * X[i]
        dXdt_vals[i] = dX_dt

        # Detección de estado estacionario: |dX/dt| <= 1e-6
        # Ignoramos los primeros pasos para evitar transitorios iniciales rápidos
        if i > 100 and not found_ss and abs(dX_dt) <= 1e-6:
            steady_state_idx = i
            found_ss = True

        # c. Paso de Euler: X_new = X_old + dt * dX/dt
        X[i+1] = X[i] + dt * dX_dt
        S[i+1] = S[i] + dt * dS_dt

    # Rellenar los últimos valores cinéticos para que coincidan los tamaños de array
    S_last = max(0, S[-1])
    mu_vals[-1] = mu_max * S_last / (Ks + S_last)
    qs_vals[-1] = mu_vals[-1] / Yxs
    dXdt_vals[-1] = (mu_vals[-1] - D) * X[-1]
    
    # Crear un DataFrame con los resultados
    df_results = pd.DataFrame({
        "Tiempo (h)": t,
        "Biomasa X (g/L)": X,
        "Sustrato S (g/L)": S,
        "µ (1/h)": mu_vals,
        "qs (g/g/h)": qs_vals,
        "dX/dt": dXdt_vals
    })
    
    return df_results, steady_state_idx, dt

# Ejecutar la simulación
df, ss_index, delta_t = run_simulation(mu_max, Ks, Yxs, D_op, S_in, X0, S0, t0, t_final)

st.write(f"*Simulación realizada con un paso de tiempo Δt = {delta_t:.6f} h*")

# ============================================
# 5. Visualización Gráfica (Plotly)
# ============================================
st.header("2. Dinámica del Proceso")

tab1, tab2 = st.tabs(["Concentraciones (X, S)", "Cinéticas (µ, qs)"])

with tab1:
    fig_conc = go.Figure()
    fig_conc.add_trace(go.Scatter(x=df["Tiempo (h)"], y=df["Biomasa X (g/L)"], mode='lines', name='Biomasa (X)'))
    fig_conc.add_trace(go.Scatter(x=df["Tiempo (h)"], y=df["Sustrato S (g/L)"], mode='lines', name='Sustrato (S)'))
    
    if ss_index:
        t_ss = df.iloc[ss_index]["Tiempo (h)"]
        fig_conc.add_vline(x=t_ss, line_width=2, line_dash="dash", line_color="green", annotation_text="Inicio EE")

    fig_conc.update_layout(title="Evolución de Biomasa y Sustrato",
                           xaxis_title="Tiempo (h)", yaxis_title="Concentración (g/L)", hovermode="x unified")
    st.plotly_chart(fig_conc, use_container_width=True)

with tab2:
    fig_kin = go.Figure()
    fig_kin.add_trace(go.Scatter(x=df["Tiempo (h)"], y=df["µ (1/h)"], mode='lines', name='Vel. Crecimiento Esp. (µ)', line=dict(color='purple')))
    fig_kin.add_trace(go.Scatter(x=df["Tiempo (h)"], y=df["qs (g/g/h)"], mode='lines', name='Vel. Consumo Esp. (qs)', line=dict(color='orange')))
    
    if ss_index:
        t_ss = df.iloc[ss_index]["Tiempo (h)"]
        fig_kin.add_vline(x=t_ss, line_width=2, line_dash="dash", line_color="green", annotation_text="Inicio EE")

    fig_kin.update_layout(title="Parámetros Cinéticos",
                           xaxis_title="Tiempo (h)", yaxis_title="Valor Específico (1/h)", hovermode="x unified")
    st.plotly_chart(fig_kin, use_container_width=True)


# ============================================
# 6. Tabla de Resultados
# ============================================
st.header("3. Puntos de Operación Clave")

# Extraer datos finales
row_final = df.iloc[-1].copy()
row_final["Estado"] = f"Tiempo Final Evaluado (t={t_final}h)"

# Extraer datos de estado estacionario si existe
rows_to_show = [row_final]

if ss_index is not None:
    row_ss = df.iloc[ss_index].copy()
    row_ss["Estado"] = f"Inicio Estado Estacionario (|dX/dt| ≤ 1e-6) (t≈{row_ss['Tiempo (h)']: .2f}h)"
    rows_to_show.insert(0, row_ss) # Poner EE primero
else:
    st.warning("⚠️ No se alcanzó el criterio de estado estacionario (|dX/dt| ≤ 1e-6) en el tiempo evaluado.")

# Crear el DataFrame para la tabla
df_table = pd.DataFrame(rows_to_show)

# Seleccionar y ordenar columnas para mostrar
cols_display = ["Estado", "Tiempo (h)", "Biomasa X (g/L)", "Sustrato S (g/L)", "µ (1/h)", "qs (g/g/h)"]
st.dataframe(df_table[cols_display].style.format("{:.4f}", subset=cols_display[1:]), use_container_width=True)

st.markdown("---")
st.caption("Desarrollado para docencia en bioprocesos. Implementación numérica: Método de Euler.")
