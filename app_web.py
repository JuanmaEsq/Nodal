import json
import re
import math
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from Pynite import FEModel3D

# ==========================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================================
st.set_page_config(
    page_title="Nodal · Sistema de Análisis Estructural",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 0;
    }

    .step-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        padding: 0 16px;
        color: #94a3b8;
        font-weight: 600;
        border: 1px solid #334155;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# BIBLIOTECA DE PLANTILLAS CANÓNICAS
# ==========================================================
PLANTILLAS = {
    "1. Viga Simplemente Apoyada - Carga Puntual al Centro": {
        "unidad_fuerza": "kN",
        "unidad_longitud": "m",
        "nodos": [
            {"id": "N1", "x": 0.0, "y": 0.0},
            {"id": "N2", "x": 6.0, "y": 0.0}
        ],
        "rotulas": [],
        "barras": [
            {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"}
        ],
        "apoyos": [
            {"nodo": "N1", "tipo": "fijo", "angulo": 0.0},
            {"nodo": "N2", "tipo": "movil", "angulo": 0.0}
        ],
        "cargas_distribuidas": [],
        "cargas_puntuales": [
            {"barra": "Barra1", "posicion_x": 3.0, "magnitud": 20.0, "direccion": "-Y", "angulo": 270.0}
        ],
        "momentos_concentrados": []
    },
    "2. Viga Simplemente Apoyada - Carga Distribuida Uniforme (q)": {
        "unidad_fuerza": "kN",
        "unidad_longitud": "m",
        "nodos": [
            {"id": "N1", "x": 0.0, "y": 0.0},
            {"id": "N2", "x": 6.0, "y": 0.0}
        ],
        "rotulas": [],
        "barras": [
            {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"}
        ],
        "apoyos": [
            {"nodo": "N1", "tipo": "fijo", "angulo": 0.0},
            {"nodo": "N2", "tipo": "movil", "angulo": 0.0}
        ],
        "cargas_distribuidas": [
            {"barra": "Barra1", "w_inicio": 10.0, "w_fin": 10.0, "x_inicio": 0.0, "x_fin": 6.0, "direccion": "-Y"}
        ],
        "cargas_puntuales": [],
        "momentos_concentrados": []
    },
    "3. Viga Simplemente Apoyada - Par / Momento Concentrado (M₀)": {
        "unidad_fuerza": "kN",
        "unidad_longitud": "m",
        "nodos": [
            {"id": "N1", "x": 0.0, "y": 0.0},
            {"id": "N2", "x": 6.0, "y": 0.0}
        ],
        "rotulas": [],
        "barras": [
            {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"}
        ],
        "apoyos": [
            {"nodo": "N1", "tipo": "fijo", "angulo": 0.0},
            {"nodo": "N2", "tipo": "movil", "angulo": 0.0}
        ],
        "cargas_distribuidas": [],
        "cargas_puntuales": [],
        "momentos_concentrados": [
            {"barra": "Barra1", "posicion_x": 3.0, "magnitud": 25.0, "sentido": "horario"}
        ]
    },
    "4. Viga en Voladizo (Ménsula) - Carga Puntual en Extremo": {
        "unidad_fuerza": "kN",
        "unidad_longitud": "m",
        "nodos": [
            {"id": "N1", "x": 0.0, "y": 0.0},
            {"id": "N2", "x": 3.0, "y": 0.0}
        ],
        "rotulas": [],
        "barras": [
            {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"}
        ],
        "apoyos": [
            {"nodo": "N1", "tipo": "empotrado", "angulo": 0.0}
        ],
        "cargas_distribuidas": [],
        "cargas_puntuales": [
            {"nodo": "N2", "magnitud": 15.0, "direccion": "-Y", "angulo": 270.0}
        ],
        "momentos_concentrados": []
    },
    "5. Viga en Voladizo (Ménsula) - Carga Distribuida Uniforme": {
        "unidad_fuerza": "kN",
        "unidad_longitud": "m",
        "nodos": [
            {"id": "N1", "x": 0.0, "y": 0.0},
            {"id": "N2", "x": 4.0, "y": 0.0}
        ],
        "rotulas": [],
        "barras": [
            {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"}
        ],
        "apoyos": [
            {"nodo": "N1", "tipo": "empotrado", "angulo": 0.0}
        ],
        "cargas_distribuidas": [
            {"barra": "Barra1", "w_inicio": 8.0, "w_fin": 8.0, "x_inicio": 0.0, "x_fin": 4.0, "direccion": "-Y"}
        ],
        "cargas_puntuales": [],
        "momentos_concentrados": []
    },
    "6. Viga Continua de 2 Tramos Simétricos": {
        "unidad_fuerza": "kN",
        "unidad_longitud": "m",
        "nodos": [
            {"id": "N1", "x": 0.0, "y": 0.0},
            {"id": "N2", "x": 4.0, "y": 0.0},
            {"id": "N3", "x": 8.0, "y": 0.0}
        ],
        "rotulas": [],
        "barras": [
            {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"},
            {"id": "Barra2", "nodo_i": "N2", "nodo_j": "N3"}
        ],
        "apoyos": [
            {"nodo": "N1", "tipo": "fijo", "angulo": 0.0},
            {"nodo": "N2", "tipo": "movil", "angulo": 0.0},
            {"nodo": "N3", "tipo": "movil", "angulo": 0.0}
        ],
        "cargas_distribuidas": [
            {"barra": "Barra1", "w_inicio": 12.0, "w_fin": 12.0, "x_inicio": 0.0, "x_fin": 4.0, "direccion": "-Y"},
            {"barra": "Barra2", "w_inicio": 12.0, "w_fin": 12.0, "x_inicio": 0.0, "x_fin": 4.0, "direccion": "-Y"}
        ],
        "cargas_puntuales": [],
        "momentos_concentrados": []
    },
    "7. Pórtico Biarticulado con Carga Lateral y Gravitatoria": {
        "unidad_fuerza": "kN",
        "unidad_longitud": "m",
        "nodos": [
            {"id": "N1", "x": 0.0, "y": 0.0},
            {"id": "N2", "x": 0.0, "y": 3.0},
            {"id": "N3", "x": 5.0, "y": 3.0},
            {"id": "N4", "x": 5.0, "y": 0.0}
        ],
        "rotulas": [],
        "barras": [
            {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"},
            {"id": "Barra2", "nodo_i": "N2", "nodo_j": "N3"},
            {"id": "Barra3", "nodo_i": "N3", "nodo_j": "N4"}
        ],
        "apoyos": [
            {"nodo": "N1", "tipo": "fijo", "angulo": 0.0},
            {"nodo": "N4", "tipo": "fijo", "angulo": 0.0}
        ],
        "cargas_distribuidas": [
            {"barra": "Barra2", "w_inicio": 10.0, "w_fin": 10.0, "x_inicio": 0.0, "x_fin": 5.0, "direccion": "-Y"}
        ],
        "cargas_puntuales": [
            {"nodo": "N2", "magnitud": 15.0, "direccion": "+X", "angulo": 0.0}
        ],
        "momentos_concentrados": []
    }
}

# ==========================================================
# CATÁLOGO DE PERFILES COMERCIALES
# ==========================================================
CATALOGO_ARG = {
    "IPN (Laminado I)": {
        "IPN 80":  {"A": 7.58, "Iz": 77.8, "Wz": 19.5, "Iy": 6.29, "Wy": 3.00, "kg_m": 5.94},
        "IPN 100": {"A": 10.6, "Iz": 171.0, "Wz": 34.2, "Iy": 12.2, "Wy": 4.88, "kg_m": 8.34},
        "IPN 120": {"A": 14.2, "Iz": 328.0, "Wz": 54.7, "Iy": 21.5, "Wy": 7.41, "kg_m": 11.1},
        "IPN 140": {"A": 18.2, "Iz": 573.0, "Wz": 81.9, "Iy": 35.2, "Wy": 10.7, "kg_m": 14.3},
        "IPN 160": {"A": 22.8, "Iz": 935.0, "Wz": 117.0, "Iy": 54.7, "Wy": 14.8, "kg_m": 17.9},
        "IPN 180": {"A": 27.9, "Iz": 1450.0, "Wz": 161.0, "Iy": 81.3, "Wy": 19.8, "kg_m": 21.9},
        "IPN 200": {"A": 33.4, "Iz": 2140.0, "Wz": 214.0, "Iy": 117.0, "Wy": 26.0, "kg_m": 26.2},
        "IPN 220": {"A": 39.5, "Iz": 3060.0, "Wz": 278.0, "Iy": 162.0, "Wy": 33.1, "kg_m": 31.1},
        "IPN 240": {"A": 46.1, "Iz": 4250.0, "Wz": 354.0, "Iy": 221.0, "Wy": 41.7, "kg_m": 36.2}
    },
    "IPE (Laminado Alas Paralelas)": {
        "IPE 80":  {"A": 7.64, "Iz": 80.1, "Wz": 20.0, "Iy": 8.49, "Wy": 3.69, "kg_m": 6.0},
        "IPE 100": {"A": 10.3, "Iz": 171.0, "Wz": 34.2, "Iy": 15.9, "Wy": 5.79, "kg_m": 8.1},
        "IPE 120": {"A": 13.2, "Iz": 318.0, "Wz": 53.0, "Iy": 27.7, "Wy": 8.65, "kg_m": 10.4},
        "IPE 140": {"A": 16.4, "Iz": 541.0, "Wz": 77.3, "Iy": 44.9, "Wy": 12.3, "kg_m": 12.9},
        "IPE 160": {"A": 20.1, "Iz": 869.0, "Wz": 109.0, "Iy": 68.3, "Wy": 16.7, "kg_m": 15.8},
        "IPE 180": {"A": 23.9, "Iz": 1317.0, "Wz": 146.0, "Iy": 101.0, "Wy": 22.2, "kg_m": 18.8},
        "IPE 200": {"A": 28.5, "Iz": 1943.0, "Wz": 194.0, "Iy": 142.0, "Wy": 28.5, "kg_m": 22.4}
    },
    "UPN (Laminado U)": {
        "UPN 80":  {"A": 11.0, "Iz": 106.0, "Wz": 26.5, "Iy": 19.4, "Wy": 6.36, "kg_m": 8.64},
        "UPN 100": {"A": 13.5, "Iz": 206.0, "Wz": 41.2, "Iy": 29.3, "Wy": 8.49, "kg_m": 10.6},
        "UPN 120": {"A": 17.0, "Iz": 364.0, "Wz": 60.7, "Iy": 43.2, "Wy": 11.1, "kg_m": 13.4},
        "UPN 140": {"A": 20.4, "Iz": 605.0, "Wz": 86.4, "Iy": 62.7, "Wy": 14.8, "kg_m": 16.0},
        "UPN 160": {"A": 24.0, "Iz": 925.0, "Wz": 116.0, "Iy": 85.3, "Wy": 18.3, "kg_m": 18.8},
        "UPN 180": {"A": 28.0, "Iz": 1350.0, "Wz": 150.0, "Iy": 114.0, "Wy": 22.4, "kg_m": 22.0},
        "UPN 200": {"A": 32.2, "Iz": 1910.0, "Wz": 191.0, "Iy": 148.0, "Wy": 27.0, "kg_m": 25.3}
    },
    "Perfil C Conformado (PGC)": {
        "C 80x40x15x1.6":  {"A": 2.77, "Iz": 28.7, "Wz": 7.18, "Iy": 7.4, "Wy": 2.8, "kg_m": 2.17},
        "C 100x50x15x2.0": {"A": 4.33, "Iz": 71.3, "Wz": 14.3, "Iy": 15.9, "Wy": 4.8, "kg_m": 3.40},
        "C 120x50x15x2.0": {"A": 4.73, "Iz": 109.8, "Wz": 18.3, "Iy": 17.5, "Wy": 5.1, "kg_m": 3.71},
        "C 140x60x20x2.5": {"A": 7.00, "Iz": 217.5, "Wz": 31.1, "Iy": 34.3, "Wy": 8.3, "kg_m": 5.50},
        "C 160x60x20x2.5": {"A": 7.50, "Iz": 295.6, "Wz": 36.9, "Iy": 37.8, "Wy": 8.8, "kg_m": 5.89},
        "C 200x70x20x3.2": {"A": 11.52, "Iz": 698.4, "Wz": 69.8, "Iy": 74.5, "Wy": 14.9, "kg_m": 9.04}
    },
    "Tubos Estructurales": {
        "Tubo 50x50x2.0":    {"A": 3.66, "Iz": 13.5, "Wz": 5.41, "Iy": 13.5, "Wy": 5.41, "kg_m": 2.87},
        "Tubo 60x60x2.5":    {"A": 5.51, "Iz": 28.9, "Wz": 9.64, "Iy": 28.9, "Wy": 9.64, "kg_m": 4.33},
        "Tubo 80x80x3.2":    {"A": 9.38, "Iz": 90.5, "Wz": 22.6, "Iy": 90.5, "Wy": 22.6, "kg_m": 7.36},
        "Tubo 100x100x3.2":  {"A": 11.94, "Iz": 186.0, "Wz": 37.2, "Iy": 186.0, "Wy": 37.2, "kg_m": 9.37},
        "Tubo 100x50x3.2":   {"A": 8.74, "Iz": 120.4, "Wz": 24.1, "Iy": 40.5, "Wy": 16.2, "kg_m": 6.86},
        "Tubo 120x60x4.0":   {"A": 13.20, "Iz": 265.8, "Wz": 44.3, "Iy": 89.5, "Wy": 29.8, "kg_m": 10.36},
        "Tubo 150x100x4.75": {"A": 21.84, "Iz": 712.5, "Wz": 95.0, "Iy": 378.0, "Wy": 75.6, "kg_m": 17.15}
    }
}

# ==========================================================
# GENERADOR GRÁFICO DEL PERFIL
# ==========================================================
def rotar_coordenadas(xs, ys, ang_deg):
    rad = math.radians(ang_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [x * cos_a - y * sin_a for x, y in zip(xs, ys)], [x * sin_a + y * cos_a for x, y in zip(xs, ys)]

def dibujar_esquema_perfil(tipo_familia, nombre_perfil, orientacion_deg):
    fig = go.Figure()
    
    if "IPN" in tipo_familia or "IPE" in tipo_familia:
        x_pts = [-3.5, 3.5, 3.5, 0.7, 0.7, 3.5, 3.5, -3.5, -3.5, -0.7, -0.7, -3.5, -3.5]
        y_pts = [5.0, 5.0, 3.8, 3.8, -3.8, -3.8, -5.0, -5.0, -3.8, -3.8, 3.8, 3.8, 5.0]
        xr, yr = rotar_coordenadas(x_pts, y_pts, orientacion_deg)
        fig.add_trace(go.Scatter(x=xr, y=yr, fill="toself", fillcolor="rgba(56, 189, 248, 0.35)", line=dict(color="#38bdf8", width=2), hoverinfo="skip"))
    elif "UPN" in tipo_familia:
        x_pts = [-2.5, 3.0, 3.0, -1.1, -1.1, 3.0, 3.0, -2.5, -2.5]
        y_pts = [5.0, 5.0, 3.7, 3.7, -3.7, -3.7, -5.0, -5.0, 5.0]
        xr, yr = rotar_coordenadas(x_pts, y_pts, orientacion_deg)
        fig.add_trace(go.Scatter(x=xr, y=yr, fill="toself", fillcolor="rgba(56, 189, 248, 0.35)", line=dict(color="#38bdf8", width=2), hoverinfo="skip"))
    elif "Perfil C" in tipo_familia:
        x_pts = [-2.5, 2.8, 2.8, 2.0, 2.0, 2.2, -1.7, -1.7, 2.2, 2.0, 2.0, 2.8, 2.8, -2.5, -2.5]
        y_pts = [5.0, 5.0, 3.5, 3.5, 3.8, 4.3, 4.3, -4.3, -4.3, -3.8, -3.5, -3.5, -5.0, -5.0, 5.0]
        xr, yr = rotar_coordenadas(x_pts, y_pts, orientacion_deg)
        fig.add_trace(go.Scatter(x=xr, y=yr, fill="toself", fillcolor="rgba(56, 189, 248, 0.35)", line=dict(color="#38bdf8", width=2), hoverinfo="skip"))
    elif "Tubos" in tipo_familia:
        x_out = [-3.5, 3.5, 3.5, -3.5, -3.5]
        y_out = [5.0, 5.0, -5.0, -5.0, 5.0]
        x_in = [-2.2, 2.2, 2.2, -2.2, -2.2]
        y_in = [3.7, 3.7, -3.7, -3.7, 3.7]
        x_out_r, y_out_r = rotar_coordenadas(x_out, y_out, orientacion_deg)
        x_in_r, y_in_r = rotar_coordenadas(x_in, y_in, orientacion_deg)
        fig.add_trace(go.Scatter(x=x_out_r, y=y_out_r, fill="toself", fillcolor="rgba(56, 189, 248, 0.35)", line=dict(color="#38bdf8", width=2), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_in_r, y=y_in_r, fill="toself", fillcolor="#0f172a", line=dict(color="#38bdf8", width=2), hoverinfo="skip"))
    else:
        x_pts = [-3.0, 3.0, 3.0, -3.0, -3.0]
        y_pts = [4.5, 4.5, -4.5, -4.5, 4.5]
        xr, yr = rotar_coordenadas(x_pts, y_pts, orientacion_deg)
        fig.add_trace(go.Scatter(x=xr, y=yr, fill="toself", fillcolor="rgba(56, 189, 248, 0.35)", line=dict(color="#38bdf8", width=2), hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=[-5.5, 5.5], y=[0, 0], mode="lines", line=dict(color="#64748b", width=1, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 0], y=[-6.5, 6.5], mode="lines", line=dict(color="#64748b", width=1, dash="dash"), hoverinfo="skip"))

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        xaxis=dict(range=[-7, 7], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-7, 7], visible=False),
        margin=dict(l=5, r=5, t=5, b=5),
        height=190, showlegend=False
    )
    return fig

# ==========================================================
# GESTIÓN DEL ESTADO GLOBAL
# ==========================================================
if "version_estructura" not in st.session_state:
    st.session_state.version_estructura = 0

if "datos_estructura" not in st.session_state or st.session_state.datos_estructura is None:
    st.session_state.datos_estructura = json.loads(json.dumps(PLANTILLAS["1. Viga Simplemente Apoyada - Carga Puntual al Centro"]))

v_act = st.session_state.version_estructura

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    api_key_input = st.text_input(
        "Google AI Studio API Key",
        type="password",
        placeholder="AIzaSy...",
        key="input_api_key",
        help="Obtené tu clave gratuita en https://aistudio.google.com"
    )
    
    modelo_seleccionado = st.selectbox(
        "Modelo de Visión",
        ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"],
        key="select_modelo"
    )
    
    st.markdown("---")
    st.markdown("#### Seleccionar perfil")
    tipo_perfil = st.selectbox("Tipo de perfil", list(CATALOGO_ARG.keys()) + ["Personalizado"])
    
    if tipo_perfil != "Personalizado":
        designacion_sel = st.selectbox("Designación", list(CATALOGO_ARG[tipo_perfil].keys()))
        prop = CATALOGO_ARG[tipo_perfil][designacion_sel]
        
        orientacion_deg = st.selectbox(
            "Orientación del perfil:",
            [0, 90, 180, 270],
            format_func=lambda deg: f"{deg}° " + ("(Eje Z)" if deg in [0, 180] else "(Eje Y)")
        )
        
        if orientacion_deg in [90, 270]:
            iz_activa = prop["Iy"]
            wz_activa = prop["Wy"]
            eje_str = "Eje Y"
        else:
            iz_activa = prop["Iz"]
            wz_activa = prop["Wz"]
            eje_str = "Eje Z"
            
        area_sec_val = prop["A"]
        iz_sec_val = iz_activa
        wz_sec_val = wz_activa
        kg_m_val = prop["kg_m"]
        st.caption(f"ℹ️ **Activo ({eje_str}):** Iz = {iz_sec_val} cm⁴ | Wz = {wz_sec_val} cm³ | A = {area_sec_val} cm²")
    else:
        designacion_sel = "Manual"
        orientacion_deg = st.selectbox("Orientación del perfil:", [0, 90, 180, 270])
        area_sec_val = 100.0
        iz_sec_val = 1000.0
        wz_sec_val = 200.0

    with st.container(border=True):
        st.markdown(f"**Perfil seleccionado** ({orientacion_deg}°)")
        st.plotly_chart(dibujar_esquema_perfil(tipo_perfil, designacion_sel, orientacion_deg), use_container_width=True, config={"displayModeBar": False})

    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        e_mod = st.number_input("Módulo E (GPa)", value=200.0, step=10.0, min_value=1.0) * 1e9
    with col_mat2:
        area_sec = st.number_input("Área A (cm²)", value=float(area_sec_val), step=1.0, min_value=0.1) * 1e-4
    
    iz_sec = st.number_input("Inercia Iz (cm⁴)", value=float(iz_sec_val), step=10.0, min_value=0.1) * 1e-8
    wz_sec = float(wz_sec_val)

# Banner Principal
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏗️ Nodal · Sistema de Análisis Estructural</div>
    <p class="hero-subtitle">
        Nodal es un entorno web para el análisis matricial de estructuras planas en 2D. Permite digitalizar bocetos a mano mediante inteligencia artificial o modelar desde cero, generando diagramas interactivos de esfuerzos internos (M, Q, N), deformadas y reacciones con precisión técnica de ingeniería.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# GUÍA DE USO RÁPIDA
# ==========================================================
with st.expander("📖 ¿Cómo usar Nodal? · Guía rápida de uso", expanded=False):
    col_guia1, col_guia2 = st.columns(2)
    with col_guia1:
        st.markdown("""
        **1. Cargar o crear la estructura**
        * **Biblioteca básica:** Elegí un caso típico de libro y cargalo directamente al editor.
        * **Desde cero:** Seleccioná las unidades de fuerza ($kN, kg, t$) e iniciá un lienzo limpio.
        * **Croquis con IA:** Subí una foto o capturá con la cámara un dibujo en papel. *(Requiere pegar tu API Key gratuita de Google AI Studio en la barra lateral)*.

        **2. Asignar perfil y orientación**
        * En el panel lateral izquierdo seleccioná perfiles comerciales de Argentina (IPN, IPE, UPN, Perfil C o Tubos) o valores manuales.
        * Cambiá la orientación ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) para alternar automáticamente entre el eje ($Z$) y el eje ($Y$).
        """)
    with col_guia2:
        st.markdown("""
        **3. Ajustar en el Inspector de Propiedades**
        * **Nodos y Apoyos:** Modificá coordenadas ($X, Y$) y configurá apoyos orientables en cualquier ángulo sexagesimal (medido desde el eje $+X$ en sentido antihorario).
        * **Barras:** Conectá los nodos asegurando la continuidad del modelo.
        * **Cargas y Momentos:** Agregá solicitaciones distribuidas o puntuales indicando posición local, ángulo y sentido de la flecha.

        **4. Calcular y exportar diagramas**
        * Presioná **🖩 Calcular** para resolver la matriz de rigidez y ver los diagramas de $M$, $Q$, $N$, la elástica deformada y los vectores reactivos.
        * Los momentos flectores ($M$) se grafican del lado de las fibras traccionadas (convención FTool/ingeniería).
        * Podés descargar cualquier diagrama en formato PNG pasando el cursor por encima del gráfico y haciendo clic en el ícono de la cámara 📷.
        """)

# ==========================================================
# PROMPT Y FUNCIÓN DE EXTRACCIÓN CON IA (GEMINI)
# ==========================================================
instruccion_ia = """
Sos un Ingeniero Estructural Senior y experto en análisis matricial 2D, cinemática de mecanismos y visión computacional.
Tu misión es interpretar y digitalizar el croquis o plano estructural provisto, convirtiéndolo en un modelo topológico perfecto en el plano cartesiano global XY (con Z saliente hacia el observador).

### 1. REGLA FUNDAMENTAL DE BARRAS CONTINUAS Y CARGAS (PROHIBIDO CREAR NODOS INNECESARIOS)
- NUNCA, BAJO NINGUNA CIRCUNSTANCIA, crees nodos adicionales para colocar cargas puntuales o momentos en medio de una barra recta continua.
- Si una viga recta va de un apoyo a otro (ej. de x=0 a x=5m), DEBE SER UNA SOLA Y ÚNICA BARRA continua (ej. 'Barra1' de N1 a N2 de longitud 5.0m).
- Las cargas puntuales intermedias deben colocarse sobre la barra indicando 'barra': 'Barra1' y 'posicion_x': 3.0 (distancia métrica local desde nodo_i).
- SOLO se crean nodos en:
  1. Apoyos exteriores.
  2. Quiebres o cambios de dirección de la barra.
  3. Encuentros / intersecciones con otras barras (ej. columnas o ménsulas).
  4. Extremos libres de voladizos.
  5. Rótulas internas articuladas (M = 0).

### 2. REGLA ESTRICTA DE UNIDADES Y MAGNITUDES (NO NORMALIZAR)
- Conserva EXACTAMENTE la unidad de fuerza escrita por el usuario en el dibujo ('kg', 'kgf', 't', 'kN', 'N').
- PROHIBIDO convertir unidades (ej. NO conviertas 50 kg a 0.05 t ni a 0.1 t).
- Si el croquis dice '50 kg', 'unidad_fuerza' DEBE SER 'kg' (o 'kgf') y 'magnitud': 50.0. Conserva el valor numérico exacto.
- Longitudes: siempre en metros ('m').

### 3. CONDICIONES DE VÍNCULO Y APOYOS ORIENTABLES (ÁNGULO EN GRADOS DESDE +X ANTIHORARIO)
- Clasificación:
  * 'fijo': Apoyo articulado fijo (restringe traslación en X e Y).
  * 'movil': Apoyo móvil / rodillo (restringe traslación perpendicular al plano de deslizamiento).
  * 'empotrado': Empotramiento perfecto.
- 'angulo': Ángulo sexagesimal en grados del plano de deslizamiento / base medido en sentido antihorario desde el semieje +X:
  * 0.0°: Base horizontal sobre el suelo (+X). En apoyos móviles restringe Y (desliza en X).
  * 90.0°: Base vertical sobre pared a la derecha (+Y). En apoyos móviles restringe X (desliza en Y).
  * alpha°: Cualquier inclinación oblicua.

### 4. CARGAS PUNTUALES INCLINADAS (ÁNGULO TAL CUAL EL DIBUJO)
- Si la fuerza está inclinada con un ángulo escrito en el dibujo (ej. '45°'):
  * 'angulo': Coloca el número exacto escrito en el croquis (ej. 45.0).
  * 'sentido': Uno de los 4 cuadrantes hacia donde apunta la flecha:
    - 'abajo_izquierda': Apunta hacia abajo y hacia la izquierda (↙).
    - 'abajo_derecha': Apunta hacia abajo y hacia la derecha (↘).
    - 'arriba_derecha': Apunta hacia arriba y hacia la derecha (↗).
    - 'arriba_izquierda': Apunta hacia arriba y hacia la izquierda (↖).
- Si la fuerza es vertical u horizontal pura:
  * 'direccion': '-Y', '+Y', '-X' o '+X' (con 'angulo': null).

### 5. CARGAS DISTRIBUIDAS (q)
- 'barra': ID de la barra sobre la que actúa.
- 'w_inicio' y 'w_fin': Magnitud en la unidad indicada.
- 'direccion': '-Y' (gravitatoria vertical), '+Y', '-X', '+X', 'perpendicular_adentro', 'perpendicular_afuera'.

Devolvé EXCLUSIVAMENTE un bloque JSON válido:
{
  "unidad_fuerza": "kg",
  "unidad_longitud": "m",
  "nodos": [
    {"id": "N1", "x": 0.0, "y": 0.0},
    {"id": "N2", "x": 5.0, "y": 0.0}
  ],
  "rotulas": [],
  "barras": [
    {"id": "Barra1", "nodo_i": "N1", "nodo_j": "N2"}
  ],
  "apoyos": [
    {"nodo": "N1", "tipo": "fijo", "angulo": 0.0},
    {"nodo": "N2", "tipo": "movil", "angulo": 90.0}
  ],
  "cargas_distribuidas": [],
  "cargas_puntuales": [
    {
      "barra": "Barra1",
      "posicion_x": 3.0,
      "magnitud": 50.0,
      "angulo": 45.0,
      "sentido": "abajo_izquierda"
    }
  ],
  "momentos_concentrados": []
}
"""

def extraer_con_fallback(client, modelo_nombre, foto):
    config = types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    modelos = [modelo_nombre]
    for alt in ['gemini-2.5-flash', 'gemini-3.5-flash-lite']:
        if alt not in modelos:
            modelos.append(alt)

    ultimo_error = None
    foto_rgb = foto.convert("RGB")

    for mod in modelos:
        try:
            res = client.models.generate_content(model=mod, contents=[instruccion_ia, foto_rgb], config=config)
            match = re.search(r"\{[\s\S]*\}", res.text)
            if match:
                return json.loads(match.group(0)), None
            else:
                ultimo_error = f"El modelo {mod} no devolvió un JSON legible."
        except Exception as e:
            ultimo_error = f"Error en {mod}: {str(e)}"
            time.sleep(1.5)

    return None, ultimo_error

# ==========================================================
# CÁLCULO VECTORIAL DE CARGA PUNTUAL
# ==========================================================
def vector_fuerza_puntual(cp):
    mag = abs(float(cp.get("magnitud", 0.0)))
    
    if "angulo" in cp and cp.get("angulo") is not None:
        ang = float(cp.get("angulo", 45.0))
        if ang > 90.0:
            rad = math.radians(ang)
            return mag * math.cos(rad), mag * math.sin(rad)
            
        rad = math.radians(ang)
        sent = str(cp.get("sentido", "abajo_izquierda")).lower()
        if "izquierda" in sent and "abajo" in sent:
            return -mag * math.cos(rad), -mag * math.sin(rad)
        elif "derecha" in sent and "abajo" in sent:
            return mag * math.cos(rad), -mag * math.sin(rad)
        elif "derecha" in sent and "arriba" in sent:
            return mag * math.cos(rad), mag * math.sin(rad)
        elif "izquierda" in sent and "arriba" in sent:
            return -mag * math.cos(rad), mag * math.sin(rad)
        else:
            return -mag * math.cos(rad), -mag * math.sin(rad)
    else:
        dir_s = str(cp.get("direccion", "-Y")).upper()
        if "+X" in dir_s: return mag, 0.0
        elif "-X" in dir_s: return -mag, 0.0
        elif "+Y" in dir_s: return 0.0, mag
        else: return 0.0, -mag

# ==========================================================
# RUTINA DE DIBUJO DE VÍNCULOS ORIENTABLES
# ==========================================================
def dibujar_vinculo_orientable(fig, x0, y0, tipo, angulo_deg, L_max):
    s = 0.05 * L_max
    rad = math.radians(angulo_deg)
    
    v_tan_x = math.cos(rad)
    v_tan_y = math.sin(rad)
    v_ground_x = math.sin(rad)
    v_ground_y = -math.cos(rad)
    
    tipo_l = tipo.lower()
    
    if "fijo" in tipo_l:
        p1_x = x0 + s * v_ground_x - 0.7 * s * v_tan_x
        p1_y = y0 + s * v_ground_y - 0.7 * s * v_tan_y
        p2_x = x0 + s * v_ground_x + 0.7 * s * v_tan_x
        p2_y = y0 + s * v_ground_y + 0.7 * s * v_tan_y
        
        fig.add_trace(go.Scatter(
            x=[x0, p1_x, p2_x, x0], y=[y0, p1_y, p2_y, y0],
            fill="toself", fillcolor="#f59e0b",
            line=dict(color="#ffffff", width=1.5), hoverinfo="skip", showlegend=False
        ))
        
        g1_x = p1_x - 0.3 * s * v_tan_x
        g1_y = p1_y - 0.3 * s * v_tan_y
        g2_x = p2_x + 0.3 * s * v_tan_x
        g2_y = p2_y + 0.3 * s * v_tan_y
        fig.add_trace(go.Scatter(x=[g1_x, g2_x], y=[g1_y, g2_y], mode="lines", line=dict(color="#f59e0b", width=2.5), hoverinfo="skip", showlegend=False))

    elif "movil" in tipo_l or "móvil" in tipo_l:
        p1_x = x0 + 0.75 * s * v_ground_x - 0.65 * s * v_tan_x
        p1_y = y0 + 0.75 * s * v_ground_y - 0.65 * s * v_tan_y
        p2_x = x0 + 0.75 * s * v_ground_x + 0.65 * s * v_tan_x
        p2_y = y0 + 0.75 * s * v_ground_y + 0.65 * s * v_tan_y
        
        fig.add_trace(go.Scatter(
            x=[x0, p1_x, p2_x, x0], y=[y0, p1_y, p2_y, y0],
            fill="toself", fillcolor="#38bdf8",
            line=dict(color="#ffffff", width=1.5), hoverinfo="skip", showlegend=False
        ))
        
        c1_x = x0 + 1.05 * s * v_ground_x - 0.35 * s * v_tan_x
        c1_y = y0 + 1.05 * s * v_ground_y - 0.35 * s * v_tan_y
        c2_x = x0 + 1.05 * s * v_ground_x + 0.35 * s * v_tan_x
        c2_y = y0 + 1.05 * s * v_ground_y + 0.35 * s * v_tan_y
        
        fig.add_trace(go.Scatter(
            x=[c1_x, c2_x], y=[c1_y, c2_y],
            mode="markers", marker=dict(size=6, color="#ffffff", line=dict(color="#38bdf8", width=1.5)),
            hoverinfo="skip", showlegend=False
        ))
        
        g1_x = x0 + 1.35 * s * v_ground_x - 0.9 * s * v_tan_x
        g1_y = y0 + 1.35 * s * v_ground_y - 0.9 * s * v_tan_y
        g2_x = x0 + 1.35 * s * v_ground_x + 0.9 * s * v_tan_x
        g2_y = y0 + 1.35 * s * v_ground_y + 0.9 * s * v_tan_y
        fig.add_trace(go.Scatter(x=[g1_x, g2_x], y=[g1_y, g2_y], mode="lines", line=dict(color="#38bdf8", width=2.5), hoverinfo="skip", showlegend=False))

    elif "empotrado" in tipo_l:
        w1_x = x0 - 0.8 * s * v_tan_x
        w1_y = y0 - 0.8 * s * v_tan_y
        w2_x = x0 + 0.8 * s * v_tan_x
        w2_y = y0 + 0.8 * s * v_tan_y
        
        fig.add_trace(go.Scatter(x=[w1_x, w2_x], y=[w1_y, w2_y], mode="lines", line=dict(color="#ec4899", width=4), hoverinfo="skip", showlegend=False))
        
        for t_step in np.linspace(-0.65, 0.65, 4):
            bx = x0 + t_step * s * v_tan_x
            by = y0 + t_step * s * v_tan_y
            ex = bx + 0.45 * s * v_ground_x + 0.25 * s * v_tan_x
            ey = by + 0.45 * s * v_ground_y + 0.25 * s * v_tan_y
            fig.add_trace(go.Scatter(x=[bx, ex], y=[by, ey], mode="lines", line=dict(color="#ec4899", width=1.5), hoverinfo="skip", showlegend=False))

# ==========================================================
# RUTINA DE RENDERIZADO DE CARGAS EN PLOTLY
# ==========================================================
def agregar_cargas_graficas(fig, datos, nodos_dict, L_max):
    un_f = datos.get("unidad_fuerza", "kN")
    un_m = f"{un_f}·m"
    barras_dict = {b["id"]: b for b in datos.get("barras", [])}

    for cd in datos.get("cargas_distribuidas", []):
        b_id = cd.get("barra")
        if b_id in barras_dict:
            b = barras_dict[b_id]
            if b["nodo_i"] in nodos_dict and b["nodo_j"] in nodos_dict:
                xi, yi = nodos_dict[b["nodo_i"]]
                xj, yj = nodos_dict[b["nodo_j"]]
                L_b = math.hypot(xj - xi, yj - yi)
                if L_b < 1e-4: continue
                cos_t, sin_t = (xj - xi) / L_b, (yj - yi) / L_b
                nx, ny = -sin_t, cos_t

                dir_s = str(cd.get("direccion", "perpendicular_adentro")).lower()
                if "adentro" in dir_s or "perpendicular_adentro" in dir_s:
                    ux, uy = -nx, -ny
                elif "afuera" in dir_s or "perpendicular_afuera" in dir_s:
                    ux, uy = nx, ny
                elif "+x" in dir_s: ux, uy = 1.0, 0.0
                elif "-x" in dir_s: ux, uy = -1.0, 0.0
                elif "+y" in dir_s: ux, uy = 0.0, 1.0
                elif "-y" in dir_s: ux, uy = 0.0, -1.0
                else: ux, uy = -nx, -ny

                x1 = float(cd.get("x_inicio", 0.0))
                x2 = float(cd.get("x_fin", L_b))
                w_mag = float(cd.get("w_inicio", cd.get("w_fin", 0.0)))
                h_block = 0.14 * L_max

                pb1 = (xi + x1 * cos_t, yi + x1 * sin_t)
                pb2 = (xi + x2 * cos_t, yi + x2 * sin_t)
                pt1 = (pb1[0] - ux * h_block, pb1[1] - uy * h_block)
                pt2 = (pb2[0] - ux * h_block, pb2[1] - uy * h_block)

                fig.add_trace(go.Scatter(
                    x=[pb1[0], pt1[0], pt2[0], pb2[0], pb1[0]],
                    y=[pb1[1], pt1[1], pt2[1], pb2[1], pb1[1]],
                    fill='toself', fillcolor='rgba(56, 189, 248, 0.22)',
                    line=dict(color='#38bdf8', width=2),
                    hoverinfo='skip', showlegend=False
                ))

                for s in np.linspace(0.15, 0.85, 4):
                    px = pb1[0] + s * (pb2[0] - pb1[0])
                    py = pb1[1] + s * (pb2[1] - pb1[1])
                    tx = px - ux * h_block
                    ty = py - uy * h_block
                    fig.add_annotation(
                        x=px, y=py, ax=tx, ay=ty,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#38bdf8"
                    )

                fig.add_annotation(
                    x=(pt1[0] + pt2[0]) / 2 - ux * 0.05 * L_max,
                    y=(pt1[1] + pt2[1]) / 2 - uy * 0.05 * L_max,
                    text=f"<b>q = {w_mag:.1f} {un_f}/m</b>",
                    showarrow=False, font=dict(color="#38bdf8", size=11),
                    bgcolor="#0f172a", bordercolor="#38bdf8", borderwidth=1, borderpad=3
                )

    for cp in datos.get("cargas_puntuales", []):
        mag = abs(float(cp.get("magnitud", 0.0)))
        px, py = None, None
        
        fx, fy = vector_fuerza_puntual(cp)
        norm_f = math.hypot(fx, fy)
        if norm_f > 1e-6:
            ux, uy = fx / norm_f, fy / norm_f
        else:
            ux, uy = 0.0, -1.0
        
        if "nodo" in cp and cp["nodo"] in nodos_dict:
            px, py = nodos_dict[cp["nodo"]]
        elif "barra" in cp and cp["barra"] in barras_dict:
            b = barras_dict[cp["barra"]]
            if b["nodo_i"] in nodos_dict and b["nodo_j"] in nodos_dict:
                xi, yi = nodos_dict[b["nodo_i"]]
                xj, yj = nodos_dict[b["nodo_j"]]
                L_b = math.hypot(xj - xi, yj - yi)
                if L_b >= 1e-4:
                    cos_t, sin_t = (xj - xi) / L_b, (yj - yi) / L_b
                    pos = float(cp.get("posicion_x", 0.0))
                    px, py = xi + pos * cos_t, yi + pos * sin_t

        if px is not None and py is not None:
            l_arr = 0.18 * L_max
            fig.add_annotation(
                x=px, y=py, ax=px - ux * l_arr, ay=py - uy * l_arr,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=2.5, arrowcolor="#f43f5e",
                text=f"<b>P = {mag:.1f} {un_f}</b>",
                font=dict(color="#f43f5e", size=11),
                bgcolor="#0f172a", bordercolor="#f43f5e", borderwidth=1, borderpad=3
            )

    for mc in datos.get("momentos_concentrados", []):
        mag = abs(float(mc.get("magnitud", 0.0)))
        st_txt = mc.get("sentido", "horario").lower()
        px, py = None, None
        
        if "nodo" in mc and mc["nodo"] in nodos_dict:
            px, py = nodos_dict[mc["nodo"]]
        elif "barra" in mc and mc["barra"] in barras_dict:
            b = barras_dict[mc["barra"]]
            if b["nodo_i"] in nodos_dict and b["nodo_j"] in nodos_dict:
                xi, yi = nodos_dict[b["nodo_i"]]
                xj, yj = nodos_dict[b["nodo_j"]]
                L_b = math.hypot(xj - xi, yj - yi)
                if L_b >= 1e-4:
                    pos = float(mc.get("posicion_x", 0.0))
                    px = xi + pos * (xj - xi) / L_b
                    py = yi + pos * (yj - yi) / L_b

        if px is not None and py is not None:
            r_arc = 0.11 * L_max
            t_vals = np.linspace(5*np.pi/6, np.pi/6, 20) if st_txt == "horario" else np.linspace(np.pi/6, 5*np.pi/6, 20)
            x_arc = px + r_arc * np.cos(t_vals)
            y_arc = py + r_arc * np.sin(t_vals)
            
            fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', line=dict(color='#a855f7', width=2.5), hoverinfo='skip', showlegend=False))
            fig.add_annotation(
                x=x_arc[-1], y=y_arc[-1], ax=x_arc[-2], ay=y_arc[-2],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor="#a855f7"
            )
            fig.add_annotation(
                x=px, y=py + r_arc + 0.05 * L_max,
                text=f"<b>M₀ = {mag:.1f} {un_m}</b>",
                showarrow=False, font=dict(color="#a855f7", size=11),
                bgcolor="#0f172a", bordercolor="#a855f7", borderwidth=1, borderpad=3
            )

# ==========================================================
# GENERADOR DEL ESQUEMA EN VIVO
# ==========================================================
def crear_esquema_modelo(datos):
    fig = go.Figure()
    nodos = datos.get("nodos", [])
    nodos_dict = {n["id"]: (float(n["x"]), float(n["y"])) for n in nodos}
    
    if not nodos_dict:
        fig.add_annotation(
            text="<b>Lienzo Vacío</b><br>Agregá nodos en el Inspector para comenzar a modelar.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#64748b", size=14)
        )
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            xaxis=dict(range=[-1, 6], showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155"),
            yaxis=dict(range=[-1, 4], showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155"),
            margin=dict(l=10, r=10, t=10, b=10), height=430
        )
        return fig

    xs = [x for x, y in nodos_dict.values()]
    ys = [y for x, y in nodos_dict.values()]
    L_max = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    pad = 0.35 * L_max
    fig.add_trace(go.Scatter(
        x=[min(xs) - pad, max(xs) + pad], y=[min(ys) - pad, max(ys) + pad],
        mode='markers', marker=dict(size=0.1, opacity=0), hoverinfo='skip', showlegend=False
    ))

    for b in datos.get("barras", []):
        if b["nodo_i"] in nodos_dict and b["nodo_j"] in nodos_dict:
            xi, yi = nodos_dict[b["nodo_i"]]
            xj, yj = nodos_dict[b["nodo_j"]]
            L_elem = math.hypot(xj - xi, yj - yi)
            fig.add_trace(go.Scatter(
                x=[xi, xj], y=[yi, yj], mode='lines+text',
                line=dict(color='#94a3b8', width=4),
                text=["", f"<b>{b['id']}</b> ({L_elem:.2f}m)"],
                textposition="top center", name=b["id"], hoverinfo='text',
                hovertext=f"<b>{b['id']}</b><br>L = {L_elem:.2f} m<br>Conecta {b['nodo_i']} → {b['nodo_j']}"
            ))

    rotulas = set(datos.get("rotulas", []))
    for n_id, (x, y) in nodos_dict.items():
        es_rotula = n_id in rotulas
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(
                size=14 if es_rotula else 12,
                color='#ffffff' if es_rotula else '#3b82f6',
                line=dict(color='#ffffff' if not es_rotula else '#ef4444', width=2)
            ),
            text=[f"<b>{n_id}</b>"], textposition="bottom right",
            hoverinfo='text', hovertext=f"<b>Nodo {n_id}</b>: ({x:.2f}, {y:.2f}) m" + (" <br>[Rótula]" if es_rotula else ""),
            showlegend=False
        ))

    for ap in datos.get("apoyos", []):
        if ap["nodo"] in nodos_dict:
            ax, ay = nodos_dict[ap["nodo"]]
            ang = float(ap.get("angulo", 0.0))
            dibujar_vinculo_orientable(fig, ax, ay, ap["tipo"], ang, L_max)

    agregar_cargas_graficas(fig, datos, nodos_dict, L_max)

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=430, showlegend=False
    )
    return fig

# ==========================================================
# SECCIÓN 1: INGESTA / SELECCIÓN
# ==========================================================
st.markdown('<div class="step-badge">1. Selección o Creación de Estructura</div>', unsafe_allow_html=True)

tab_prontuario, tab_crear_cero, tab_ia = st.tabs([
    "📚 Biblioteca de Estructuras Básicas", 
    "🆕 Crear Estructura desde Cero", 
    "🔍 Interpretar Croquis con IA"
])

with tab_prontuario:
    col_p1, col_p2 = st.columns([3, 1], gap="medium")
    with col_p1:
        caso_elegido = st.selectbox("Seleccionar caso canónico:", list(PLANTILLAS.keys()), key="select_prontuario")
    with col_p2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📥 Cargar al Editor", type="primary", use_container_width=True):
            st.session_state.datos_estructura = json.loads(json.dumps(PLANTILLAS[caso_elegido]))
            st.session_state.version_estructura += 1
            st.session_state.pop("modelo_calculado", None)
            st.rerun()

with tab_crear_cero:
    col_cero1, col_cero2, col_cero3 = st.columns([1.5, 1.5, 1.5], gap="medium")
    with col_cero1:
        u_fuerza = st.selectbox("Unidad de Fuerza:", ["kN", "kg", "t", "N"], index=0, key="cero_uf")
    with col_cero2:
        u_long = st.selectbox("Unidad de Longitud:", ["m"], index=0, key="cero_ul")
    with col_cero3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("✨ Iniciar Lienzo en Blanco", type="primary", use_container_width=True, key="btn_iniciar_cero"):
            st.session_state.datos_estructura = {
                "unidad_fuerza": u_fuerza, "unidad_longitud": u_long,
                "nodos": [], "rotulas": [], "barras": [], "apoyos": [],
                "cargas_distribuidas": [], "cargas_puntuales": [], "momentos_concentrados": []
            }
            st.session_state.version_estructura += 1
            st.session_state.pop("modelo_calculado", None)
            st.rerun()

with tab_ia:
    col_ia1, col_ia2 = st.columns([1, 1], gap="medium")
    with col_ia1:
        fuente_img = st.radio("Fuente de imagen:", ["Subir Archivo JPG/PNG", "Cámara"], horizontal=True)
        archivo_subido = (
            st.file_uploader("Seleccionar croquis", type=["jpg", "jpeg", "png"], key="croquis_uploader")
            if "Subir" in fuente_img else st.camera_input("Capturar con cámara", key="croquis_camera")
        )
    with col_ia2:
        if archivo_subido and st.button("✨ Procesar con Gemini", type="primary", use_container_width=True):
            if not api_key_input:
                st.error("Ingresá tu API Key en la barra lateral.")
            else:
                with st.spinner("Interpretando croquis técnico..."):
                    img = Image.open(archivo_subido)
                    cliente = genai.Client(api_key=api_key_input)
                    res_ia, err_ia = extraer_con_fallback(cliente, modelo_seleccionado, img)
                    if res_ia:
                        st.session_state.datos_estructura = res_ia
                        st.session_state.version_estructura += 1
                        st.session_state.pop("modelo_calculado", None)
                        st.success("¡Estructura digitalizada con éxito!")
                        st.rerun()
                    else:
                        st.error(f"Falla en visión: {err_ia}")

# ==========================================================
# SECCIÓN 2: MODELO E INSPECTOR INTUITIVO (CLAVES DINÁMICAS)
# ==========================================================
st.markdown("---")
st.markdown('<div class="step-badge">2. Editor de Elementos e Inspector de Propiedades</div>', unsafe_allow_html=True)

col_canvas, col_inspector = st.columns([3, 2], gap="large")
datos = st.session_state.datos_estructura

with col_inspector:
    st.markdown("#### 🛠️ Inspector de Propiedades")
    pestanias_editor = ["⭕ Nodos y Apoyos", "📏 Barras", "⬇️ Cargas y Momentos", "📋 Tabla General"]
    tab_seleccionada = st.radio("Herramienta activa:", pestanias_editor, horizontal=True)

    # 1. NODOS Y APOYOS
    if tab_seleccionada == "⭕ Nodos y Apoyos":
        lista_nodos = [n["id"] for n in datos.get("nodos", [])]
        
        with st.expander("➕ **Agregar Nuevo Nodo**", expanded=(len(lista_nodos) == 0)):
            c_nid, c_nx, c_ny = st.columns([1.2, 1, 1])
            sug_id = f"N{len(lista_nodos) + 1}"
            n_id_in = c_nid.text_input("ID Nodo:", value=sug_id, key=f"input_nuevo_nid_{v_act}")
            n_x_in = c_nx.number_input("X (m):", value=0.0, step=0.5, key=f"input_nuevo_nx_{v_act}")
            n_y_in = c_ny.number_input("Y (m):", value=0.0, step=0.5, key=f"input_nuevo_ny_{v_act}")
            
            if st.button("✨ Crear Nodo", use_container_width=True, key=f"btn_crear_nodo_quick_{v_act}"):
                if n_id_in in lista_nodos:
                    st.warning(f"Ya existe el nodo {n_id_in}.")
                else:
                    datos["nodos"].append({"id": n_id_in, "x": n_x_in, "y": n_y_in})
                    st.session_state.datos_estructura = datos
                    st.rerun()

        if lista_nodos:
            with st.container(border=True):
                nodo_sel = st.selectbox("Seleccionar Nodo:", lista_nodos, key=f"sel_nodo_insp_{v_act}")
                idx_n_real = next(i for i, n in enumerate(datos["nodos"]) if n["id"] == nodo_sel)
                
                c_x, c_y = st.columns(2)
                nuevo_x = c_x.number_input("Coordenada X (m)", value=float(datos["nodos"][idx_n_real]["x"]), step=0.5, key=f"nx_{nodo_sel}_{v_act}")
                nuevo_y = c_y.number_input("Coordenada Y (m)", value=float(datos["nodos"][idx_n_real]["y"]), step=0.5, key=f"ny_{nodo_sel}_{v_act}")
                datos["nodos"][idx_n_real]["x"] = nuevo_x
                datos["nodos"][idx_n_real]["y"] = nuevo_y

                st.markdown("##### Condiciones de Vínculo")
                apoyo_obj = next((ap for ap in datos.get("apoyos", []) if ap["nodo"] == nodo_sel), None)
                apoyo_actual = apoyo_obj["tipo"] if apoyo_obj else "ninguno"
                ang_actual = float(apoyo_obj.get("angulo", 0.0)) if apoyo_obj else 0.0

                opciones_ap = ["ninguno", "fijo", "movil", "empotrado"]
                idx_ap = opciones_ap.index(apoyo_actual) if apoyo_actual in opciones_ap else 0
                nuevo_ap = st.selectbox("Vínculo Externo (Apoyo):", opciones_ap, index=idx_ap, key=f"ap_{nodo_sel}_{v_act}")

                if nuevo_ap != "ninguno":
                    nuevo_ang = st.number_input(
                        "Inclinación del Apoyo (° desde +X antihorario):", 
                        value=ang_actual, step=15.0, min_value=-360.0, max_value=360.0,
                        key=f"ang_{nodo_sel}_{v_act}", 
                        help="0° = Plano horizontal (+X). 90° = Plano vertical (+Y)."
                    )
                else:
                    nuevo_ang = 0.0

                es_rot = nodo_sel in datos.get("rotulas", [])
                nueva_rot = st.checkbox("Rótula Interna (Articulación M = 0)", value=es_rot, key=f"rot_{nodo_sel}_{v_act}")

                datos["apoyos"] = [ap for ap in datos.get("apoyos", []) if ap["nodo"] != nodo_sel]
                if nuevo_ap != "ninguno":
                    datos["apoyos"].append({"nodo": nodo_sel, "tipo": nuevo_ap, "angulo": nuevo_ang})

                rotulas_set = set(datos.get("rotulas", []))
                if nueva_rot: rotulas_set.add(nodo_sel)
                else: rotulas_set.discard(nodo_sel)
                datos["rotulas"] = list(rotulas_set)

                if st.button(f"🗑️ Eliminar Nodo {nodo_sel}", key=f"btn_del_nodo_{nodo_sel}_{v_act}"):
                    datos["nodos"].pop(idx_n_real)
                    datos["apoyos"] = [ap for ap in datos.get("apoyos", []) if ap["nodo"] != nodo_sel]
                    datos["rotulas"] = [r for r in datos.get("rotulas", []) if r != nodo_sel]
                    datos["barras"] = [b for b in datos.get("barras", []) if b["nodo_i"] != nodo_sel and b["nodo_j"] != nodo_sel]
                    st.session_state.datos_estructura = datos
                    st.rerun()

    # 2. BARRAS
    elif tab_seleccionada == "📏 Barras":
        lista_barras = [b["id"] for b in datos.get("barras", [])]
        lista_nodos = [n["id"] for n in datos.get("nodos", [])]

        with st.expander("➕ **Agregar Nueva Barra**", expanded=(len(lista_barras) == 0)):
            if len(lista_nodos) < 2:
                st.info("Necesitás crear al menos 2 nodos primero.")
            else:
                c_bid, c_bni, c_bnj = st.columns([1.2, 1, 1])
                sug_bid = f"Barra{len(lista_barras) + 1}"
                b_id_in = c_bid.text_input("ID Barra:", value=sug_bid, key=f"input_nuevo_bid_{v_act}")
                b_ni_in = c_bni.selectbox("Desde Nodo (i):", lista_nodos, index=0, key=f"input_nuevo_bni_{v_act}")
                b_nj_in = c_bnj.selectbox("Hasta Nodo (j):", lista_nodos, index=min(1, len(lista_nodos)-1), key=f"input_nuevo_bnj_{v_act}")

                if st.button("✨ Crear Barra", use_container_width=True, key=f"btn_crear_barra_quick_{v_act}"):
                    if b_ni_in == b_nj_in:
                        st.warning("Una barra debe conectar dos nodos distintos.")
                    elif b_id_in in lista_barras:
                        st.warning(f"Ya existe {b_id_in}.")
                    else:
                        datos["barras"].append({"id": b_id_in, "nodo_i": b_ni_in, "nodo_j": b_nj_in})
                        st.session_state.datos_estructura = datos
                        st.rerun()

        if lista_barras and lista_nodos:
            with st.container(border=True):
                barra_sel = st.selectbox("Seleccionar Barra:", lista_barras, key=f"sel_barra_insp_{v_act}")
                idx_b = next(i for i, b in enumerate(datos["barras"]) if b["id"] == barra_sel)
                
                c_ni, c_nj = st.columns(2)
                n_i_actual = datos["barras"][idx_b]["nodo_i"]
                n_j_actual = datos["barras"][idx_b]["nodo_j"]
                
                idx_i = lista_nodos.index(n_i_actual) if n_i_actual in lista_nodos else 0
                idx_j = lista_nodos.index(n_j_actual) if n_j_actual in lista_nodos else min(1, len(lista_nodos)-1)
                
                nuevo_ni = c_ni.selectbox("Nodo Inicial (i):", lista_nodos, index=idx_i, key=f"ni_{barra_sel}_{v_act}")
                nuevo_nj = c_nj.selectbox("Nodo Final (j):", lista_nodos, index=idx_j, key=f"nj_{barra_sel}_{v_act}")
                
                datos["barras"][idx_b]["nodo_i"] = nuevo_ni
                datos["barras"][idx_b]["nodo_j"] = nuevo_nj

                nodos_d = {n["id"]: (float(n["x"]), float(n["y"])) for n in datos.get("nodos", [])}
                if nuevo_ni in nodos_d and nuevo_nj in nodos_d:
                    x1, y1 = nodos_d[nuevo_ni]
                    x2, y2 = nodos_d[nuevo_nj]
                    st.caption(f"📏 Longitud calculada: **{math.hypot(x2 - x1, y2 - y1):.2f} m**")

                if st.button(f"🗑️ Eliminar Barra {barra_sel}", key=f"del_{barra_sel}_{v_act}"):
                    datos["barras"].pop(idx_b)
                    datos["cargas_distribuidas"] = [cd for cd in datos.get("cargas_distribuidas", []) if cd.get("barra") != barra_sel]
                    datos["cargas_puntuales"] = [cp for cp in datos.get("cargas_puntuales", []) if cp.get("barra") != barra_sel]
                    datos["momentos_concentrados"] = [mc for mc in datos.get("momentos_concentrados", []) if mc.get("barra") != barra_sel]
                    st.session_state.datos_estructura = datos
                    st.rerun()

    # 3. CARGAS Y MOMENTOS
    elif tab_seleccionada == "⬇️ Cargas y Momentos":
        lista_barras = [b["id"] for b in datos.get("barras", [])]
        lista_nodos = [n["id"] for n in datos.get("nodos", [])]
        
        cant_cd = len(datos.get("cargas_distribuidas", []))
        cant_cp = len(datos.get("cargas_puntuales", []))
        cant_mc = len(datos.get("momentos_concentrados", []))
        
        tipo_c = st.segmented_control(
            "Tipo de Solicitación:", 
            [f"Distribuidas ({cant_cd})", f"Puntuales ({cant_cp})", f"Momentos ({cant_mc})"], 
            default=f"Distribuidas ({cant_cd})" if cant_cd >= cant_cp else f"Puntuales ({cant_cp})"
        )

        if "Distribuidas" in str(tipo_c):
            cds = datos.get("cargas_distribuidas", [])
            
            if st.button("➕ Agregar Carga Distribuida (q)", use_container_width=True, key=f"btn_add_cd_{v_act}"):
                if lista_barras:
                    datos["cargas_distribuidas"].append({
                        "barra": lista_barras[0], "w_inicio": 10.0, "w_fin": 10.0, 
                        "x_inicio": 0.0, "x_fin": 5.0, "direccion": "-Y"
                    })
                    st.session_state.datos_estructura = datos
                    st.rerun()
                else:
                    st.warning("Creá al menos una barra primero.")

            if cds:
                idx_cd = st.selectbox("Seleccionar Carga a Editar:", range(len(cds)), format_func=lambda i: f"q{i+1} en {cds[i].get('barra', '')}", key=f"sel_cd_{v_act}")
                uid = f"cd_{idx_cd}_{len(cds)}_{v_act}"
                
                with st.container(border=True):
                    if lista_barras:
                        curr_b = cds[idx_cd].get("barra", lista_barras[0])
                        idx_b = lista_barras.index(curr_b) if curr_b in lista_barras else 0
                        cds[idx_cd]["barra"] = st.selectbox("Barra destino:", lista_barras, index=idx_b, key=f"b_{uid}")

                    c_w1, c_w2 = st.columns(2)
                    cds[idx_cd]["w_inicio"] = c_w1.number_input(f"w inicio ({datos.get('unidad_fuerza', 'kN')}/m)", value=float(cds[idx_cd].get("w_inicio", 10.0)), step=1.0, key=f"w1_{uid}")
                    cds[idx_cd]["w_fin"] = c_w2.number_input(f"w fin ({datos.get('unidad_fuerza', 'kN')}/m)", value=float(cds[idx_cd].get("w_fin", 10.0)), step=1.0, key=f"w2_{uid}")
                    
                    c_x1, c_x2 = st.columns(2)
                    cds[idx_cd]["x_inicio"] = c_x1.number_input("x inicio (m)", value=float(cds[idx_cd].get("x_inicio", 0.0)), step=0.5, key=f"x1_{uid}")
                    cds[idx_cd]["x_fin"] = c_x2.number_input("x fin (m)", value=float(cds[idx_cd].get("x_fin", 5.0)), step=0.5, key=f"x2_{uid}")
                    
                    dirs = ["-Y", "+Y", "-X", "+X", "perpendicular_adentro", "perpendicular_afuera"]
                    curr_dir = cds[idx_cd].get("direccion", "-Y")
                    idx_d = dirs.index(curr_dir) if curr_dir in dirs else 0
                    cds[idx_cd]["direccion"] = st.selectbox("Dirección:", dirs, index=idx_d, key=f"dir_{uid}")
                    
                    if st.button("🗑️ Eliminar Carga Distribuida", key=f"del_{uid}"):
                        datos["cargas_distribuidas"].pop(idx_cd)
                        st.session_state.datos_estructura = datos
                        st.rerun()

        elif "Puntuales" in str(tipo_c):
            cps = datos.get("cargas_puntuales", [])
            
            if st.button("➕ Agregar Carga Puntual (P)", use_container_width=True, key=f"btn_add_cp_{v_act}"):
                if lista_barras:
                    datos["cargas_puntuales"].append({"barra": lista_barras[0], "posicion_x": 0.0, "magnitud": 50.0, "angulo": 45.0, "sentido": "abajo_izquierda"})
                    st.session_state.datos_estructura = datos
                    st.rerun()
                elif lista_nodos:
                    datos["cargas_puntuales"].append({"nodo": lista_nodos[0], "magnitud": 50.0, "direccion": "-Y", "angulo": None})
                    st.session_state.datos_estructura = datos
                    st.rerun()

            if cps:
                idx_cp = st.selectbox("Seleccionar Carga a Editar:", range(len(cps)), format_func=lambda i: f"P{i+1} ({cps[i].get('magnitud', '')} {datos.get('unidad_fuerza', 'kN')})", key=f"sel_cp_{v_act}")
                uid = f"cp_{idx_cp}_{len(cps)}_{v_act}"
                
                with st.container(border=True):
                    es_nodo = "nodo" in cps[idx_cp]
                    tipo_apli = st.radio("Ubicación de aplicación:", ["Barra", "Nodo"], index=1 if es_nodo else 0, key=f"tipo_{uid}", horizontal=True)
                    
                    if tipo_apli == "Barra":
                        if "nodo" in cps[idx_cp]:
                            del cps[idx_cp]["nodo"]
                            cps[idx_cp]["barra"] = lista_barras[0] if lista_barras else ""
                            cps[idx_cp]["posicion_x"] = 0.0
                        if lista_barras:
                            idx_b = lista_barras.index(cps[idx_cp]["barra"]) if cps[idx_cp]["barra"] in lista_barras else 0
                            cps[idx_cp]["barra"] = st.selectbox("Barra destino:", lista_barras, index=idx_b, key=f"b_{uid}")
                            cps[idx_cp]["posicion_x"] = st.number_input("Posición x desde nodo (i) [m]:", value=float(cps[idx_cp].get("posicion_x", 0.0)), step=0.5, key=f"px_{uid}")
                    else:
                        if "barra" in cps[idx_cp]:
                            del cps[idx_cp]["barra"]
                            if "posicion_x" in cps[idx_cp]: del cps[idx_cp]["posicion_x"]
                            cps[idx_cp]["nodo"] = lista_nodos[0] if lista_nodos else ""
                        if lista_nodos:
                            idx_n = lista_nodos.index(cps[idx_cp]["nodo"]) if cps[idx_cp]["nodo"] in lista_nodos else 0
                            cps[idx_cp]["nodo"] = st.selectbox("Nodo destino:", lista_nodos, index=idx_n, key=f"n_{uid}")

                    cps[idx_cp]["magnitud"] = st.number_input(f"Magnitud ({datos.get('unidad_fuerza', 'kN')}):", value=float(cps[idx_cp].get("magnitud", 50.0)), step=5.0, key=f"m_{uid}")
                    
                    es_inclinada = cps[idx_cp].get("angulo") is not None
                    modo_dir = st.radio("Modo de Dirección:", ["Inclinada (Ángulo + Sentido)", "Ortogonal pura (-Y, +Y, -X, +X)"], index=0 if es_inclinada else 1, key=f"mododir_{uid}", horizontal=True)
                    
                    if "Inclinada" in modo_dir:
                        c_ang, c_sent = st.columns(2)
                        raw_ang = float(cps[idx_cp].get("angulo", 45.0) if cps[idx_cp].get("angulo") is not None else 45.0)
                        disp_ang = raw_ang % 90.0 if raw_ang > 90.0 else raw_ang
                        if disp_ang == 0.0 and raw_ang > 0.0: disp_ang = 45.0
                        
                        cps[idx_cp]["angulo"] = c_ang.number_input(
                            "Ángulo acotado (°):", value=disp_ang, step=5.0, min_value=0.0, max_value=90.0, key=f"angval_{uid}",
                            help="El ángulo positivo escrito en el croquis respecto a la horizontal."
                        )
                        
                        opciones_sent = [
                            "↙ Hacia abajo e izquierda",
                            "↘ Hacia abajo y derecha",
                            "↗ Hacia arriba y derecha",
                            "↖ Hacia arriba e izquierda"
                        ]
                        sent_val = str(cps[idx_cp].get("sentido", "abajo_izquierda")).lower()
                        idx_sent = 0
                        if "abajo" in sent_val and "derecha" in sent_val: idx_sent = 1
                        elif "arriba" in sent_val and "derecha" in sent_val: idx_sent = 2
                        elif "arriba" in sent_val and "izquierda" in sent_val: idx_sent = 3
                        
                        sent_sel = c_sent.selectbox("Sentido de la Flecha:", opciones_sent, index=idx_sent, key=f"sentval_{uid}")
                        if "izquierda" in sent_sel and "abajo" in sent_sel: cps[idx_cp]["sentido"] = "abajo_izquierda"
                        elif "derecha" in sent_sel and "abajo" in sent_sel: cps[idx_cp]["sentido"] = "abajo_derecha"
                        elif "derecha" in sent_sel and "arriba" in sent_sel: cps[idx_cp]["sentido"] = "arriba_derecha"
                        elif "izquierda" in sent_sel and "arriba" in sent_sel: cps[idx_cp]["sentido"] = "arriba_izquierda"
                        
                    else:
                        dirs = ["-Y", "+Y", "-X", "+X"]
                        curr_dir = cps[idx_cp].get("direccion", "-Y")
                        idx_d = dirs.index(curr_dir) if curr_dir in dirs else 0
                        cps[idx_cp]["direccion"] = st.selectbox("Dirección:", dirs, index=idx_d, key=f"dir_{uid}")
                        cps[idx_cp]["angulo"] = None
                    
                    if st.button("🗑️ Eliminar Carga Puntual", key=f"del_{uid}"):
                        datos["cargas_puntuales"].pop(idx_cp)
                        st.session_state.datos_estructura = datos
                        st.rerun()

        else:
            mcs = datos.get("momentos_concentrados", [])
            
            if st.button("➕ Agregar Momento Concentrado (M₀)", use_container_width=True, key=f"btn_add_mc_{v_act}"):
                if lista_barras:
                    datos["momentos_concentrados"].append({"barra": lista_barras[0], "posicion_x": 0.0, "magnitud": 15.0, "sentido": "horario"})
                    st.session_state.datos_estructura = datos
                    st.rerun()
                elif lista_nodos:
                    datos["momentos_concentrados"].append({"nodo": lista_nodos[0], "magnitud": 15.0, "sentido": "horario"})
                    st.session_state.datos_estructura = datos
                    st.rerun()

            if mcs:
                idx_mc = st.selectbox("Seleccionar Momento:", range(len(mcs)), format_func=lambda i: f"M₀_{i+1} ({mcs[i].get('magnitud', '')} {datos.get('unidad_fuerza', 'kN')}·m)", key=f"sel_mc_{v_act}")
                uid = f"mc_{idx_mc}_{len(mcs)}_{v_act}"
                
                with st.container(border=True):
                    es_nodo = "nodo" in mcs[idx_mc]
                    tipo_apli = st.radio("Ubicación de aplicación:", ["Barra", "Nodo"], index=1 if es_nodo else 0, key=f"tipo_{uid}", horizontal=True)
                    
                    if tipo_apli == "Barra":
                        if "nodo" in mcs[idx_mc]:
                            del mcs[idx_mc]["nodo"]
                            mcs[idx_mc]["barra"] = lista_barras[0] if lista_barras else ""
                            mcs[idx_mc]["posicion_x"] = 0.0
                        if lista_barras:
                            idx_b = lista_barras.index(mcs[idx_mc]["barra"]) if mcs[idx_mc]["barra"] in lista_barras else 0
                            mcs[idx_mc]["barra"] = st.selectbox("Barra destino:", lista_barras, index=idx_b, key=f"b_{uid}")
                            mcs[idx_mc]["posicion_x"] = st.number_input("Posición x desde nodo (i) [m]:", value=float(mcs[idx_mc].get("posicion_x", 0.0)), step=0.5, key=f"px_{uid}")
                    else:
                        if "barra" in mcs[idx_mc]:
                            del mcs[idx_mc]["barra"]
                            if "posicion_x" in mcs[idx_mc]: del mcs[idx_mc]["posicion_x"]
                            mcs[idx_mc]["nodo"] = lista_nodos[0] if lista_nodos else ""
                        if lista_nodos:
                            idx_n = lista_nodos.index(mcs[idx_mc]["nodo"]) if mcs[idx_mc]["nodo"] in lista_nodos else 0
                            mcs[idx_mc]["nodo"] = st.selectbox("Nodo destino:", lista_nodos, index=idx_n, key=f"n_{uid}")

                    mcs[idx_mc]["magnitud"] = st.number_input(f"Magnitud ({datos.get('unidad_fuerza', 'kN')}·m):", value=float(mcs[idx_mc].get("magnitud", 10.0)), step=1.0, key=f"m_{uid}")
                    
                    curr_sentido = mcs[idx_mc].get("sentido", "horario").lower()
                    idx_s = 0 if curr_sentido == "horario" else 1
                    mcs[idx_mc]["sentido"] = st.selectbox("Sentido de Giro:", ["horario", "antihorario"], index=idx_s, key=f"s_{uid}")
                    
                    if st.button("🗑️ Eliminar Momento", key=f"del_{uid}"):
                        datos["momentos_concentrados"].pop(idx_mc)
                        st.session_state.datos_estructura = datos
                        st.rerun()

    # 4. TABLA GENERAL MASIVA
    else:
        st.caption("Edición masiva de todas las entidades:")
        t_tab1, t_tab2, t_tab3 = st.tabs(["Nodos", "Barras", "Apoyos"])
        with t_tab1: datos["nodos"] = st.data_editor(datos.get("nodos", []), num_rows="dynamic", key=f"tab_nodos_{v_act}", use_container_width=True)
        with t_tab2: datos["barras"] = st.data_editor(datos.get("barras", []), num_rows="dynamic", key=f"tab_barras_{v_act}", use_container_width=True)
        with t_tab3: datos["apoyos"] = st.data_editor(datos.get("apoyos", []), num_rows="dynamic", key=f"tab_apoyos_{v_act}", use_container_width=True)

    st.session_state.datos_estructura = datos

# Canvas
with col_canvas:
    st.markdown("#### 📐 Vista Previa del Modelo")
    st.plotly_chart(crear_esquema_modelo(st.session_state.datos_estructura), use_container_width=True)

# ==========================================================
# MOTOR DE CÁLCULO (PYNITE) CON BLINDAJE CINEMÁTICO
# ==========================================================
def resolver_modelo(datos):
    modelo = FEModel3D()
    
    un_f = str(datos.get("unidad_fuerza", "kN")).lower()
    if "t" in un_f:
        factor_fuerza = 9806.65
    elif "kg" in un_f:
        factor_fuerza = 9.80665
    elif "kn" in un_f:
        factor_fuerza = 1000.0
    else:
        factor_fuerza = 1.0

    for n in datos.get("nodos", []):
        modelo.add_node(n["id"], float(n["x"]), float(n["y"]), 0.0)

    for n_id in modelo.nodes:
        modelo.def_support(n_id, support_DZ=True, support_RX=True, support_RY=True)

    modelo.add_material('Mat', E=e_mod, G=77e9, nu=0.3, rho=7850)
    modelo.add_section('Sec', A=area_sec, Iy=iz_sec, Iz=iz_sec, J=2*iz_sec)

    for ap in datos.get("apoyos", []):
        nodo = ap["nodo"]
        t = ap["tipo"].lower()
        ang_deg = float(ap.get("angulo", 0.0)) % 180.0
        
        if "empotrado" in t:
            modelo.def_support(nodo, support_DX=True, support_DY=True, support_DZ=True, support_RX=True, support_RY=True, support_RZ=True)
        elif "fijo" in t:
            modelo.def_support(nodo, support_DX=True, support_DY=True, support_DZ=True, support_RX=True, support_RY=True, support_RZ=False)
        elif "movil" in t or "móvil" in t:
            if abs(ang_deg) < 1e-2:
                modelo.def_support(nodo, support_DX=False, support_DY=True, support_DZ=True, support_RX=True, support_RY=True, support_RZ=False)
            elif abs(ang_deg - 90.0) < 1e-2:
                modelo.def_support(nodo, support_DX=True, support_DY=False, support_DZ=True, support_RX=True, support_RY=True, support_RZ=False)
            else:
                rad = math.radians(float(ap.get("angulo", 0.0)))
                nx_s = -math.sin(rad)
                ny_s = math.cos(rad)
                
                n_obj = modelo.nodes[nodo]
                x0 = float(getattr(n_obj, 'X', getattr(n_obj, 'x', 0)))
                y0 = float(getattr(n_obj, 'Y', getattr(n_obj, 'y', 0)))
                
                g_id = f"gnd_{nodo}"
                modelo.add_node(g_id, x0 - 1.0 * nx_s, y0 - 1.0 * ny_s, 0.0)
                modelo.def_support(g_id, support_DX=True, support_DY=True, support_DZ=True, support_RX=True, support_RY=True, support_RZ=True)
                
                link_id = f"link_{nodo}"
                modelo.add_member(link_id, g_id, nodo, 'Mat', 'Sec')
                modelo.def_releases(link_id, Rzi=True, Rzj=True)
                modelo.def_support(nodo, support_DX=False, support_DY=False, support_DZ=True, support_RX=True, support_RY=True, support_RZ=False)

    for b in datos.get("barras", []):
        modelo.add_member(b["id"], b["nodo_i"], b["nodo_j"], 'Mat', 'Sec')

    rotulas = set(datos.get("rotulas", []))
    rotulas_lib = set()
    for b in datos.get("barras", []):
        if b["nodo_i"] in rotulas and b["nodo_i"] not in rotulas_lib:
            modelo.def_releases(b["id"], Rzi=True)
            rotulas_lib.add(b["nodo_i"])
        elif b["nodo_j"] in rotulas and b["nodo_j"] not in rotulas_lib:
            modelo.def_releases(b["id"], Rzj=True)
            rotulas_lib.add(b["nodo_j"])

    for cp in datos.get("cargas_puntuales", []):
        fx_u, fy_u = vector_fuerza_puntual(cp)
        fx_si = fx_u * factor_fuerza
        fy_si = fy_u * factor_fuerza
        
        if "nodo" in cp:
            if abs(fx_si) > 1e-4: modelo.add_node_load(cp["nodo"], 'FX', fx_si, case='Case 1')
            if abs(fy_si) > 1e-4: modelo.add_node_load(cp["nodo"], 'FY', fy_si, case='Case 1')
        elif "barra" in cp:
            m_elem = modelo.members[cp["barra"]]
            pos = float(cp.get("posicion_x", 0))
            
            ni = m_elem.i_node if not isinstance(m_elem.i_node, str) else modelo.nodes[m_elem.i_node]
            nj = m_elem.j_node if not isinstance(m_elem.j_node, str) else modelo.nodes[m_elem.j_node]
            dx = float(getattr(nj, 'X', getattr(nj, 'x', 0))) - float(getattr(ni, 'X', getattr(ni, 'x', 0)))
            dy = float(getattr(nj, 'Y', getattr(nj, 'y', 0))) - float(getattr(ni, 'Y', getattr(ni, 'y', 0)))
            L_b = math.hypot(dx, dy)
            cos_b, sin_b = dx / L_b, dy / L_b
            
            fx_loc = fx_si * cos_b + fy_si * sin_b
            fy_loc = -fx_si * sin_b + fy_si * cos_b
            
            if abs(fx_loc) > 1e-4: modelo.add_member_pt_load(cp["barra"], 'Fx', fx_loc, pos, case='Case 1')
            if abs(fy_loc) > 1e-4: modelo.add_member_pt_load(cp["barra"], 'Fy', fy_loc, pos, case='Case 1')

    for cd in datos.get("cargas_distribuidas", []):
        m_elem = modelo.members[cd["barra"]]
        x1 = float(cd.get("x_inicio", 0.0))
        x2 = float(cd.get("x_fin", m_elem.L()))
        w1_si = abs(float(cd.get("w_inicio", 0))) * factor_fuerza
        w2_si = abs(float(cd.get("w_fin", 0))) * factor_fuerza
        dir_cd = str(cd.get("direccion", "-Y")).upper()
        
        if dir_cd == "-Y":
            modelo.add_member_dist_load(cd["barra"], "FY", -w1_si, -w2_si, x1, x2, case='Case 1')
        elif dir_cd == "+Y":
            modelo.add_member_dist_load(cd["barra"], "FY", w1_si, w2_si, x1, x2, case='Case 1')
        elif dir_cd == "-X":
            modelo.add_member_dist_load(cd["barra"], "FX", -w1_si, -w2_si, x1, x2, case='Case 1')
        elif dir_cd == "+X":
            modelo.add_member_dist_load(cd["barra"], "FX", w1_si, w2_si, x1, x2, case='Case 1')
        elif dir_cd == "PERPENDICULAR_ADENTRO":
            modelo.add_member_dist_load(cd["barra"], "Fy", -w1_si, -w2_si, x1, x2, case='Case 1')
        elif dir_cd == "PERPENDICULAR_AFUERA":
            modelo.add_member_dist_load(cd["barra"], "Fy", w1_si, w2_si, x1, x2, case='Case 1')

    for mc in datos.get("momentos_concentrados", []):
        mag_si = abs(float(mc.get("magnitud", 0))) * factor_fuerza
        mz_si = -mag_si if "horario" in mc.get("sentido", "horario").lower() and "anti" not in mc.get("sentido", "").lower() else mag_si
        
        if "nodo" in mc:
            modelo.add_node_load(mc["nodo"], 'MZ', mz_si, case='Case 1')
        elif "barra" in mc:
            modelo.add_member_pt_load(mc["barra"], "Mz", mz_si, float(mc.get("posicion_x", 0)), case='Case 1')

    if 'Combo 1' not in modelo.load_combos:
        modelo.add_load_combo('Combo 1', {'Case 1': 1.0})

    modelo.analyze()
    return modelo, factor_fuerza

# ==========================================================
# DIAGRAMAS INTERACTIVOS (M, Q, N)
# ==========================================================
def construir_diagrama_plotly(modelo, datos, tipo_diagrama, factor_fuerza):
    unidad_f = datos.get("unidad_fuerza", "kN")
    unidad_m = f"{unidad_f}·m"
    
    colores = {
        "M": ("#ef4444", "rgba(239, 68, 68, 0.22)", "Momento Flector (M) [Fibras Traccionadas]", unidad_m),
        "Q": ("#3b82f6", "rgba(59, 130, 246, 0.22)", "Esfuerzo de Corte (Q)", unidad_f),
        "N": ("#10b981", "rgba(16, 185, 129, 0.22)", "Esfuerzo Normal (N)", unidad_f)
    }
    
    color_linea, color_fill, titulo_diag, un_txt = colores[tipo_diagrama]
    fig = go.Figure()

    miembros_reales = {k: v for k, v in modelo.members.items() if not k.startswith("link_")}

    xs = [float(getattr(n, 'X', getattr(n, 'x', 0))) for k, n in modelo.nodes.items() if not k.startswith("gnd_")]
    ys = [float(getattr(n, 'Y', getattr(n, 'y', 0))) for k, n in modelo.nodes.items() if not k.startswith("gnd_")]
    L_max = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    pad = 0.25 * L_max
    fig.add_trace(go.Scatter(
        x=[min(xs) - pad, max(xs) + pad], y=[min(ys) - pad, max(ys) + pad],
        mode='markers', marker=dict(size=0.1, opacity=0), hoverinfo='skip', showlegend=False
    ))

    max_val = 1.0
    for m in miembros_reales.values():
        if tipo_diagrama == "M": v = np.max(np.abs(m.moment_array("Mz", n_points=200)[1])) / factor_fuerza
        elif tipo_diagrama == "Q": v = np.max(np.abs(m.shear_array("Fy", n_points=200)[1])) / factor_fuerza
        else: v = np.max(np.abs(m.axial_array(n_points=200)[1])) / factor_fuerza
        if v > max_val: max_val = v

    escala = (0.18 * L_max) / (max_val if max_val > 0 else 1.0)

    for nombre_m, miembro in miembros_reales.items():
        n_i = miembro.i_node if not isinstance(miembro.i_node, str) else modelo.nodes[miembro.i_node]
        n_j = miembro.j_node if not isinstance(miembro.j_node, str) else modelo.nodes[miembro.j_node]
        xi, yi = float(getattr(n_i, 'X', getattr(n_i, 'x', 0))), float(getattr(n_i, 'Y', getattr(n_i, 'y', 0)))
        xj, yj = float(getattr(n_j, 'X', getattr(n_j, 'x', 0))), float(getattr(n_j, 'Y', getattr(n_j, 'y', 0)))

        L_elem = miembro.L()
        if L_elem < 1e-4: continue
        cos_t, sin_t = (xj - xi) / L_elem, (yj - yi) / L_elem

        fig.add_trace(go.Scatter(
            x=[xi, xj], y=[yi, yj],
            mode='lines', line=dict(color='#cbd5e1', width=3),
            hoverinfo='skip', showlegend=False
        ))

        if tipo_diagrama == "M": 
            x_loc, vals_si = miembro.moment_array("Mz", n_points=300)
            vals = vals_si / factor_fuerza
            nx, ny = -sin_t, cos_t
            vals_plot = vals
        elif tipo_diagrama == "Q": 
            x_loc, vals_si = miembro.shear_array("Fy", n_points=300)
            vals = vals_si / factor_fuerza
            nx, ny = -sin_t, cos_t
            vals_plot = vals
        else: 
            x_loc, vals_si = miembro.axial_array(n_points=300)
            vals = vals_si / factor_fuerza
            nx, ny = -sin_t, cos_t
            vals_plot = vals

        x_base = xi + x_loc * cos_t
        y_base = yi + x_loc * sin_t
        x_diag = x_base + (vals_plot * escala) * nx
        y_diag = y_base + (vals_plot * escala) * ny

        x_poly = np.concatenate([x_base, x_diag[::-1], [x_base[0]]])
        y_poly = np.concatenate([y_base, y_diag[::-1], [y_base[0]]])

        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly, fill='toself',
            fillcolor=color_fill, line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip', showlegend=False
        ))

        hover_texts = [
            f"<b>{nombre_m}</b><br>x: {x_loc[k]:.2f} m<br><b>{tipo_diagrama}:</b> {vals[k]:.2f} {un_txt}"
            for k in range(len(x_loc))
        ]
        fig.add_trace(go.Scatter(
            x=x_diag, y=y_diag, mode='lines',
            line=dict(color=color_linea, width=2.5),
            text=hover_texts, hoverinfo='text',
            name=nombre_m
        ))

        idx_ext = [0, len(vals)-1, int(np.argmax(np.abs(vals)))]
        for idx in set(idx_ext):
            if abs(vals[idx]) > 0.05:
                txt_val = f"{abs(vals[idx]):.2f}" if tipo_diagrama == "M" else f"{('(-) ' if vals[idx] < 0 else '(+) ') if tipo_diagrama == 'N' else ''}{abs(vals[idx]):.2f}"
                fig.add_annotation(
                    x=x_diag[idx], y=y_diag[idx],
                    text=f"<b>{txt_val} {un_txt}</b>",
                    showarrow=False, font=dict(color=color_linea, size=11),
                    bgcolor="#1e293b", bordercolor=color_linea, borderwidth=1, borderpad=3
                )

    rotulas = set(datos.get("rotulas", []))
    for r_id in rotulas:
        if r_id in modelo.nodes:
            nr = modelo.nodes[r_id]
            fig.add_trace(go.Scatter(
                x=[float(getattr(nr, 'X', getattr(nr, 'x', 0)))],
                y=[float(getattr(nr, 'Y', getattr(nr, 'y', 0)))],
                mode='markers', marker=dict(size=12, color='#ffffff', line=dict(color='#0f172a', width=2)),
                hoverinfo='skip', showlegend=False
            ))

    fig.update_layout(
        title=f"<b>{titulo_diag}</b>",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=520, showlegend=False
    )
    return fig

# ==========================================================
# DIAGRAMA DE VECTORES REACTIVOS EN APOYOS
# ==========================================================
def extraer_reaccion(node_obj, dof):
    val = getattr(node_obj, dof, 0.0)
    if isinstance(val, dict):
        return float(val.get('Combo 1', list(val.values())[0] if val else 0.0))
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0

def construir_diagrama_reacciones(modelo, datos, factor_fuerza):
    fig = go.Figure()
    unidad_f = datos.get("unidad_fuerza", "kN")
    unidad_m = f"{unidad_f}·m"

    miembros_reales = {k: v for k, v in modelo.members.items() if not k.startswith("link_")}
    xs = [float(getattr(n, 'X', getattr(n, 'x', 0))) for k, n in modelo.nodes.items() if not k.startswith("gnd_")]
    ys = [float(getattr(n, 'Y', getattr(n, 'y', 0))) for k, n in modelo.nodes.items() if not k.startswith("gnd_")]
    L_max = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    pad = 0.35 * L_max
    fig.add_trace(go.Scatter(
        x=[min(xs) - pad, max(xs) + pad], y=[min(ys) - pad, max(ys) + pad],
        mode='markers', marker=dict(size=0.1, opacity=0), hoverinfo='skip', showlegend=False
    ))

    for m in miembros_reales.values():
        ni = m.i_node if not isinstance(m.i_node, str) else modelo.nodes[m.i_node]
        nj = m.j_node if not isinstance(m.j_node, str) else modelo.nodes[m.j_node]
        fig.add_trace(go.Scatter(
            x=[float(getattr(ni, 'X', getattr(ni, 'x', 0))), float(getattr(nj, 'X', getattr(nj, 'x', 0)))],
            y=[float(getattr(ni, 'Y', getattr(ni, 'y', 0))), float(getattr(nj, 'Y', getattr(nj, 'y', 0)))],
            mode='lines', line=dict(color='#64748b', width=4), hoverinfo='skip', showlegend=False
        ))

    for n_id, (x, y) in {k: (float(getattr(v, 'X', getattr(v, 'x', 0))), float(getattr(v, 'Y', getattr(v, 'y', 0)))) for k, v in modelo.nodes.items() if not k.startswith("gnd_")}.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=10, color='#38bdf8'),
            text=[f"<b>{n_id}</b>"], textposition="top left",
            hoverinfo='skip', showlegend=False
        ))

    l_arr = 0.16 * L_max
    for ap in datos.get("apoyos", []):
        n_id = ap["nodo"]
        if n_id in modelo.nodes:
            n_obj = modelo.nodes[n_id]
            x0 = float(getattr(n_obj, 'X', getattr(n_obj, 'x', 0)))
            y0 = float(getattr(n_obj, 'Y', getattr(n_obj, 'y', 0)))
            
            link_id = f"link_{n_id}"
            if link_id in modelo.members:
                f_link = modelo.members[link_id].axial_array(n_points=2)[1][0]
                rad = math.radians(float(ap.get("angulo", 0.0)))
                nx_s = -math.sin(rad)
                ny_s = math.cos(rad)
                rx = (-f_link * nx_s) / factor_fuerza
                ry = (-f_link * ny_s) / factor_fuerza
                mz = 0.0
            else:
                rx = extraer_reaccion(n_obj, 'RxnFX') / factor_fuerza
                ry = extraer_reaccion(n_obj, 'RxnFY') / factor_fuerza
                mz = extraer_reaccion(n_obj, 'RxnMZ') / factor_fuerza

            if abs(rx) > 1e-3:
                signo_x = 1.0 if rx > 0 else -1.0
                fig.add_annotation(
                    x=x0, y=y0, ax=x0 - signo_x * l_arr, ay=y0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=2.5, arrowcolor="#f59e0b",
                    text=f"<b>Rx = {abs(rx):.2f} {unidad_f}</b>",
                    font=dict(color="#f59e0b", size=11),
                    bgcolor="#0f172a", bordercolor="#f59e0b", borderwidth=1, borderpad=3
                )

            if abs(ry) > 1e-3:
                signo_y = 1.0 if ry > 0 else -1.0
                fig.add_annotation(
                    x=x0, y=y0, ax=x0, ay=y0 - signo_y * l_arr,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=2.5, arrowcolor="#10b981",
                    text=f"<b>Ry = {abs(ry):.2f} {unidad_f}</b>",
                    font=dict(color="#10b981", size=11),
                    bgcolor="#0f172a", bordercolor="#10b981", borderwidth=1, borderpad=3
                )

            if abs(mz) > 1e-3:
                r_arc = 0.10 * L_max
                st_horario = (mz < 0)
                t_vals = np.linspace(5*np.pi/6, np.pi/6, 20) if st_horario else np.linspace(np.pi/6, 5*np.pi/6, 20)
                x_arc = x0 + r_arc * np.cos(t_vals)
                y_arc = y0 + r_arc * np.sin(t_vals)
                
                fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', line=dict(color='#e879f9', width=2.5), hoverinfo='skip', showlegend=False))
                fig.add_annotation(
                    x=x_arc[-1], y=y_arc[-1], ax=x_arc[-2], ay=y_arc[-2],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor="#e879f9"
                )
                fig.add_annotation(
                    x=x0, y=y0 + r_arc + 0.05 * L_max,
                    text=f"<b>Mz = {abs(mz):.2f} {unidad_m}</b>",
                    showarrow=False, font=dict(color="#e879f9", size=11),
                    bgcolor="#0f172a", bordercolor="#e879f9", borderwidth=1, borderpad=3
                )

    fig.update_layout(
        title="<b>Vectores Reactivos en Apoyos (Reacciones de Vínculo)</b>",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=520, showlegend=False
    )
    return fig

# ==========================================================
# DEFORMADA ELÁSTICA CONTINUA
# ==========================================================
def construir_deformada_plotly(modelo, datos):
    fig = go.Figure()

    def extraer_disp_nodo(nodo, dof):
        val = getattr(nodo, dof, 0.0)
        if isinstance(val, dict):
            return float(val.get('Combo 1', list(val.values())[0] if val else 0.0))
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    miembros_reales = {k: v for k, v in modelo.members.items() if not k.startswith("link_")}
    nodos_dict = {n_id: (float(getattr(node, 'X', getattr(node, 'x', 0))),
                         float(getattr(node, 'Y', getattr(node, 'y', 0))))
                  for n_id, node in modelo.nodes.items() if not n_id.startswith("gnd_")}

    xs = [x for x, y in nodos_dict.values()]
    ys = [y for x, y in nodos_dict.values()]
    L_max = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    datos_deformada = []
    max_disp_global = 1e-9

    for nombre_m, miembro in miembros_reales.items():
        n_i = miembro.i_node if not isinstance(miembro.i_node, str) else modelo.nodes[miembro.i_node]
        n_j = miembro.j_node if not isinstance(miembro.j_node, str) else modelo.nodes[miembro.j_node]

        xi, yi = float(getattr(n_i, 'X', getattr(n_i, 'x', 0))), float(getattr(n_i, 'Y', getattr(n_i, 'y', 0)))
        xj, yj = float(getattr(n_j, 'X', getattr(n_j, 'x', 0))), float(getattr(n_j, 'Y', getattr(n_j, 'y', 0)))

        L_elem = miembro.L()
        if L_elem < 1e-4: continue
        cos_t, sin_t = (xj - xi) / L_elem, (yj - yi) / L_elem

        u_xi = extraer_disp_nodo(n_i, 'DX')
        u_yi = extraer_disp_nodo(n_i, 'DY')
        u_xj = extraer_disp_nodo(n_j, 'DX')
        u_yj = extraer_disp_nodo(n_j, 'DY')

        u_xloc_i = u_xi * cos_t + u_yi * sin_t
        u_xloc_j = u_xj * cos_t + u_yj * sin_t

        x_loc, dy_vals = miembro.deflection_array("dy", n_points=300)
        ux_loc = u_xloc_i + (x_loc / L_elem) * (u_xloc_j - u_xloc_i)
        uy_loc = dy_vals

        u_X_global = ux_loc * cos_t - uy_loc * sin_t
        u_Y_global = ux_loc * sin_t + uy_loc * cos_t

        mag_disp = np.sqrt(u_X_global**2 + u_Y_global**2)
        local_max = np.max(mag_disp)
        if local_max > max_disp_global:
            max_disp_global = local_max

        datos_deformada.append({
            'nombre': nombre_m, 'xi': xi, 'yi': yi, 'xj': xj, 'yj': yj,
            'cos_t': cos_t, 'sin_t': sin_t, 'x_loc': x_loc,
            'u_X': u_X_global, 'u_Y': u_Y_global, 'mag_disp': mag_disp
        })

    escala_def = (0.12 * L_max) / max_disp_global if max_disp_global > 1e-9 else 1.0

    pad = 0.30 * L_max
    fig.add_trace(go.Scatter(
        x=[min(xs) - pad, max(xs) + pad], y=[min(ys) - pad, max(ys) + pad],
        mode='markers', marker=dict(size=0.1, opacity=0), hoverinfo='skip', showlegend=False
    ))

    for elem in datos_deformada:
        fig.add_trace(go.Scatter(
            x=[elem['xi'], elem['xj']], y=[elem['yi'], elem['yj']],
            mode='lines', line=dict(color='#475569', width=2, dash='dash'),
            hoverinfo='skip', showlegend=False
        ))

        x_base = elem['xi'] + elem['x_loc'] * elem['cos_t']
        y_base = elem['yi'] + elem['x_loc'] * elem['sin_t']
        x_def = x_base + escala_def * elem['u_X']
        y_def = y_base + escala_def * elem['u_Y']

        hover_t = [
            f"<b>{elem['nombre']}</b><br>x: {elem['x_loc'][k]:.2f} m<br>"
            f"<b>δX:</b> {elem['u_X'][k]*1000:.3f} mm<br>"
            f"<b>δY:</b> {elem['u_Y'][k]*1000:.3f} mm<br>"
            f"<b>Desplazamiento total:</b> {elem['mag_disp'][k]*1000:.3f} mm"
            for k in range(len(elem['x_loc']))
        ]

        fig.add_trace(go.Scatter(x=x_def, y=y_def, mode='lines', line=dict(color='#c084fc', width=3.5), text=hover_t, hoverinfo='text', name=elem['nombre']))

    nodos_def_x, nodos_def_y, nodos_text = [], [], []
    for n_id, (x0, y0) in nodos_dict.items():
        node_obj = modelo.nodes[n_id]
        dx_n = extraer_disp_nodo(node_obj, 'DX')
        dy_n = extraer_disp_nodo(node_obj, 'DY')
        x_n_def = x0 + escala_def * dx_n
        y_n_def = y0 + escala_def * dy_n
        nodos_def_x.append(x_n_def)
        nodos_def_y.append(y_n_def)
        nodos_text.append(f"<b>Nodo {n_id}</b><br>δX: {dx_n*1000:.3f} mm<br>δY: {dy_n*1000:.3f} mm")

    fig.add_trace(go.Scatter(
        x=nodos_def_x, y=nodos_def_y, mode='markers+text',
        marker=dict(size=8, color='#e879f9', line=dict(color='#ffffff', width=1.5)),
        text=[f"<b>{n_id}'</b>" for n_id in nodos_dict.keys()], textposition="top right",
        hovertext=nodos_text, hoverinfo='text', showlegend=False
    ))

    fig.update_layout(
        title="<b>Deformada Elástica Continua (Escala Amplificada)</b>",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        margin=dict(l=20, r=20, t=50, b=20), height=520, showlegend=False
    )
    return fig

# ==========================================================
# SECCIÓN 3: CÁLCULO Y RESULTADOS
# ==========================================================
st.markdown("---")
st.markdown('<div class="step-badge">3. Solución y Diagramas de Esfuerzos</div>', unsafe_allow_html=True)

btn_ejecutar = st.button("🖩 Calcular y Generar Diagramas (M, Q y N)", type="primary", use_container_width=True, key="btn_calcular_master")

if btn_ejecutar or "modelo_calculado" in st.session_state:
    if len(datos.get("nodos", [])) < 2 or len(datos.get("barras", [])) < 1:
        st.warning("Se requieren al menos 2 nodos y 1 barra para ejecutar el análisis estructural.")
    else:
        with st.spinner("Ensamblando matrices de rigidez y resolviendo equilibrio estático..."):
            try:
                modelo_resuelto, factor_fuerza = resolver_modelo(datos)
                st.session_state.modelo_calculado = True

                unidad_f = datos.get("unidad_fuerza", "kN")
                unidad_m = f"{unidad_f}·m"

                miembros_reales = [m for k, m in modelo_resuelto.members.items() if not k.startswith("link_")]

                max_m = max([np.max(np.abs(m.moment_array("Mz", n_points=200)[1])) for m in miembros_reales] or [0.0]) / factor_fuerza
                max_q = max([np.max(np.abs(m.shear_array("Fy", n_points=200)[1])) for m in miembros_reales] or [0.0]) / factor_fuerza
                max_n = max([np.max(np.abs(m.axial_array(n_points=200)[1])) for m in miembros_reales] or [0.0]) / factor_fuerza

                col_k1, col_k2, col_k3 = st.columns(3)
                with col_k1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Momento Máximo |M|</div>
                        <div class="metric-value" style="color: #ef4444;">{max_m:.2f} <span style="font-size:0.85rem;">{unidad_m}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_k2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Corte Máximo |Q|</div>
                        <div class="metric-value" style="color: #3b82f6;">{max_q:.2f} <span style="font-size:0.85rem;">{unidad_f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_k3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Normal Máximo |N|</div>
                        <div class="metric-value" style="color: #10b981;">{max_n:.2f} <span style="font-size:0.85rem;">{unidad_f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                tab_m, tab_q, tab_n, tab_def, tab_reac = st.tabs([
                    "🔴 Momento Flector (M)",
                    "🔵 Esfuerzo de Corte (Q)",
                    "🟢 Esfuerzo Normal (N)",
                    "🟣 Deformada Elástica",
                    "🟠 Reacciones de Vínculo"
                ])

                with tab_m:
                    st.plotly_chart(construir_diagrama_plotly(modelo_resuelto, datos, "M", factor_fuerza), use_container_width=True)
                with tab_q:
                    st.plotly_chart(construir_diagrama_plotly(modelo_resuelto, datos, "Q", factor_fuerza), use_container_width=True)
                with tab_n:
                    st.plotly_chart(construir_diagrama_plotly(modelo_resuelto, datos, "N", factor_fuerza), use_container_width=True)
                with tab_def:
                    st.plotly_chart(construir_deformada_plotly(modelo_resuelto, datos), use_container_width=True)
                with tab_reac:
                    st.plotly_chart(construir_diagrama_reacciones(modelo_resuelto, datos, factor_fuerza), use_container_width=True)

            except Exception as e:
                err_msg = str(e).lower()
                if "singular" in err_msg or "unstable" in err_msg or "zero division" in err_msg:
                    st.error("⚠️ **Estructura cinemáticamente inestable (Mecanismo hipostático):** La estructura carece de vínculos suficientes para impedir movimientos de cuerpo rígido. Verificá los apoyos o la presencia de rótulas consecutivas.")
                else:
                    st.error(f"Error durante el cálculo estructural: {str(e)}")