"""
EcoSentinel UPTC - Dashboard Web (Streamlit)
Interfaz secundaria para usuarios técnicos.
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="EcoSentinel UPTC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EcoSentinel UPTC")
st.markdown("Sistema inteligente de predicción y optimización energética")

st.sidebar.title("Navegación")
st.sidebar.markdown("""
- **Consumo**: Visualiza consumo histórico
- **Predicciones**: Genera predicciones con Prophet
- **Anomalías**: Detecta patrones ineficientes
""")

# Health check
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        st.sidebar.success("API conectada")
    else:
        st.sidebar.error("API no disponible")
except requests.exceptions.ConnectionError:
    st.sidebar.error("No se puede conectar a la API")

st.markdown("---")
st.markdown("Selecciona una página en la barra lateral para comenzar.")
