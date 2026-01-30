"""
Página de consumo histórico - Dashboard EcoSentinel.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

SEDES = ["Tunja", "Duitama", "Sogamoso", "Chiquinquirá"]
SECTORES = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]

st.title("Consumo Energético Histórico")

col1, col2, col3 = st.columns(3)
with col1:
    sede = st.selectbox("Sede", SEDES)
with col2:
    sector = st.selectbox("Sector", SECTORES)
with col3:
    dias = st.slider("Días", 1, 365, 7)

if st.button("Consultar", type="primary"):
    with st.spinner("Consultando datos..."):
        try:
            response = requests.get(
                f"{API_URL}/api/consumption",
                params={"sede": sede, "sector": sector, "dias": dias},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()

                # Métricas
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Promedio", f"{data['promedio_kwh']:,.1f} kWh")
                m2.metric("Máximo", f"{data['maximo_kwh']:,.1f} kWh")
                m3.metric("Total", f"{data['total_kwh']:,.1f} kWh")
                m4.metric("CO2", f"{data['total_co2_kg']:,.1f} kg")

                st.markdown(f"**Costo estimado:** ${data['costo_estimado_cop']:,.0f} COP")

                # Mostrar gráfico si existe
                if data.get("chart_url"):
                    st.image(data["chart_url"], use_container_width=True)
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("No se puede conectar a la API")
