"""
Página de detección de anomalías - Dashboard EcoSentinel.
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

SEDES = ["Tunja", "Duitama", "Sogamoso", "Chiquinquirá"]
SECTORES = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]

st.title("Detección de Anomalías")

col1, col2, col3 = st.columns(3)
with col1:
    sede = st.selectbox("Sede", SEDES)
with col2:
    sector = st.selectbox("Sector", SECTORES)
with col3:
    threshold = st.slider("Umbral Z-Score", 1.5, 4.0, 2.5, 0.1)

if st.button("Detectar Anomalías", type="primary"):
    with st.spinner("Analizando patrones..."):
        try:
            response = requests.get(
                f"{API_URL}/api/anomalies",
                params={"sede": sede, "sector": sector, "threshold": threshold},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()

                st.metric("Anomalías detectadas", data["total_anomalias"])

                if data.get("chart_url"):
                    st.image(data["chart_url"], use_container_width=True)

                if data.get("patrones"):
                    st.subheader("Patrones ineficientes detectados")
                    for p in data["patrones"]:
                        with st.expander(f"{p['tipo']} - Severidad: {p['severidad']}"):
                            st.write(p["descripcion"])
                            if p.get("ahorro_potencial_kwh"):
                                st.write(f"**Ahorro potencial:** {p['ahorro_potencial_kwh']:,.0f} kWh")

                if data.get("anomalias"):
                    st.subheader("Detalle de anomalías")
                    st.dataframe(data["anomalias"], use_container_width=True)
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("No se puede conectar a la API")
