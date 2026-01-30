"""
Página de predicciones - Dashboard EcoSentinel.
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

SEDES = ["Tunja", "Duitama", "Sogamoso", "Chiquinquirá"]
SECTORES = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]

st.title("Predicción de Consumo Energético")

col1, col2 = st.columns(2)
with col1:
    sede = st.selectbox("Sede", SEDES)
    sector = st.selectbox("Sector", SECTORES)
with col2:
    hours = st.slider("Horas a predecir", 1, 168, 24)
    temperatura = st.slider("Temperatura estimada (°C)", 5.0, 35.0, 18.0)
    ocupacion = st.slider("Ocupación estimada (%)", 0.0, 100.0, 60.0)

if st.button("Generar Predicción", type="primary"):
    with st.spinner("Generando predicción..."):
        try:
            response = requests.post(
                f"{API_URL}/api/predict",
                json={
                    "sede": sede,
                    "sector": sector,
                    "hours_ahead": hours,
                    "temperatura": temperatura,
                    "ocupacion": ocupacion,
                    "include_chart": True,
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()

                m1, m2, m3 = st.columns(3)
                summary = data["summary"]
                m1.metric("Promedio esperado", f"{summary['promedio_kwh']:,.1f} kWh")
                m2.metric("Total estimado", f"{summary['total_kwh']:,.1f} kWh")
                m3.metric("Costo estimado", f"${summary['costo_estimado_cop']:,.0f} COP")

                if data.get("chart_url"):
                    st.image(data["chart_url"], use_container_width=True)
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("No se puede conectar a la API")
