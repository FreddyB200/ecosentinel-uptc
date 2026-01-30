"""
Configuración central del proyecto.
Carga variables de entorno y define constantes globales.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Supabase ---
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

# --- Gemini ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# --- Servidor ---
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8000")

# --- Rutas ---
MODELS_DIR: str = os.getenv("MODELS_DIR", "models/trained")
CHARTS_DIR: str = os.getenv("CHARTS_DIR", "static/charts")
DATA_DIR: str = os.getenv("DATA_DIR", "data")

# --- Constantes del dominio ---
SEDES = ["Tunja", "Duitama", "Sogamoso", "Chiquinquirá"]
SECTORES = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]

SECTOR_COLUMNS = {
    "Comedores": "energia_comedor_kwh",
    "Salones": "energia_salones_kwh",
    "Laboratorios": "energia_laboratorios_kwh",
    "Auditorios": "energia_auditorios_kwh",
    "Oficinas": "energia_oficinas_kwh",
}

# Factor de emisión CO2 para Colombia (kg CO2 / kWh)
CO2_FACTOR = 0.126

# Precio promedio kWh en Colombia (COP)
PRECIO_KWH_COP = 650.0
