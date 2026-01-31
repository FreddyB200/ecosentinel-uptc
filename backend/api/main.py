"""
EcoSentinel UPTC - API Principal
Sistema de predicción y optimización energética para la UPTC.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import chat, predictions, consumption, anomalies, recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecosentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos al arrancar y los libera al cerrar."""
    logger.info("Iniciando EcoSentinel API...")

    # Asegurar que existan directorios necesarios
    os.makedirs("static/charts", exist_ok=True)

    # Cargar modelos Prophet en memoria al iniciar
    try:
        from api.ml.predictor import model_manager
        loaded = model_manager.load_all_models()
        logger.info(f"Modelos cargados: {loaded}")
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}. La API arranca sin predicciones.")

    yield
    logger.info("Cerrando EcoSentinel API...")


app = FastAPI(
    title="EcoSentinel UPTC API",
    description="Predicción, detección de anomalías y recomendaciones energéticas para la UPTC",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir gráficos generados como archivos estáticos
app.mount("/charts", StaticFiles(directory="static/charts"), name="charts")

# Registrar routers con prefijo /api (para dashboard Streamlit)
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(predictions.router, prefix="/api", tags=["Predicciones"])
app.include_router(consumption.router, prefix="/api", tags=["Consumo"])
app.include_router(anomalies.router, prefix="/api", tags=["Anomalías"])
app.include_router(recommendations.router, prefix="/api", tags=["Recomendaciones"])

# Registrar /predict también en raíz para n8n (llama a POST /predict directamente)
app.include_router(predictions.router, tags=["n8n"])


@app.get("/health")
def health_check():
    """Verificación de estado del servicio."""
    from api.ml.predictor import model_manager
    models_loaded = len(model_manager._models)
    return {
        "status": "healthy",
        "service": "ecosentinel-api",
        "version": "1.0.0",
        "models_loaded": models_loaded,
    }
