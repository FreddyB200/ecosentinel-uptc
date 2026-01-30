"""
Router para generación de recomendaciones energéticas con LLM.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.config import SEDES, SECTORES
from api.utils.db import get_consumos, get_consumos_resumen
from api.ml.anomaly_detector import generate_anomaly_report
from api.llm.recommender import generate_recommendations

logger = logging.getLogger("ecosentinel.recommendations_router")
router = APIRouter()


class RecommendationRequest(BaseModel):
    """Esquema de solicitud de recomendaciones."""
    sede: str = Field(..., description="Sede UPTC", examples=["Tunja"])
    sector: str = Field(..., description="Sector", examples=["Comedores"])


@router.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Genera recomendaciones personalizadas para reducir consumo energético.

    Usa Gemini para generar recomendaciones accionables basadas en datos
    históricos, anomalías y patrones detectados.
    """
    if request.sede not in SEDES:
        raise HTTPException(status_code=400, detail=f"Sede inválida. Opciones: {SEDES}")
    if request.sector not in SECTORES:
        raise HTTPException(status_code=400, detail=f"Sector inválido. Opciones: {SECTORES}")

    resumen = get_consumos_resumen(request.sede, request.sector)
    df = get_consumos(request.sede, request.sector)

    anomalias_data = []
    patrones_data = []
    if not df.empty:
        reporte = generate_anomaly_report(df, request.sede, request.sector)
        anomalias_data = reporte.get("anomalias", [])
        patrones_data = reporte.get("patrones", [])

    recommendations = await generate_recommendations(
        sede=request.sede,
        sector=request.sector,
        consumo_promedio=resumen.get("promedio", 0),
        consumo_maximo=resumen.get("maximo", 0),
        anomalias=anomalias_data,
        patrones=patrones_data,
    )

    return {
        "sede": request.sede,
        "sector": request.sector,
        "consumo_actual": resumen,
        "recommendations": recommendations,
    }
