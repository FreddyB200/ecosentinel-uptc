"""
Router para predicciones energéticas con Prophet.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.config import SEDES, SECTORES, CO2_FACTOR, PRECIO_KWH_COP
from api.ml.predictor import model_manager
from api.utils.db import get_consumos
from api.utils.charts import generate_prediction_chart

logger = logging.getLogger("ecosentinel.predictions")
router = APIRouter()


class PredictRequest(BaseModel):
    """Esquema de solicitud de predicción."""
    sede: str = Field(..., description="Sede UPTC", examples=["Tunja"])
    sector: str = Field(..., description="Sector", examples=["Comedores"])
    hours_ahead: int = Field(24, ge=1, le=168, description="Horas a predecir (1-168)")
    temperatura: float = Field(18.0, description="Temperatura exterior estimada (°C)")
    ocupacion: float = Field(60.0, ge=0, le=100, description="Ocupación estimada (%)")
    include_chart: bool = Field(True, description="Incluir URL del gráfico")


class PredictResponse(BaseModel):
    """Esquema de respuesta de predicción."""
    sede: str
    sector: str
    hours_ahead: int
    predictions: list[float]
    timestamps: list[str]
    confidence_lower: list[float]
    confidence_upper: list[float]
    summary: dict
    chart_url: Optional[str] = None


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    Genera predicción de consumo energético.

    Usa modelo Prophet entrenado para la combinación sede+sector solicitada.
    Retorna valores predichos, intervalos de confianza y opcionalmente un gráfico.
    """
    if request.sede not in SEDES:
        raise HTTPException(status_code=400, detail=f"Sede inválida. Opciones: {SEDES}")
    if request.sector not in SECTORES:
        raise HTTPException(status_code=400, detail=f"Sector inválido. Opciones: {SECTORES}")

    forecast = model_manager.predict(
        sede=request.sede,
        sector=request.sector,
        hours_ahead=request.hours_ahead,
        temperatura=request.temperatura,
        ocupacion=request.ocupacion,
    )

    if forecast is None:
        raise HTTPException(
            status_code=404,
            detail=f"No hay modelo entrenado para {request.sede}/{request.sector}",
        )

    # Generar gráfico si se solicita
    chart_url = None
    if request.include_chart:
        df_hist = get_consumos(request.sede, request.sector)
        if not df_hist.empty:
            chart_url = generate_prediction_chart(df_hist, forecast, request.sede, request.sector)

    total_pred = forecast["yhat"].sum()
    return PredictResponse(
        sede=request.sede,
        sector=request.sector,
        hours_ahead=request.hours_ahead,
        predictions=forecast["yhat"].round(2).tolist(),
        timestamps=forecast["ds"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        confidence_lower=forecast["yhat_lower"].round(2).tolist(),
        confidence_upper=forecast["yhat_upper"].round(2).tolist(),
        summary={
            "promedio_kwh": round(forecast["yhat"].mean(), 2),
            "maximo_kwh": round(forecast["yhat"].max(), 2),
            "minimo_kwh": round(forecast["yhat"].min(), 2),
            "total_kwh": round(total_pred, 2),
            "total_co2_kg": round(total_pred * CO2_FACTOR, 2),
            "costo_estimado_cop": round(total_pred * PRECIO_KWH_COP, 0),
        },
        chart_url=chart_url,
    )
