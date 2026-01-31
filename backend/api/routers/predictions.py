"""
Router para predicciones energéticas con Prophet.

Soporta dos formatos de request/response:
- POST /predict: formato n8n (fecha_inicio + horas_prediccion, respuesta con base64)
- GET /api/predict/simple: formato dashboard (hours_ahead, respuesta con URL)
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.config import SEDES, SECTORES, CO2_FACTOR, PRECIO_KWH_COP
from api.ml.predictor import model_manager, get_model_metrics
from api.utils.db import get_historical_for_chart, get_hourly_profiles
from api.utils.charts import (
    generate_prediction_chart,
    generate_prediction_chart_b64,
)

logger = logging.getLogger("ecosentinel.predictions")
router = APIRouter()


# --- Esquemas para n8n (WhatsApp) ---

class PredictRequest(BaseModel):
    """Request desde n8n: el usuario pide predicción via WhatsApp."""
    sede: str = Field(..., description="Sede UPTC", examples=["Tunja"])
    sector: str = Field(..., description="Sector", examples=["Laboratorios"])
    fecha_inicio: Optional[str] = Field(
        None, description="Inicio de predicción ISO format",
        examples=["2026-01-31T08:00:00"],
    )
    horas_prediccion: int = Field(24, ge=1, le=168, description="Horas a predecir")
    incluir_grafica: bool = Field(True, description="Incluir gráfica en base64")

    # Campos opcionales para compatibilidad con dashboard
    hours_ahead: Optional[int] = Field(None, description="Alias de horas_prediccion")
    temperatura: Optional[float] = Field(None, description="Temperatura fija (override)")
    ocupacion: Optional[float] = Field(None, description="Ocupación fija (override)")


@router.post("/predict")
async def predict_endpoint(request: PredictRequest):
    """
    Genera predicción de consumo energético.

    Formato de respuesta diseñado para el workflow de n8n:
    - predicciones: array con detalle por hora
    - metricas: calidad del modelo (MAE, R2, MAPE)
    - grafica_base64: imagen PNG codificada en base64

    El endpoint estima temperatura y ocupación por hora usando perfiles
    históricos del dataset, lo que produce predicciones más realistas
    que usar valores fijos.
    """
    sede = _normalize_value(request.sede, SEDES)
    sector = _normalize_value(request.sector, SECTORES)

    if sede is None:
        raise HTTPException(status_code=400, detail=f"Sede inválida. Opciones: {SEDES}")
    if sector is None:
        raise HTTPException(status_code=400, detail=f"Sector inválido. Opciones: {SECTORES}")

    horas = request.hours_ahead or request.horas_prediccion

    # Determinar fecha de inicio
    if request.fecha_inicio:
        try:
            fecha_inicio = datetime.fromisoformat(request.fecha_inicio)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="fecha_inicio debe estar en formato ISO (YYYY-MM-DDThh:mm:ss)",
            )
    else:
        fecha_inicio = datetime.now()

    # Si se pasan temperatura/ocupacion fijas, usar predict simple
    if request.temperatura is not None or request.ocupacion is not None:
        forecast = model_manager.predict(
            sede=sede,
            sector=sector,
            hours_ahead=horas,
            temperatura=request.temperatura or 18.0,
            ocupacion=request.ocupacion or 60.0,
        )
        if forecast is None:
            raise HTTPException(
                status_code=404,
                detail=f"No hay modelo entrenado para {sede}/{sector}",
            )

        # Formato simple para dashboard
        total_pred = forecast["yhat"].sum()
        return {
            "sede": sede,
            "sector": sector,
            "hours_ahead": horas,
            "predictions": forecast["yhat"].round(2).tolist(),
            "timestamps": forecast["ds"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "confidence_lower": forecast["yhat_lower"].round(2).tolist(),
            "confidence_upper": forecast["yhat_upper"].round(2).tolist(),
            "summary": {
                "promedio_kwh": round(forecast["yhat"].mean(), 2),
                "maximo_kwh": round(forecast["yhat"].max(), 2),
                "minimo_kwh": round(forecast["yhat"].min(), 2),
                "total_kwh": round(total_pred, 2),
                "total_co2_kg": round(total_pred * CO2_FACTOR, 2),
                "costo_estimado_cop": round(total_pred * PRECIO_KWH_COP, 0),
            },
        }

    # Predicción con perfiles horarios (formato n8n)
    profiles = get_hourly_profiles(sede)
    forecast = model_manager.predict_from_datetime(
        sede=sede,
        sector=sector,
        fecha_inicio=fecha_inicio,
        horas=horas,
        hourly_profiles=profiles,
    )

    if forecast is None:
        raise HTTPException(
            status_code=404,
            detail=f"No hay modelo entrenado para {sede}/{sector}",
        )

    # Construir array de predicciones en formato n8n
    predicciones = []
    for _, row in forecast.iterrows():
        predicciones.append({
            "fecha_hora": row["ds"].strftime("%Y-%m-%dT%H:%M:%S"),
            "consumo_predicho_kwh": round(float(row["yhat"]), 2),
            "intervalo_confianza_inferior": round(float(row["yhat_lower"]), 2),
            "intervalo_confianza_superior": round(float(row["yhat_upper"]), 2),
            "temperatura_estimada": round(float(row["temperatura_estimada"]), 1),
            "ocupacion_estimada": round(float(row["ocupacion_estimada"]), 1),
        })

    # Metricas del modelo
    metricas = get_model_metrics(sede, sector)

    # Grafica base64
    grafica_base64 = None
    if request.incluir_grafica:
        historico = get_historical_for_chart(sede, sector, dias=3)
        grafica_base64 = generate_prediction_chart_b64(
            forecast, sede, sector,
            historico=historico if not historico.empty else None,
        )

    response = {
        "predicciones": predicciones,
        "sede": sede,
        "sector": sector,
        "metricas": metricas,
    }

    if grafica_base64:
        response["grafica_base64"] = grafica_base64

    return response


def _normalize_value(value: str, valid_list: list[str]) -> Optional[str]:
    """Normaliza un valor contra una lista de opciones válidas (case-insensitive)."""
    if not value:
        return None
    for item in valid_list:
        if item.lower() == value.lower().strip():
            return item
    return None
