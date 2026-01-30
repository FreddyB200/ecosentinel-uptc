"""
Router para detección y consulta de anomalías.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.config import SEDES, SECTORES
from api.utils.db import get_consumos, get_anomalias_recientes
from api.ml.anomaly_detector import generate_anomaly_report
from api.utils.charts import generate_anomaly_chart

logger = logging.getLogger("ecosentinel.anomalies_router")
router = APIRouter()


@router.get("/anomalies")
async def get_anomalies(
    sede: str = Query(..., description="Sede UPTC"),
    sector: str = Query(..., description="Sector"),
    dias: int = Query(30, ge=1, le=365, description="Días hacia atrás"),
    threshold: float = Query(2.5, ge=1.5, le=4.0, description="Umbral Z-Score"),
    include_chart: bool = Query(True, description="Incluir gráfico"),
):
    """
    Detecta anomalías en consumo energético usando Z-Score.

    Retorna anomalías estadísticas y patrones ineficientes detectados,
    junto con ahorro potencial estimado.
    """
    if sede not in SEDES:
        raise HTTPException(status_code=400, detail=f"Sede inválida. Opciones: {SEDES}")
    if sector not in SECTORES:
        raise HTTPException(status_code=400, detail=f"Sector inválido. Opciones: {SECTORES}")

    df = get_consumos(sede, sector, limit=5000)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No hay datos disponibles para {sede}/{sector}",
        )

    reporte = generate_anomaly_report(df, sede, sector, threshold=threshold)

    chart_url = None
    if include_chart and reporte["total_anomalias"] > 0:
        from api.ml.anomaly_detector import detect_zscore_anomalies
        anomalias_df = detect_zscore_anomalies(df, threshold=threshold)
        if not anomalias_df.empty:
            anomalias_idx = anomalias_df.index.tolist()
            chart_url = generate_anomaly_chart(df, anomalias_idx, sede, sector)

    reporte["chart_url"] = chart_url
    return reporte
