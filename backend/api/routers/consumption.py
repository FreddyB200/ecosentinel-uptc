"""
Router para consultas de consumo histórico.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.config import SEDES, SECTORES, CO2_FACTOR, PRECIO_KWH_COP
from api.utils.db import get_consumos, get_consumos_resumen
from api.utils.charts import generate_consumption_chart, generate_comparison_chart

logger = logging.getLogger("ecosentinel.consumption")
router = APIRouter()


class ConsumptionSummary(BaseModel):
    """Resumen de consumo para una sede/sector."""
    sede: str
    sector: str
    dias: int
    promedio_kwh: float
    maximo_kwh: float
    minimo_kwh: float
    total_kwh: float
    total_co2_kg: float
    costo_estimado_cop: float
    registros: int
    chart_url: Optional[str] = None


@router.get("/consumption", response_model=ConsumptionSummary)
async def get_consumption(
    sede: str = Query(..., description="Sede UPTC"),
    sector: str = Query(..., description="Sector"),
    dias: int = Query(7, ge=1, le=365, description="Días hacia atrás"),
    include_chart: bool = Query(True, description="Incluir gráfico"),
):
    """
    Consulta resumen de consumo histórico por sede y sector.
    """
    if sede not in SEDES:
        raise HTTPException(status_code=400, detail=f"Sede inválida. Opciones: {SEDES}")
    if sector not in SECTORES:
        raise HTTPException(status_code=400, detail=f"Sector inválido. Opciones: {SECTORES}")

    resumen = get_consumos_resumen(sede, sector, dias=dias)

    chart_url = None
    if include_chart and resumen["registros"] > 0:
        df = get_consumos(sede, sector)
        if not df.empty:
            chart_url = generate_consumption_chart(df, sede, sector)

    return ConsumptionSummary(
        sede=sede,
        sector=sector,
        dias=dias,
        promedio_kwh=resumen["promedio"],
        maximo_kwh=resumen["maximo"],
        minimo_kwh=resumen["minimo"],
        total_kwh=resumen["total"],
        total_co2_kg=round(resumen["total"] * CO2_FACTOR, 2),
        costo_estimado_cop=round(resumen["total"] * PRECIO_KWH_COP, 0),
        registros=resumen["registros"],
        chart_url=chart_url,
    )


@router.get("/consumption/compare")
async def compare_consumption(
    sede: str = Query(..., description="Sede UPTC"),
    dias: int = Query(7, ge=1, le=365, description="Días hacia atrás"),
):
    """
    Compara consumo entre todos los sectores de una sede.
    """
    if sede not in SEDES:
        raise HTTPException(status_code=400, detail=f"Sede inválida. Opciones: {SEDES}")

    datos = {}
    detalles = []
    for sector in SECTORES:
        resumen = get_consumos_resumen(sede, sector, dias=dias)
        datos[sector] = resumen["promedio"]
        detalles.append({
            "sector": sector,
            "promedio_kwh": resumen["promedio"],
            "maximo_kwh": resumen["maximo"],
            "total_kwh": resumen["total"],
        })

    chart_url = generate_comparison_chart(datos, titulo=f"Consumo por sector - {sede}")

    return {
        "sede": sede,
        "dias": dias,
        "sectores": detalles,
        "chart_url": chart_url,
    }
