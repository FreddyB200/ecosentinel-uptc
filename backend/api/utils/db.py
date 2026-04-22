"""
Servicio de datos para consultas de consumo energetico.

Carga el dataset limpio (CSV) en memoria al primer acceso y expone
funciones de consulta que los routers usan directamente.
El dataset completo (~275k filas) ocupa ~80MB en RAM, aceptable
para un servidor con 16GB.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from api.config import DATASET_FILE, SECTOR_COLUMNS

logger = logging.getLogger("ecosentinel.db")

# Dataset en memoria (singleton)
_df: Optional[pd.DataFrame] = None


def _get_df() -> pd.DataFrame:
    """Carga el dataset limpio en memoria (una sola vez)."""
    global _df
    if _df is None:
        logger.info(f"Cargando dataset: {DATASET_FILE}")
        _df = pd.read_csv(DATASET_FILE, parse_dates=["timestamp"])
        logger.info(f"Dataset cargado: {len(_df):,} registros")
    return _df


def get_consumos(
    sede: str,
    sector: str,
    fecha_inicio: Optional[datetime] = None,
    fecha_fin: Optional[datetime] = None,
    limit: int = 2000,
) -> pd.DataFrame:
    """
    Consulta consumos historicos filtrados por sede y sector.

    Retorna un DataFrame con columnas 'timestamp' y 'energia_kwh'
    que es el formato que esperan los routers y la generacion de graficos.
    """
    df = _get_df()
    col = SECTOR_COLUMNS.get(sector)
    if col is None:
        return pd.DataFrame()

    mask = df["sede"] == sede

    if fecha_fin is None:
        fecha_fin = df["timestamp"].max()
    if fecha_inicio is None:
        fecha_inicio = fecha_fin - timedelta(days=30)

    mask = mask & (df["timestamp"] >= pd.Timestamp(fecha_inicio))
    mask = mask & (df["timestamp"] <= pd.Timestamp(fecha_fin))

    result = df.loc[mask, ["timestamp", col, "temperatura_exterior_c",
                           "ocupacion_pct", "es_fin_semana"]].copy()
    result = result.rename(columns={col: "energia_kwh"})
    result = result.sort_values("timestamp").tail(limit)

    return result.reset_index(drop=True)


def get_consumos_resumen(sede: str, sector: str, dias: int = 7) -> dict:
    """
    Calcula resumen estadistico del consumo reciente.

    Usa los ultimos N dias del dataset (no desde datetime.now()
    porque el dataset llega hasta oct 2025).
    """
    df = _get_df()
    fecha_fin = df["timestamp"].max()
    fecha_inicio = fecha_fin - timedelta(days=dias)

    consumos = get_consumos(sede, sector, fecha_inicio, fecha_fin, limit=5000)

    if consumos.empty:
        return {"promedio": 0, "maximo": 0, "minimo": 0, "total": 0, "registros": 0}

    return {
        "promedio": round(consumos["energia_kwh"].mean(), 4),
        "maximo": round(consumos["energia_kwh"].max(), 4),
        "minimo": round(consumos["energia_kwh"].min(), 4),
        "total": round(consumos["energia_kwh"].sum(), 2),
        "registros": len(consumos),
    }


def get_consumos_all_sectors(sede: str, dias: int = 7) -> dict[str, pd.DataFrame]:
    """Retorna consumos de todos los sectores de una sede."""
    df = _get_df()
    fecha_fin = df["timestamp"].max()
    fecha_inicio = fecha_fin - timedelta(days=dias)

    result = {}
    for sector in SECTOR_COLUMNS:
        df_sector = get_consumos(sede, sector, fecha_inicio, fecha_fin, limit=5000)
        if not df_sector.empty:
            result[sector] = df_sector
    return result


def get_hourly_profiles(sede: str) -> pd.DataFrame:
    """
    Calcula perfiles horarios promedio de temperatura y ocupacion para una sede.
    Util para estimar valores futuros de regresores en predicciones.
    Agrupa por hora del dia y tipo de dia (laboral vs fin de semana).
    """
    df = _get_df()
    df_sede = df[df["sede"] == sede].copy()
    if df_sede.empty:
        return pd.DataFrame()

    df_sede["hora"] = df_sede["timestamp"].dt.hour

    profiles = df_sede.groupby(["hora", "es_fin_semana"]).agg(
        temperatura_media=("temperatura_exterior_c", "mean"),
        ocupacion_media=("ocupacion_pct", "mean"),
    ).reset_index()

    return profiles


def get_historical_for_chart(
    sede: str,
    sector: str,
    dias: int = 7,
) -> pd.DataFrame:
    """
    Obtiene datos historicos recientes para graficar junto con predicciones.
    Retorna las ultimas N dias de datos.
    """
    df = _get_df()
    fecha_fin = df["timestamp"].max()
    fecha_inicio = fecha_fin - timedelta(days=dias)
    return get_consumos(sede, sector, fecha_inicio, fecha_fin, limit=5000)
