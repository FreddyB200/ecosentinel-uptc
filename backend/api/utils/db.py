"""
Cliente de Supabase para operaciones con la base de datos.
Provee funciones de consulta para consumos, predicciones, anomalías y recomendaciones.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from supabase import create_client, Client

from api.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger("ecosentinel.db")

_client: Optional[Client] = None


def get_client() -> Client:
    """Obtiene o crea la instancia del cliente Supabase."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL y SUPABASE_KEY deben estar configurados")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Cliente Supabase inicializado")
    return _client


def get_consumos(
    sede: str,
    sector: str,
    fecha_inicio: Optional[datetime] = None,
    fecha_fin: Optional[datetime] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Consulta consumos históricos filtrados por sede y sector.

    Args:
        sede: Nombre de la sede (Tunja, Duitama, Sogamoso, Chiquinquirá)
        sector: Nombre del sector (Comedores, Salones, etc.)
        fecha_inicio: Filtro fecha inicio (default: últimos 30 días)
        fecha_fin: Filtro fecha fin (default: ahora)
        limit: Máximo de registros a retornar

    Returns:
        DataFrame con los consumos encontrados
    """
    client = get_client()

    if fecha_fin is None:
        fecha_fin = datetime.now()
    if fecha_inicio is None:
        fecha_inicio = fecha_fin - timedelta(days=30)

    query = (
        client.table("consumos")
        .select("*")
        .eq("sede", sede)
        .eq("sector", sector)
        .gte("timestamp", fecha_inicio.isoformat())
        .lte("timestamp", fecha_fin.isoformat())
        .order("timestamp", desc=False)
        .limit(limit)
    )

    response = query.execute()
    if response.data:
        return pd.DataFrame(response.data)
    return pd.DataFrame()


def get_consumos_resumen(sede: str, sector: str, dias: int = 7) -> dict:
    """
    Calcula resumen estadístico del consumo reciente.

    Args:
        sede: Nombre de la sede
        sector: Nombre del sector
        dias: Cantidad de días hacia atrás para el resumen

    Returns:
        Diccionario con promedio, máximo, mínimo y total
    """
    df = get_consumos(
        sede=sede,
        sector=sector,
        fecha_inicio=datetime.now() - timedelta(days=dias),
    )

    if df.empty:
        return {"promedio": 0, "maximo": 0, "minimo": 0, "total": 0, "registros": 0}

    return {
        "promedio": round(df["energia_kwh"].mean(), 2),
        "maximo": round(df["energia_kwh"].max(), 2),
        "minimo": round(df["energia_kwh"].min(), 2),
        "total": round(df["energia_kwh"].sum(), 2),
        "registros": len(df),
    }


def save_predicciones(predicciones: list[dict]) -> bool:
    """
    Guarda predicciones generadas en la tabla predicciones.

    Args:
        predicciones: Lista de diccionarios con sede, sector, fecha_prediccion,
                      energia_predicha_kwh y confianza

    Returns:
        True si se guardaron correctamente
    """
    try:
        client = get_client()
        client.table("predicciones").insert(predicciones).execute()
        logger.info(f"Guardadas {len(predicciones)} predicciones")
        return True
    except Exception as e:
        logger.error(f"Error guardando predicciones: {e}")
        return False


def save_anomalias(anomalias: list[dict]) -> bool:
    """Guarda anomalías detectadas en la tabla anomalias."""
    try:
        client = get_client()
        client.table("anomalias").insert(anomalias).execute()
        logger.info(f"Guardadas {len(anomalias)} anomalías")
        return True
    except Exception as e:
        logger.error(f"Error guardando anomalías: {e}")
        return False


def save_recomendaciones(recomendaciones: list[dict]) -> bool:
    """Guarda recomendaciones generadas en la tabla recomendaciones."""
    try:
        client = get_client()
        client.table("recomendaciones").insert(recomendaciones).execute()
        logger.info(f"Guardadas {len(recomendaciones)} recomendaciones")
        return True
    except Exception as e:
        logger.error(f"Error guardando recomendaciones: {e}")
        return False


def get_anomalias_recientes(sede: str, sector: str, dias: int = 7) -> list[dict]:
    """Obtiene anomalías detectadas en los últimos N días."""
    client = get_client()
    fecha_inicio = (datetime.now() - timedelta(days=dias)).isoformat()

    response = (
        client.table("anomalias")
        .select("*")
        .eq("sede", sede)
        .eq("sector", sector)
        .gte("timestamp", fecha_inicio)
        .order("timestamp", desc=True)
        .limit(50)
        .execute()
    )
    return response.data or []
