"""
Detección de anomalías en consumo energético.
Usa Z-Score para detectar valores atípicos y patrones ineficientes.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from api.config import CO2_FACTOR

logger = logging.getLogger("ecosentinel.anomalias")


def detect_zscore_anomalies(
    df: pd.DataFrame,
    column: str = "energia_kwh",
    threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Detecta anomalías usando Z-Score.

    Args:
        df: DataFrame con datos de consumo
        column: Columna numérica a analizar
        threshold: Umbral de Z-Score (default 2.5 = ~1.2% datos normales)

    Returns:
        DataFrame con las filas anómalas y columnas adicionales:
        z_score, tipo_anomalia, severidad
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    valores = df[column].astype(float)
    mean = valores.mean()
    std = valores.std()

    if std == 0:
        return pd.DataFrame()

    z_scores = (valores - mean) / std
    mask_anomalia = z_scores.abs() > threshold

    anomalias = df[mask_anomalia].copy()
    anomalias["z_score"] = z_scores[mask_anomalia]
    anomalias["valor_esperado"] = round(mean, 2)

    # Clasificar tipo y severidad
    anomalias["tipo_anomalia"] = np.where(
        anomalias["z_score"] > 0, "consumo_excesivo", "consumo_inusualmente_bajo"
    )
    anomalias["severidad"] = pd.cut(
        anomalias["z_score"].abs(),
        bins=[threshold, 3.0, 4.0, float("inf")],
        labels=["media", "alta", "critica"],
    )

    logger.info(f"Detectadas {len(anomalias)} anomalías (umbral z={threshold})")
    return anomalias


def detect_pattern_anomalies(
    df: pd.DataFrame,
    column: str = "energia_kwh",
) -> list[dict]:
    """
    Detecta patrones anómalos como consumo nocturno elevado o fines de semana.

    Args:
        df: DataFrame con 'timestamp', column, y 'es_fin_semana'

    Returns:
        Lista de patrones anómalos detectados
    """
    patterns = []
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hora"] = df["timestamp"].dt.hour

    # Patrón 1: Consumo nocturno elevado (22:00-06:00)
    nocturno = df[df["hora"].isin(range(22, 24)) | df["hora"].isin(range(0, 6))]
    diurno = df[~df.index.isin(nocturno.index)]

    if not nocturno.empty and not diurno.empty:
        ratio = nocturno[column].mean() / diurno[column].mean()
        if ratio > 0.5:
            patterns.append({
                "tipo": "consumo_nocturno_elevado",
                "severidad": "alta" if ratio > 0.7 else "media",
                "descripcion": (
                    f"Consumo nocturno es {ratio:.0%} del diurno. "
                    f"Promedio noche: {nocturno[column].mean():.1f} kWh, "
                    f"día: {diurno[column].mean():.1f} kWh"
                ),
                "ahorro_potencial_kwh": round(
                    nocturno[column].sum() * 0.3, 2  # 30% reducible
                ),
            })

    # Patrón 2: Consumo fin de semana similar a entre semana
    if "es_fin_semana" in df.columns:
        fds = df[df["es_fin_semana"] == True]
        entre_sem = df[df["es_fin_semana"] == False]

        if not fds.empty and not entre_sem.empty:
            ratio_fds = fds[column].mean() / entre_sem[column].mean()
            if ratio_fds > 0.7:
                patterns.append({
                    "tipo": "consumo_fds_elevado",
                    "severidad": "alta" if ratio_fds > 0.85 else "media",
                    "descripcion": (
                        f"Consumo fin de semana es {ratio_fds:.0%} del entre semana. "
                        f"Se esperaría una reducción significativa."
                    ),
                    "ahorro_potencial_kwh": round(
                        fds[column].sum() * (1 - 0.4), 2  # Debería bajar al 40%
                    ),
                })

    logger.info(f"Detectados {len(patterns)} patrones anómalos")
    return patterns


def generate_anomaly_report(
    df: pd.DataFrame,
    sede: str,
    sector: str,
    column: str = "energia_kwh",
    threshold: float = 2.5,
) -> dict:
    """
    Genera reporte completo de anomalías para una sede/sector.

    Args:
        df: DataFrame con datos de consumo
        sede: Nombre de la sede
        sector: Nombre del sector
        column: Columna a analizar
        threshold: Umbral Z-Score

    Returns:
        Diccionario con anomalías estadísticas, patrones y resumen
    """
    anomalias_df = detect_zscore_anomalies(df, column, threshold)
    patrones = detect_pattern_anomalies(df, column)

    # Formatear anomalías para respuesta
    anomalias_list = []
    for _, row in anomalias_df.iterrows():
        anomalias_list.append({
            "timestamp": str(row.get("timestamp", "")),
            "valor_actual": round(float(row[column]), 2),
            "valor_esperado": float(row["valor_esperado"]),
            "z_score": round(float(row["z_score"]), 2),
            "tipo": row["tipo_anomalia"],
            "severidad": str(row["severidad"]),
        })

    ahorro_total = sum(p.get("ahorro_potencial_kwh", 0) for p in patrones)

    return {
        "sede": sede,
        "sector": sector,
        "total_anomalias": len(anomalias_list),
        "anomalias": anomalias_list[:20],  # Limitar a 20 más recientes
        "patrones": patrones,
        "ahorro_potencial_kwh": round(ahorro_total, 2),
        "ahorro_potencial_co2_kg": round(ahorro_total * CO2_FACTOR, 2),
    }
