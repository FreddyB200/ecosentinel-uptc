"""
Generación de gráficos PNG con Matplotlib.
Los gráficos se guardan en static/charts/ y se sirven como archivos estáticos.
"""

import logging
import os
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para servidores
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from api.config import CHARTS_DIR, BASE_URL

logger = logging.getLogger("ecosentinel.charts")

# Estilo global
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "dark": "#2C3E50",
    "light": "#ECF0F1",
}


def _save_chart(fig: plt.Figure, prefix: str = "chart") -> str:
    """
    Guarda una figura matplotlib como PNG y retorna la URL pública.

    Args:
        fig: Figura de matplotlib
        prefix: Prefijo para el nombre del archivo

    Returns:
        URL pública del gráfico generado
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filename = f"{prefix}_{int(time.time() * 1000)}.png"
    filepath = os.path.join(CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Gráfico guardado: {filepath}")
    return f"{BASE_URL}/charts/{filename}"


def generate_consumption_chart(
    df: pd.DataFrame,
    sede: str,
    sector: str,
    titulo: Optional[str] = None,
) -> str:
    """
    Genera gráfico de consumo histórico.

    Args:
        df: DataFrame con columnas 'timestamp' y 'energia_kwh'
        sede: Nombre de la sede
        sector: Nombre del sector
        titulo: Título personalizado (opcional)

    Returns:
        URL del gráfico generado
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    timestamps = pd.to_datetime(df["timestamp"])
    valores = df["energia_kwh"]

    ax.plot(timestamps, valores, color=COLORS["primary"], linewidth=1.5, alpha=0.8)
    ax.fill_between(timestamps, valores, alpha=0.15, color=COLORS["primary"])

    # Media móvil 24h
    if len(valores) > 24:
        media_movil = valores.rolling(window=24, min_periods=1).mean()
        ax.plot(
            timestamps, media_movil,
            color=COLORS["accent"], linewidth=2, linestyle="--",
            label="Media móvil 24h",
        )

    ax.set_title(
        titulo or f"Consumo Energético - {sede} / {sector}",
        fontsize=14, fontweight="bold", color=COLORS["dark"],
    )
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Energía (kWh)", fontsize=11)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    fig.tight_layout()
    return _save_chart(fig, prefix=f"consumo_{sede}_{sector}")


def generate_prediction_chart(
    historico: pd.DataFrame,
    predicciones: pd.DataFrame,
    sede: str,
    sector: str,
) -> str:
    """
    Genera gráfico de predicción con intervalo de confianza.

    Args:
        historico: DataFrame con 'timestamp' y 'energia_kwh' (datos reales recientes)
        predicciones: DataFrame con 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        sede: Nombre de la sede
        sector: Nombre del sector

    Returns:
        URL del gráfico generado
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Datos históricos
    hist_ts = pd.to_datetime(historico["timestamp"])
    ax.plot(
        hist_ts, historico["energia_kwh"],
        color=COLORS["primary"], linewidth=1.5, label="Histórico",
    )

    # Predicciones
    pred_ts = pd.to_datetime(predicciones["ds"])
    ax.plot(
        pred_ts, predicciones["yhat"],
        color=COLORS["accent"], linewidth=2, label="Predicción",
    )

    # Intervalo de confianza
    ax.fill_between(
        pred_ts,
        predicciones["yhat_lower"],
        predicciones["yhat_upper"],
        alpha=0.2, color=COLORS["accent"], label="Intervalo 95%",
    )

    # Línea divisoria
    ax.axvline(
        x=hist_ts.iloc[-1], color=COLORS["danger"],
        linestyle=":", linewidth=1.5, alpha=0.7,
    )

    ax.set_title(
        f"Predicción Energética - {sede} / {sector}",
        fontsize=14, fontweight="bold", color=COLORS["dark"],
    )
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Energía (kWh)", fontsize=11)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))

    fig.tight_layout()
    return _save_chart(fig, prefix=f"pred_{sede}_{sector}")


def generate_anomaly_chart(
    df: pd.DataFrame,
    anomalias_idx: list,
    sede: str,
    sector: str,
) -> str:
    """
    Genera gráfico con anomalías resaltadas.

    Args:
        df: DataFrame con 'timestamp' y 'energia_kwh'
        anomalias_idx: Índices de las filas que son anomalías
        sede: Nombre de la sede
        sector: Nombre del sector

    Returns:
        URL del gráfico generado
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    timestamps = pd.to_datetime(df["timestamp"])
    valores = df["energia_kwh"]

    ax.plot(timestamps, valores, color=COLORS["primary"], linewidth=1, alpha=0.7)

    # Marcar anomalías
    if anomalias_idx:
        ax.scatter(
            timestamps.iloc[anomalias_idx],
            valores.iloc[anomalias_idx],
            color=COLORS["danger"], s=50, zorder=5,
            label=f"Anomalías ({len(anomalias_idx)})",
        )

    ax.set_title(
        f"Detección de Anomalías - {sede} / {sector}",
        fontsize=14, fontweight="bold", color=COLORS["dark"],
    )
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Energía (kWh)", fontsize=11)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return _save_chart(fig, prefix=f"anomalias_{sede}_{sector}")


def generate_comparison_chart(
    datos: dict[str, float],
    titulo: str = "Comparación por Sede",
) -> str:
    """
    Genera gráfico de barras comparativo entre sedes o sectores.

    Args:
        datos: Diccionario {nombre: valor_kwh}
        titulo: Título del gráfico

    Returns:
        URL del gráfico generado
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    nombres = list(datos.keys())
    valores = list(datos.values())
    colores = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]]

    bars = ax.bar(nombres, valores, color=colores[:len(nombres)], edgecolor="white", linewidth=0.8)

    # Etiquetas en las barras
    for bar, val in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(valores) * 0.02,
            f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title(titulo, fontsize=14, fontweight="bold", color=COLORS["dark"])
    ax.set_ylabel("Energía (kWh)", fontsize=11)

    fig.tight_layout()
    return _save_chart(fig, prefix="comparacion")
