"""
Wrapper del modelo Prophet para predicciones energéticas.
Gestiona la carga de 20 modelos (4 sedes x 5 sectores) y genera predicciones multi-horizonte.
"""

import logging
import os
import pickle
from typing import Optional

import pandas as pd
import numpy as np
from prophet import Prophet

from api.config import MODELS_DIR, SEDES, SECTORES, SECTOR_COLUMNS

logger = logging.getLogger("ecosentinel.predictor")


class ModelManager:
    """Gestor de modelos Prophet. Carga y mantiene modelos en memoria."""

    def __init__(self):
        self._models: dict[str, Prophet] = {}

    def _model_key(self, sede: str, sector: str) -> str:
        """Genera clave única para un modelo sede+sector."""
        return f"{sede}_{sector}"

    def _model_path(self, sede: str, sector: str) -> str:
        """Ruta al archivo .pkl del modelo."""
        return os.path.join(MODELS_DIR, f"{self._model_key(sede, sector)}.pkl")

    def load_all_models(self) -> int:
        """
        Carga todos los modelos entrenados desde disco.

        Returns:
            Cantidad de modelos cargados exitosamente
        """
        loaded = 0
        for sede in SEDES:
            for sector in SECTORES:
                path = self._model_path(sede, sector)
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            self._models[self._model_key(sede, sector)] = pickle.load(f)
                        loaded += 1
                    except Exception as e:
                        logger.warning(f"Error cargando modelo {sede}/{sector}: {e}")
                else:
                    logger.debug(f"Modelo no encontrado: {path}")

        logger.info(f"Modelos cargados: {loaded}/20")
        return loaded

    def get_model(self, sede: str, sector: str) -> Optional[Prophet]:
        """Obtiene un modelo cargado por sede y sector."""
        return self._models.get(self._model_key(sede, sector))

    def predict(
        self,
        sede: str,
        sector: str,
        hours_ahead: int = 24,
        temperatura: float = 18.0,
        ocupacion: float = 60.0,
    ) -> Optional[pd.DataFrame]:
        """
        Genera predicciones para una sede y sector dados.

        Args:
            sede: Nombre de la sede
            sector: Nombre del sector
            hours_ahead: Horas hacia el futuro a predecir (1-168)
            temperatura: Temperatura exterior estimada (°C)
            ocupacion: Porcentaje de ocupación estimado (0-100)

        Returns:
            DataFrame con columnas ds, yhat, yhat_lower, yhat_upper o None si no hay modelo
        """
        model = self.get_model(sede, sector)
        if model is None:
            logger.warning(f"No hay modelo para {sede}/{sector}")
            return None

        hours_ahead = max(1, min(hours_ahead, 168))  # Limitar a 7 días

        # Crear dataframe futuro
        future = model.make_future_dataframe(periods=hours_ahead, freq="h")
        future = future.tail(hours_ahead)  # Solo las horas futuras

        # Agregar regresores
        future["temperatura_exterior_c"] = temperatura
        future["ocupacion_pct"] = ocupacion
        future["es_fin_semana"] = future["ds"].dt.dayofweek.isin([5, 6]).astype(int)

        forecast = model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        # Asegurar valores no negativos
        result["yhat"] = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

        logger.info(f"Predicción generada: {sede}/{sector}, {hours_ahead}h")
        return result


def train_model(
    df: pd.DataFrame,
    sede: str,
    sector: str,
    save: bool = True,
) -> Prophet:
    """
    Entrena un modelo Prophet para una sede y sector específicos.

    Args:
        df: DataFrame con el dataset completo
        sede: Nombre de la sede a filtrar
        sector: Nombre del sector (columna del CSV)
        save: Si True, guarda el modelo entrenado en disco

    Returns:
        Modelo Prophet entrenado
    """
    col = SECTOR_COLUMNS.get(sector)
    if col is None:
        raise ValueError(f"Sector desconocido: {sector}")

    # Filtrar por sede
    df_sede = df[df["sede"] == sede].copy()
    if df_sede.empty:
        raise ValueError(f"No hay datos para la sede: {sede}")

    # Preparar formato Prophet
    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(df_sede["timestamp"]),
        "y": df_sede[col].astype(float),
        "temperatura_exterior_c": df_sede["temperatura_exterior_c"].astype(float),
        "ocupacion_pct": df_sede["ocupacion_pct"].astype(float),
        "es_fin_semana": df_sede["es_fin_semana"].astype(int),
    }).dropna()

    logger.info(f"Entrenando modelo {sede}/{sector} con {len(df_prophet)} registros...")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95,
        growth="linear",
    )
    model.add_regressor("temperatura_exterior_c")
    model.add_regressor("ocupacion_pct")
    model.add_regressor("es_fin_semana")

    model.fit(df_prophet)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{sede}_{sector}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Modelo guardado: {path}")

    return model


# Instancia global del gestor de modelos
model_manager = ModelManager()
