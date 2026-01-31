"""
Wrapper del modelo Prophet para predicciones energéticas.
Gestiona la carga de 20 modelos (4 sedes x 5 sectores) y genera predicciones multi-horizonte.
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from prophet import Prophet

from api.config import MODELS_DIR, SEDES, SECTORES, SECTOR_COLUMNS

logger = logging.getLogger("ecosentinel.predictor")

# Metricas de entrenamiento cargadas desde CSV
_training_metrics: Optional[pd.DataFrame] = None


def _load_training_metrics() -> pd.DataFrame:
    """Carga metricas de entrenamiento desde CSV (una sola vez)."""
    global _training_metrics
    if _training_metrics is None:
        path = os.path.join(os.path.dirname(MODELS_DIR), "training_results.csv")
        if os.path.exists(path):
            _training_metrics = pd.read_csv(path)
            logger.info(f"Metricas de entrenamiento cargadas: {len(_training_metrics)} registros")
        else:
            logger.warning(f"Archivo de metricas no encontrado: {path}")
            _training_metrics = pd.DataFrame()
    return _training_metrics


def get_model_metrics(sede: str, sector: str) -> dict:
    """Obtiene metricas de entrenamiento para un modelo sede+sector."""
    df = _load_training_metrics()
    if df.empty:
        return {}

    row = df[(df["sede"] == sede) & (df["sector"] == sector)]
    if row.empty:
        return {}

    row = row.iloc[0]
    return {
        "mae": round(float(row.get("mae", 0)), 4),
        "r2": round(float(row.get("r2", 0)), 4),
        "mape": round(float(row.get("mape", 0)), 2),
    }


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
        Genera predicciones simples con valores fijos de regresores.
        Usado por el dashboard de Streamlit.
        """
        model = self.get_model(sede, sector)
        if model is None:
            logger.warning(f"No hay modelo para {sede}/{sector}")
            return None

        hours_ahead = max(1, min(hours_ahead, 168))

        future = model.make_future_dataframe(periods=hours_ahead, freq="h")
        future = future.tail(hours_ahead)

        future["temperatura_exterior_c"] = temperatura
        future["ocupacion_pct"] = ocupacion
        future["es_fin_semana"] = future["ds"].dt.dayofweek.isin([5, 6]).astype(int)
        future["es_festivo"] = 0

        forecast = model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        result["yhat"] = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

        logger.info(f"Prediccion generada: {sede}/{sector}, {hours_ahead}h")
        return result

    def predict_from_datetime(
        self,
        sede: str,
        sector: str,
        fecha_inicio: datetime,
        horas: int = 24,
        hourly_profiles: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Genera predicciones desde una fecha especifica, estimando temperatura
        y ocupacion por hora a partir de perfiles historicos del dataset.

        Prophet puede predecir para cualquier timestamp: no necesita que
        las fechas sean consecutivas al entrenamiento. Los componentes de
        tendencia, estacionalidad y regresores se evaluan independientemente.

        Args:
            sede: Nombre de la sede
            sector: Nombre del sector
            fecha_inicio: Datetime desde donde empezar la prediccion
            horas: Cantidad de horas a predecir (1-168)
            hourly_profiles: DataFrame con perfiles horarios (de get_hourly_profiles)

        Returns:
            DataFrame con ds, yhat, yhat_lower, yhat_upper,
            temperatura_estimada, ocupacion_estimada
        """
        model = self.get_model(sede, sector)
        if model is None:
            logger.warning(f"No hay modelo para {sede}/{sector}")
            return None

        horas = max(1, min(horas, 168))

        # Generar timestamps desde fecha_inicio
        timestamps = pd.date_range(start=fecha_inicio, periods=horas, freq="h")
        future = pd.DataFrame({"ds": timestamps})

        future["es_fin_semana"] = future["ds"].dt.dayofweek.isin([5, 6]).astype(int)
        future["es_festivo"] = 0
        future["hora"] = future["ds"].dt.hour

        # Estimar temperatura y ocupacion con perfiles historicos
        if hourly_profiles is not None and not hourly_profiles.empty:
            temp_values = []
            ocup_values = []
            for _, row in future.iterrows():
                profile = hourly_profiles[
                    (hourly_profiles["hora"] == row["hora"]) &
                    (hourly_profiles["es_fin_semana"] == row["es_fin_semana"])
                ]
                if not profile.empty:
                    temp_values.append(round(profile.iloc[0]["temperatura_media"], 1))
                    ocup_values.append(round(profile.iloc[0]["ocupacion_media"], 1))
                else:
                    temp_values.append(14.0)
                    ocup_values.append(30.0)
            future["temperatura_exterior_c"] = temp_values
            future["ocupacion_pct"] = ocup_values
        else:
            future["temperatura_exterior_c"] = 14.0
            future["ocupacion_pct"] = 30.0

        # Guardar estimaciones para incluirlas en la respuesta
        temp_est = future["temperatura_exterior_c"].values.copy()
        ocup_est = future["ocupacion_pct"].values.copy()

        # Limpiar columna auxiliar antes de pasar a Prophet
        future = future.drop(columns=["hora"])

        forecast = model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        result["yhat"] = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

        result["temperatura_estimada"] = temp_est
        result["ocupacion_estimada"] = ocup_est

        logger.info(f"Prediccion datetime generada: {sede}/{sector}, "
                     f"{fecha_inicio} +{horas}h")
        return result


def train_model(
    df: pd.DataFrame,
    sede: str,
    sector: str,
    save: bool = True,
) -> tuple[Prophet, pd.DataFrame]:
    """
    Entrena un modelo Prophet para una sede y sector especificos.

    Args:
        df: DataFrame con el dataset completo (consumos_uptc_clean.csv)
        sede: Nombre de la sede a filtrar
        sector: Nombre del sector
        save: Si True, guarda el modelo entrenado en disco

    Returns:
        Tupla (modelo_entrenado, dataframe_prophet) para evaluacion posterior
    """
    col = SECTOR_COLUMNS.get(sector)
    if col is None:
        raise ValueError(f"Sector desconocido: {sector}")

    # Filtrar por sede
    df_sede = df[df["sede"] == sede].copy()
    if df_sede.empty:
        raise ValueError(f"No hay datos para la sede: {sede}")

    # Preparar formato Prophet (ds=timestamp, y=variable objetivo)
    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(df_sede["timestamp"]),
        "y": df_sede[col].astype(float),
        "temperatura_exterior_c": df_sede["temperatura_exterior_c"].astype(float),
        "ocupacion_pct": df_sede["ocupacion_pct"].astype(float),
        "es_fin_semana": df_sede["es_fin_semana"].astype(int),
        "es_festivo": df_sede["es_festivo"].astype(int),
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
    model.add_regressor("es_festivo")

    model.fit(df_prophet)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{sede}_{sector}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Modelo guardado: {path}")

    return model, df_prophet


# Instancia global del gestor de modelos
model_manager = ModelManager()
