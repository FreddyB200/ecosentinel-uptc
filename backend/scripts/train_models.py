"""
Entrenamiento y evaluacion de modelos Prophet (4 sedes x 5 sectores = 20).

Divide los datos temporalmente: todo hasta julio 2025 para entrenar,
agosto-octubre 2025 para evaluar. Esto simula prediccion real hacia
el futuro, no evaluacion sobre datos ya vistos.

Uso:
    cd backend
    python3 -m scripts.train_models
    python3 -m scripts.train_models --sede Tunja --sector Comedores
"""

import argparse
import logging
import time
import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

from api.config import SEDES, SECTORES, SECTOR_COLUMNS, DATASET_FILE
from api.ml.predictor import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")

# Fecha de corte para train/test split temporal
# Entrenamos con datos hasta julio 2025, evaluamos con agosto-octubre 2025
TRAIN_CUTOFF = "2025-07-31"


def evaluate_model(model, df_prophet: pd.DataFrame, cutoff: str) -> dict:
    """
    Evalua un modelo Prophet con split temporal.

    El modelo se entreno con datos hasta cutoff, y evaluamos su capacidad
    de predecir los datos posteriores al corte. Esto es mas realista que
    cross-validation aleatoria porque en series temporales siempre
    predecimos hacia el futuro.

    Metricas:
    - MAE: error absoluto medio (en la unidad de la variable, kWh)
    - R2: proporcion de varianza explicada (1.0 = perfecto)
    - MAPE: error porcentual medio (independiente de escala)
    """
    cutoff_dt = pd.Timestamp(cutoff)
    test = df_prophet[df_prophet["ds"] > cutoff_dt].copy()

    if len(test) < 10:
        return {"mae": None, "r2": None, "mape": None, "n_test": len(test)}

    # Generar predicciones sobre el periodo de test
    future = test[["ds", "temperatura_exterior_c", "ocupacion_pct",
                    "es_fin_semana", "es_festivo"]].copy()
    forecast = model.predict(future)

    y_true = test["y"].values
    y_pred = forecast["yhat"].values

    # Evitar division por cero en MAPE
    mask = y_true > 0.01
    if mask.sum() < 10:
        mape = None
    else:
        mape = round(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100, 2)

    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
        "mape": mape,
        "n_test": len(test),
    }


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelos Prophet")
    parser.add_argument("--sede", default=None, help="Entrenar solo esta sede")
    parser.add_argument("--sector", default=None, help="Entrenar solo este sector")
    args = parser.parse_args()

    logger.info(f"Cargando dataset: {DATASET_FILE}")
    df = pd.read_csv(DATASET_FILE, parse_dates=["timestamp"])
    logger.info(f"Registros: {len(df):,}")

    sedes = [args.sede] if args.sede else SEDES
    sectores = [args.sector] if args.sector else SECTORES

    resultados = []
    total = len(sedes) * len(sectores)
    count = 0

    for sede in sedes:
        for sector in sectores:
            count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"[{count}/{total}] {sede} / {sector}")
            logger.info(f"{'='*60}")

            start = time.time()
            try:
                model, df_prophet = train_model(df, sede, sector, save=True)
                elapsed = time.time() - start

                # Evaluar con datos posteriores al corte
                metrics = evaluate_model(model, df_prophet, TRAIN_CUTOFF)
                logger.info(f"Entrenado en {elapsed:.1f}s | n_test={metrics['n_test']}")
                logger.info(f"MAE={metrics['mae']} | R2={metrics['r2']} | MAPE={metrics['mape']}%")

                resultados.append({
                    "sede": sede,
                    "sector": sector,
                    "status": "OK",
                    "tiempo_s": round(elapsed, 1),
                    **metrics,
                })

            except Exception as e:
                logger.error(f"Error: {e}")
                resultados.append({
                    "sede": sede,
                    "sector": sector,
                    "status": f"ERROR: {e}",
                    "tiempo_s": 0,
                })

    # Resumen final
    logger.info(f"\n{'='*60}")
    logger.info("RESUMEN DE ENTRENAMIENTO")
    logger.info(f"{'='*60}")

    df_results = pd.DataFrame(resultados)
    logger.info(f"\n{df_results.to_string(index=False)}")

    ok = df_results[df_results["status"] == "OK"]
    logger.info(f"\nModelos entrenados: {len(ok)}/{total}")

    if len(ok) > 0 and ok["mae"].notna().any():
        logger.info(f"MAE promedio: {ok['mae'].mean():.4f}")
        logger.info(f"R2 promedio: {ok['r2'].mean():.4f}")

    # Guardar resumen como CSV para referencia
    results_path = os.path.join("models", "training_results.csv")
    os.makedirs("models", exist_ok=True)
    df_results.to_csv(results_path, index=False)
    logger.info(f"Resultados guardados: {results_path}")


if __name__ == "__main__":
    main()
