"""
Script para entrenar los 20 modelos Prophet (4 sedes x 5 sectores).

Uso:
    python -m scripts.train_models --csv data/consumos_uptc.csv
    python -m scripts.train_models --csv data/consumos_uptc.csv --sede Tunja --sector Comedores
"""

import argparse
import logging
import time

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

from api.config import SEDES, SECTORES
from api.ml.predictor import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")


def evaluate_model(model, df_prophet: pd.DataFrame) -> dict:
    """Evalúa un modelo Prophet con las métricas objetivo."""
    # Usar últimos 30 días como test
    cutoff = df_prophet["ds"].max() - pd.Timedelta(days=30)
    train = df_prophet[df_prophet["ds"] <= cutoff]
    test = df_prophet[df_prophet["ds"] > cutoff]

    if test.empty:
        return {"mae": None, "r2": None, "mape": None}

    future = model.make_future_dataframe(periods=len(test), freq="h")
    future = future.tail(len(test))
    future["temperatura_exterior_c"] = test["temperatura_exterior_c"].values[:len(future)]
    future["ocupacion_pct"] = test["ocupacion_pct"].values[:len(future)]
    future["es_fin_semana"] = future["ds"].dt.dayofweek.isin([5, 6]).astype(int)

    forecast = model.predict(future)
    y_true = test["y"].values[:len(forecast)]
    y_pred = forecast["yhat"].values

    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "r2": round(r2_score(y_true, y_pred), 4),
        "mape": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelos Prophet")
    parser.add_argument("--csv", default="data/consumos_uptc.csv", help="Ruta al CSV")
    parser.add_argument("--sede", default=None, help="Entrenar solo esta sede")
    parser.add_argument("--sector", default=None, help="Entrenar solo este sector")
    args = parser.parse_args()

    logger.info(f"Cargando dataset: {args.csv}")
    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    logger.info(f"Registros: {len(df)}")

    sedes = [args.sede] if args.sede else SEDES
    sectores = [args.sector] if args.sector else SECTORES

    resultados = []
    total = len(sedes) * len(sectores)
    count = 0

    for sede in sedes:
        for sector in sectores:
            count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"[{count}/{total}] Entrenando: {sede} / {sector}")
            logger.info(f"{'='*60}")

            start = time.time()
            try:
                model = train_model(df, sede, sector, save=True)
                elapsed = time.time() - start

                metrics = {"mae": "N/A", "r2": "N/A", "mape": "N/A"}
                logger.info(f"Modelo entrenado en {elapsed:.1f}s")
                logger.info(f"Métricas: MAE={metrics['mae']}, R²={metrics['r2']}, MAPE={metrics['mape']}%")

                resultados.append({
                    "sede": sede,
                    "sector": sector,
                    "status": "OK",
                    "tiempo_s": round(elapsed, 1),
                    **metrics,
                })

            except Exception as e:
                logger.error(f"Error entrenando {sede}/{sector}: {e}")
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

    ok_count = len(df_results[df_results["status"] == "OK"])
    logger.info(f"\nModelos entrenados: {ok_count}/{total}")


if __name__ == "__main__":
    main()
