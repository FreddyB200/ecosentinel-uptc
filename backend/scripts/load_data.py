"""
Script para cargar el dataset CSV a Supabase.
Transforma el formato ancho (columna por sector) al formato largo (una fila por registro).

Uso:
    python -m scripts.load_data --csv data/consumos_uptc.csv --batch-size 500
"""

import argparse
import logging
import sys
import os

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_data")

SECTOR_COLUMNS = {
    "Comedores": "energia_comedor_kwh",
    "Salones": "energia_salones_kwh",
    "Laboratorios": "energia_laboratorios_kwh",
    "Auditorios": "energia_auditorios_kwh",
    "Oficinas": "energia_oficinas_kwh",
}


def load_csv(path: str) -> pd.DataFrame:
    """Lee y valida el CSV del dataset."""
    logger.info(f"Leyendo CSV: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"Registros leídos: {len(df)}")
    logger.info(f"Columnas: {list(df.columns)}")
    logger.info(f"Sedes: {df['sede'].unique()}")
    logger.info(f"Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def transform_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el dataset de formato ancho (columna por sector)
    a formato largo (una fila por sede+sector+timestamp).
    """
    rows = []
    for _, row in df.iterrows():
        for sector, col in SECTOR_COLUMNS.items():
            rows.append({
                "timestamp": row["timestamp"].isoformat(),
                "sede": row["sede"],
                "sector": sector,
                "energia_kwh": float(row[col]),
                "temperatura_c": float(row["temperatura_exterior_c"]),
                "ocupacion_pct": float(row["ocupacion_pct"]),
                "co2_kg": round(float(row[col]) * 0.126, 4),
            })

    result = pd.DataFrame(rows)
    logger.info(f"Registros transformados: {len(result)}")
    return result


def upload_to_supabase(df: pd.DataFrame, batch_size: int = 500):
    """Sube datos a Supabase en lotes."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        logger.error("SUPABASE_URL y SUPABASE_KEY requeridos en .env")
        sys.exit(1)

    client = create_client(url, key)
    total = len(df)
    uploaded = 0

    for i in range(0, total, batch_size):
        batch = df.iloc[i:i + batch_size].to_dict(orient="records")
        try:
            client.table("consumos").insert(batch).execute()
            uploaded += len(batch)
            logger.info(f"Progreso: {uploaded}/{total} ({uploaded/total*100:.1f}%)")
        except Exception as e:
            logger.error(f"Error en lote {i}: {e}")
            continue

    logger.info(f"Carga completada: {uploaded}/{total} registros")


def main():
    parser = argparse.ArgumentParser(description="Cargar dataset a Supabase")
    parser.add_argument("--csv", default="data/consumos_uptc.csv", help="Ruta al CSV")
    parser.add_argument("--batch-size", type=int, default=500, help="Tamaño de lote")
    parser.add_argument("--dry-run", action="store_true", help="Solo transformar sin subir")
    args = parser.parse_args()

    df = load_csv(args.csv)
    df_long = transform_to_long(df)

    if args.dry_run:
        logger.info("Modo dry-run: no se subirán datos")
        logger.info(f"\nMuestra:\n{df_long.head(10)}")
        return

    upload_to_supabase(df_long, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
