"""
Genera dataset sintético realista de consumo energético para las sedes UPTC.
270,000 registros horarios desde 2018-01-01 hasta 2025-10-31.

Uso:
    python -m scripts.generate_dataset --output data/consumos_uptc.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_dataset")

# Perfiles base por sede (kWh promedio por sector)
SEDE_PROFILES = {
    "Tunja": {
        "estudiantes": 18000,
        "base_multiplier": 1.0,
        "temp_media": 13.0,
        "temp_std": 3.0,
    },
    "Duitama": {
        "estudiantes": 5500,
        "base_multiplier": 0.45,
        "temp_media": 16.0,
        "temp_std": 3.5,
    },
    "Sogamoso": {
        "estudiantes": 6000,
        "base_multiplier": 0.50,
        "temp_media": 17.0,
        "temp_std": 4.0,
    },
    "Chiquinquirá": {
        "estudiantes": 2000,
        "base_multiplier": 0.25,
        "temp_media": 18.0,
        "temp_std": 3.0,
    },
}

# Consumo base por sector (kWh) - para Tunja
SECTOR_BASE = {
    "energia_comedor_kwh": 120.0,
    "energia_salones_kwh": 180.0,
    "energia_laboratorios_kwh": 200.0,
    "energia_auditorios_kwh": 80.0,
    "energia_oficinas_kwh": 95.0,
}

# Periodos académicos UPTC
PERIODOS = [
    ("01-20", "06-15", "Primer semestre"),
    ("07-15", "12-05", "Segundo semestre"),
]

VACACIONES_RANGES = [
    ("01-01", "01-19"),
    ("06-16", "07-14"),
    ("12-06", "12-31"),
]


def is_vacation(date: pd.Timestamp) -> bool:
    """Determina si una fecha está en periodo de vacaciones."""
    mmdd = date.strftime("%m-%d")
    for start, end in VACACIONES_RANGES:
        if start <= mmdd <= end:
            return True
    return False


def get_periodo(date: pd.Timestamp) -> str:
    """Determina el periodo académico de una fecha."""
    mmdd = date.strftime("%m-%d")
    for start, end, nombre in PERIODOS:
        if start <= mmdd <= end:
            return nombre
    return "Vacaciones"


def generate_hourly_pattern(hour: int, is_weekend: bool, is_vac: bool) -> float:
    """Genera factor de consumo según hora del día."""
    if is_vac:
        # Vacaciones: consumo reducido
        if 8 <= hour <= 16:
            return 0.3
        return 0.1

    if is_weekend:
        if 9 <= hour <= 14:
            return 0.25
        return 0.08

    # Día laboral normal
    if 6 <= hour <= 7:
        return 0.5
    elif 8 <= hour <= 12:
        return 1.0
    elif 12 <= hour <= 14:
        return 0.85  # Almuerzo
    elif 14 <= hour <= 18:
        return 0.95
    elif 18 <= hour <= 21:
        return 0.4
    else:
        return 0.1


def main():
    parser = argparse.ArgumentParser(description="Generar dataset sintético UPTC")
    parser.add_argument("--output", default="data/consumos_uptc.csv", help="Ruta de salida")
    args = parser.parse_args()

    # Rango temporal
    start_date = pd.Timestamp("2018-01-01 00:00:00")
    end_date = pd.Timestamp("2025-10-31 23:00:00")
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    logger.info(f"Timestamps generados: {len(timestamps)}")

    rng = np.random.default_rng(42)
    all_rows = []

    for sede, profile in SEDE_PROFILES.items():
        logger.info(f"Generando datos para {sede}...")
        multiplier = profile["base_multiplier"]

        for ts in timestamps:
            hour = ts.hour
            is_weekend = ts.dayofweek >= 5
            is_vac = is_vacation(ts)
            periodo = get_periodo(ts)

            hourly_factor = generate_hourly_pattern(hour, is_weekend, is_vac)

            # Temperatura con patrón diario
            temp_daily = profile["temp_media"] + 5 * np.sin((hour - 6) * np.pi / 12)
            # Variación estacional (más frío en julio en Boyacá)
            month_factor = -2 * np.cos((ts.month - 1) * np.pi / 6)
            temp = temp_daily + month_factor + rng.normal(0, profile["temp_std"] * 0.5)
            temp = np.clip(temp, 5, 35)

            # Ocupación
            if is_vac:
                ocupacion = rng.uniform(0, 15)
            elif is_weekend:
                ocupacion = rng.uniform(0, 20) if 9 <= hour <= 14 else rng.uniform(0, 5)
            else:
                if 7 <= hour <= 18:
                    ocupacion = rng.uniform(40, 95)
                else:
                    ocupacion = rng.uniform(0, 10)

            row = {
                "timestamp": ts,
                "sede": sede,
                "es_fin_semana": is_weekend,
                "periodo_academico": periodo,
                "temperatura_exterior_c": round(temp, 1),
                "ocupacion_pct": round(ocupacion, 1),
            }

            # Generar consumo por sector
            for col, base in SECTOR_BASE.items():
                noise = rng.normal(1.0, 0.08)
                # Efecto temperatura (más consumo en extremos)
                temp_effect = 1.0 + 0.01 * abs(temp - 18)
                # Tendencia anual (incremento gradual del 2% anual)
                year_offset = (ts.year - 2018)
                trend = 1.0 + 0.02 * year_offset

                value = base * multiplier * hourly_factor * noise * temp_effect * trend
                # Ocasionalmente inyectar anomalías (~0.5%)
                if rng.random() < 0.005:
                    value *= rng.uniform(2.0, 3.5)

                row[col] = round(max(0, value), 2)

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["sede", "timestamp"]).reset_index(drop=True)
    logger.info(f"Total registros: {len(df)}")
    logger.info(f"Columnas: {list(df.columns)}")

    df.to_csv(args.output, index=False)
    logger.info(f"Dataset guardado: {args.output}")


if __name__ == "__main__":
    main()
