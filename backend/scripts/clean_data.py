"""
Limpieza del dataset de consumo energetico UPTC.

Toma consumos_uptc.csv (original de los organizadores) y genera
consumos_uptc_clean.csv con los siguientes tratamientos:

1. Valores negativos en energia → valor absoluto
2. Normalizacion de periodo_academico (inconsistencias de casing)
3. Imputacion de nulos en energia por sector (interpolacion por sede+hora)
4. Imputacion de nulos en temperatura y ocupacion (interpolacion temporal)
5. Imputacion de nulos en agua_litros (mediana por sede+hora)
6. Tratamiento de outliers con IQR (clip, no eliminacion)
7. Recalculo de energia_total_kwh como suma de sectores
8. Recalculo de co2_kg con factor de emision colombiano

Uso:
    cd backend
    python -m scripts.clean_data
"""

import logging
import os

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("clean_data")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "consumos_uptc.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "consumos_uptc_clean.csv")

# Factor de emision CO2 para el SIN colombiano (kg CO2 / kWh)
# Fuente: UPME / XM - factor promedio del sistema interconectado
CO2_FACTOR = 0.126

# Columnas de energia por sector
ENERGY_COLS = [
    "energia_comedor_kwh",
    "energia_salones_kwh",
    "energia_laboratorios_kwh",
    "energia_auditorios_kwh",
    "energia_oficinas_kwh",
]


def load_raw() -> pd.DataFrame:
    """Carga el dataset original."""
    logger.info(f"Cargando {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Columnas: {len(df.columns)}")
    logger.info(f"  Nulos totales: {df.isnull().sum().sum():,}")
    return df


def fix_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 1: Corregir valores negativos en columnas de energia.
    La energia no puede ser negativa, probablemente errores de medicion.
    Tomamos valor absoluto en vez de eliminar para no perder registros.
    """
    cols = ENERGY_COLS + ["energia_total_kwh", "potencia_total_kw", "co2_kg"]
    total_fixed = 0
    for col in cols:
        mask = df[col] < 0
        count = mask.sum()
        if count > 0:
            df.loc[mask, col] = df.loc[mask, col].abs()
            total_fixed += count
            logger.info(f"  {col}: {count} negativos corregidos")
    logger.info(f"  Total negativos corregidos: {total_fixed}")
    return df


def normalize_periodo_academico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 2: Normalizar etiquetas de periodo academico.
    El dataset tiene variantes como SEMESTRE_1, Semestre_1, semestre1
    que representan lo mismo. Unificamos a minusculas con formato consistente.
    """
    mapping = {
        "semestre_1": "semestre_1",
        "semestre1": "semestre_1",
        "SEMESTRE_1": "semestre_1",
        "Semestre_1": "semestre_1",
        "semestre_2": "semestre_2",
        "vacaciones_mitad": "vacaciones_mitad",
        "vacaciones_fin": "vacaciones_fin",
        "vacaciones": "vacaciones_fin",
    }

    before = df["periodo_academico"].nunique()
    df["periodo_academico"] = df["periodo_academico"].map(mapping).fillna(df["periodo_academico"])
    after = df["periodo_academico"].nunique()
    logger.info(f"  Periodos normalizados: {before} categorias → {after}")
    logger.info(f"  Distribucion: {df['periodo_academico'].value_counts().to_dict()}")
    return df


def impute_energy_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 3: Imputar nulos en columnas de energia por sector.
    Estrategia: mediana por grupo (sede + hora del dia).
    Esto respeta los patrones de consumo de cada sede en cada franja horaria.
    Ejemplo: un null en Tunja-Comedores a las 13:00 se llena con la mediana
    de lo que Tunja-Comedores consume tipicamente a las 13:00.
    """
    for col in ENERGY_COLS:
        nulls_before = df[col].isnull().sum()
        if nulls_before == 0:
            continue

        # Imputar con mediana por sede+hora
        medians = df.groupby(["sede", "hora"])[col].transform("median")
        df[col] = df[col].fillna(medians)

        # Si aun quedan nulos (grupo sin datos), usar mediana global de la sede
        remaining = df[col].isnull().sum()
        if remaining > 0:
            sede_medians = df.groupby("sede")[col].transform("median")
            df[col] = df[col].fillna(sede_medians)

        nulls_after = df[col].isnull().sum()
        logger.info(f"  {col}: {nulls_before} nulos → {nulls_after}")

    return df


def impute_context_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 4: Imputar nulos en temperatura, ocupacion y agua.
    - Temperatura: interpolacion lineal por sede (cambia gradualmente)
    - Ocupacion: mediana por sede+hora+dia_semana (patron repetitivo)
    - Agua: mediana por sede+hora (patron de uso similar al energetico)
    """
    # Temperatura - interpolacion temporal por sede
    nulls = df["temperatura_exterior_c"].isnull().sum()
    df["temperatura_exterior_c"] = df.groupby("sede")["temperatura_exterior_c"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    logger.info(f"  temperatura_exterior_c: {nulls} nulos → {df['temperatura_exterior_c'].isnull().sum()}")

    # Ocupacion - mediana por sede+hora+dia_semana
    nulls = df["ocupacion_pct"].isnull().sum()
    medians = df.groupby(["sede", "hora", "dia_semana"])["ocupacion_pct"].transform("median")
    df["ocupacion_pct"] = df["ocupacion_pct"].fillna(medians)
    # Fallback a mediana por sede+hora
    remaining = df["ocupacion_pct"].isnull().sum()
    if remaining > 0:
        medians2 = df.groupby(["sede", "hora"])["ocupacion_pct"].transform("median")
        df["ocupacion_pct"] = df["ocupacion_pct"].fillna(medians2)
    logger.info(f"  ocupacion_pct: {nulls} nulos → {df['ocupacion_pct'].isnull().sum()}")

    # Agua - mediana por sede+hora
    nulls = df["agua_litros"].isnull().sum()
    medians = df.groupby(["sede", "hora"])["agua_litros"].transform("median")
    df["agua_litros"] = df["agua_litros"].fillna(medians)
    remaining = df["agua_litros"].isnull().sum()
    if remaining > 0:
        sede_medians = df.groupby("sede")["agua_litros"].transform("median")
        df["agua_litros"] = df["agua_litros"].fillna(sede_medians)
    logger.info(f"  agua_litros: {nulls} nulos → {df['agua_litros'].isnull().sum()}")

    return df


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 5: Tratamiento de outliers con metodo IQR por sede+franja horaria.
    El consumo universitario es bimodal: bajo en la noche y alto en el dia.
    Si calculamos IQR global por sede, los picos diurnos legitimos salen
    como outliers. Por eso segmentamos en dos franjas:
    - Horario activo (7-21h entre semana): consumo alto esperado
    - Horario inactivo (noche + fines de semana): consumo bajo esperado
    Usamos IQR * 3 por franja+sede para solo atrapar anomalias reales.
    """
    cols_to_check = ENERGY_COLS + ["agua_litros", "potencia_total_kw"]
    total_clipped = 0

    # Crear mascara de horario activo (7-21h en dias de semana)
    horario_activo = (df["hora"].between(7, 21)) & (~df["es_fin_semana"])

    for col in cols_to_check:
        col_clipped = 0
        for sede in df["sede"].unique():
            for is_active in [True, False]:
                mask = (df["sede"] == sede) & (horario_activo == is_active)
                values = df.loc[mask, col]

                if values.empty:
                    continue

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:
                    # Distribucion sin varianza, usar percentil 99
                    upper = values.quantile(0.99)
                    lower = 0
                else:
                    upper = q3 + 3 * iqr
                    lower = max(0, q1 - 3 * iqr)

                outliers = ((values > upper) | (values < lower)).sum()
                if outliers > 0:
                    df.loc[mask, col] = values.clip(lower=lower, upper=upper)
                    col_clipped += outliers

        if col_clipped > 0:
            total_clipped += col_clipped
            logger.info(f"  {col}: {col_clipped} outliers recortados (por sede+franja)")

    logger.info(f"  Total valores recortados: {total_clipped}")
    return df


def recalculate_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 6: Recalcular energia_total_kwh y co2_kg.
    El total original no coincide con la suma de sectores en ~23k filas.
    Recalcular garantiza consistencia interna del dataset.
    """
    old_total = df["energia_total_kwh"].sum()
    df["energia_total_kwh"] = df[ENERGY_COLS].sum(axis=1).round(4)
    new_total = df["energia_total_kwh"].sum()
    logger.info(f"  energia_total_kwh recalculada (diff: {abs(old_total - new_total):,.2f} kWh)")

    # co2 = energia_total * factor de emision del SIN colombiano
    df["co2_kg"] = (df["energia_total_kwh"] * CO2_FACTOR).round(4)
    logger.info(f"  co2_kg recalculado con factor {CO2_FACTOR} kg CO2/kWh")

    return df


def validate(df: pd.DataFrame) -> None:
    """Validaciones finales del dataset limpio."""
    logger.info("=== VALIDACION FINAL ===")
    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Nulos restantes: {df.isnull().sum().sum()}")

    # Verificar que no haya negativos
    for col in ENERGY_COLS + ["energia_total_kwh", "co2_kg"]:
        negs = (df[col] < 0).sum()
        if negs > 0:
            logger.warning(f"  {col} aun tiene {negs} negativos!")

    # Verificar consistencia de total
    diff = abs(df["energia_total_kwh"] - df[ENERGY_COLS].sum(axis=1))
    logger.info(f"  Max diff energia_total vs suma sectores: {diff.max():.6f}")

    # Resumen por sede
    logger.info("  Consumo promedio por sede (kWh/h):")
    for sede in ["Tunja", "Duitama", "Sogamoso", "Chiquinquirá"]:
        mean = df[df["sede"] == sede]["energia_total_kwh"].mean()
        logger.info(f"    {sede}: {mean:.2f}")


def main():
    logger.info("=" * 60)
    logger.info("LIMPIEZA DE DATOS - EcoSentinel UPTC")
    logger.info("=" * 60)

    df = load_raw()

    logger.info("\n--- Paso 1: Corregir negativos ---")
    df = fix_negatives(df)

    logger.info("\n--- Paso 2: Normalizar periodo academico ---")
    df = normalize_periodo_academico(df)

    logger.info("\n--- Paso 3: Imputar nulos en energia ---")
    df = impute_energy_nulls(df)

    logger.info("\n--- Paso 4: Imputar nulos en contexto ---")
    df = impute_context_nulls(df)

    logger.info("\n--- Paso 5: Tratar outliers (IQR clip) ---")
    df = treat_outliers(df)

    logger.info("\n--- Paso 6: Recalcular totales ---")
    df = recalculate_totals(df)

    logger.info("\n")
    validate(df)

    # Guardar
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    logger.info(f"\nDataset limpio guardado: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    logger.info(f"Registros finales: {len(df):,}")


if __name__ == "__main__":
    main()
